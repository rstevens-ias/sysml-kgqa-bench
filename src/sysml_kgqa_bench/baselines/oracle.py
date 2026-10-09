"""Oracle baseline: the LLM receives the gold proof subgraph of each instance as context.

The proof subgraph is joined from data/qa/candidates_sampled.jsonl and serialized with linearize_subgraph.
"""

import argparse
import json
import re

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.baselines.common import (
    add_common_args,
    assert_consistent_model,
    build_baseline_client,
    complete_and_extract,
    gold_dry_run_response,
    linearize_subgraph,
    local_name_maps,
    results_path,
    run_checkpointed,
)
from sysml_kgqa_bench.embeddings import get_default_embed_fn
from sysml_kgqa_bench.evaluation.extraction import PROMPT_SHAPE, extract_prediction
from sysml_kgqa_bench.evaluation.metrics import build_edge_exists_fn, score_instance
from sysml_kgqa_bench.graph.backend import KuzuBackend
from sysml_kgqa_bench.utils import load_jsonl, load_yaml, record_key


def fetch_node_meta(backend, ids):
    if not ids:
        return {}
    id_list = ", ".join(f"'{i}'" for i in set(ids) if i)
    rows = backend.run_rows(
        f"MATCH (n:Entity) WHERE n.id IN [{id_list}] RETURN n.id AS id, n.name AS name, n.node_types AS node_types, "
        f"n.default_value AS default_value, n.rationale AS rationale"
    )
    return {
        r["id"]: {
            "id": r["id"],
            "name": r["name"],
            "node_types": r["node_types"],
            "default_value": r["default_value"],
            "rationale": r["rationale"],
        }
        for r in rows
    }


def fetch_aggregation_population(backend, count_cypher):
    """Rewrite a T4 count query to return the counted requirements themselves."""
    match = re.search(r"RETURN\s+count\((\w+)\)\s+AS\s+\w+", count_cypher, re.IGNORECASE)
    if not match:
        raise RuntimeError(f"unexpected aggregation gold_query shape: {count_cypher!r}")
    var = match.group(1)
    population_cypher = count_cypher[: match.start()] + f"RETURN {var}.id AS id, {var}.name AS name"
    rows = backend.run_rows(population_cypher)
    return [{"id": r["id"], "name": r["name"], "node_types": ["TMT Requirement"]} for r in rows]


def build_context_for_record(raw_record, backend):
    """Serialize the gold proof subgraph (plus bound nodes) of one candidate."""
    template_id = raw_record["template_id"]
    proof_edges = raw_record.get("proof_subgraph") or []

    if template_id == "tracing.requirement_derivation" and not proof_edges:
        seen = set()
        for path in raw_record["answer"].get("all_paths", []):
            nodes, rels = path["nodes"], path["relations"]
            for i in range(len(rels)):
                key = (nodes[i], nodes[i + 1], rels[i])
                if key not in seen:
                    seen.add(key)
                    proof_edges.append({"from": nodes[i], "to": nodes[i + 1], "relation": rels[i]})

    node_ids = set()
    edges = []
    for e in proof_edges:
        node_ids.add(e["from"])
        node_ids.add(e["to"])
        edges.append({"from": e["from"], "to": e["to"], "relation": e["relation"]})
    for v in raw_record.get("bindings", {}).values():
        node_ids.add(v)

    aggregation_population = None
    if template_id == "aggregation.requirement_count":
        aggregation_population = fetch_aggregation_population(backend, raw_record["gold_query"]["cypher"])

    node_meta = fetch_node_meta(backend, node_ids)
    nodes = list(node_meta.values())
    if aggregation_population:
        nodes = nodes + aggregation_population
    context = linearize_subgraph(nodes, edges)

    extra = ""
    if template_id == "aggregation.requirement_count":
        n_shown = len(aggregation_population) if aggregation_population else 0
        extra = (
            f"\nScope: {raw_record.get('scope_name')}. The {n_shown} Requirement(s) meeting the "
            f"stated qualifiers within this scope are listed above in NODES -- count them."
        )

    return context + extra, node_meta


def build_prompt(template_id, question_text, context):
    return (
        f"You are answering a question about a systems-engineering model. Below is the exact "
        f"relevant subgraph -- use ONLY this information, do not assume anything beyond it.\n\n"
        f"{context}\n\n"
        f"Question: {question_text}\n\n"
        f"{PROMPT_SHAPE[template_id]}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap, "oracle")
    ap.add_argument("--candidates", default=str(paths.CANDIDATES_SAMPLED))
    args = ap.parse_args()

    records = load_jsonl(args.input)
    if args.template_id:
        records = [r for r in records if r["template_id"] == args.template_id]
    raw_by_key = {record_key(r): r for r in load_jsonl(args.candidates)}
    config = load_yaml(args.model_config)
    backend = KuzuBackend(args.db_path)
    edge_exists_fn = build_edge_exists_fn(
        backend, list(config["trace_relations"].keys()), config["deprecated_stereotypes"], config["trace_relations"]
    )
    embed_fn, embed_provenance = get_default_embed_fn()
    client = build_baseline_client(args)

    def process_fn(record, question_text):
        template_id = record["template_id"]
        raw_record = raw_by_key.get(record_key(record))
        if raw_record is None:
            raise RuntimeError(f"no candidate record with a proof subgraph for {record_key(record)}")
        context, node_meta = build_context_for_record(raw_record, backend)
        prompt = build_prompt(template_id, question_text, context)
        name_to_ids, name_to_id = local_name_maps(node_meta)
        extract_kwargs = {
            "name_to_ids": name_to_ids,
            "node_name_to_id_for_path": name_to_id,
            "known_ids": set(node_meta.keys()),
        }

        if args.dry_run:
            response = gold_dry_run_response(record)
            raw_response = json.dumps(response)
            predicted, extraction_meta = extract_prediction(template_id, response, **extract_kwargs)
            extraction_meta["extraction_failed"] = False
        else:
            predicted, extraction_meta, raw_response = complete_and_extract(
                client, prompt, 0.0, template_id, extract_kwargs=extract_kwargs
            )

        result = score_instance(record, predicted, edge_exists_fn=edge_exists_fn, embed_fn=embed_fn)
        provenance = {"dry_run": args.dry_run} if args.dry_run else client.provenance()
        if template_id == "rationale_abstention":
            provenance["embedding"] = embed_provenance
        return {
            "template_id": template_id,
            "predicted": predicted,
            "raw_response": raw_response,
            "extraction_meta": extraction_meta,
            "scores": result,
            "context_chars": len(context),
            "provenance": provenance,
        }

    out_path = results_path(args, "oracle")
    if not args.dry_run:
        assert_consistent_model(out_path, client.provenance()["model"])

    for question_field in ("canonical_question", "question"):
        print(f"\n=== oracle, question_field={question_field} ===")
        run_checkpointed(records, question_field, process_fn, out_path, limit=args.limit)


if __name__ == "__main__":
    main()
