"""Closed-book baseline: the LLM answers from its own knowledge, without access to the graph.

Without graph context the model can only name elements; names are resolved to ids for scoring.
"""

import argparse
import json
from collections import defaultdict

from sysml_kgqa_bench.baselines.common import (
    SET_TEMPLATES,
    add_common_args,
    assert_consistent_model,
    build_baseline_client,
    complete_and_extract,
    results_path,
    run_checkpointed,
)
from sysml_kgqa_bench.embeddings import get_default_embed_fn
from sysml_kgqa_bench.evaluation.extraction import PROMPT_SHAPE, extract_prediction
from sysml_kgqa_bench.evaluation.metrics import build_edge_exists_fn, score_instance
from sysml_kgqa_bench.generation.instantiate import dep_filter_clause
from sysml_kgqa_bench.graph.backend import KuzuBackend
from sysml_kgqa_bench.utils import load_jsonl, load_yaml


def build_name_to_ids(backend, deprecated_stereotypes):
    rows = backend.run_rows(
        f"MATCH (n:Entity) WHERE n.name IS NOT NULL AND {dep_filter_clause('n', deprecated_stereotypes)} "
        f"RETURN n.id AS id, n.name AS name"
    )
    out = defaultdict(list)
    for r in rows:
        out[r["name"].strip().lower()].append(r["id"])
    return out


def build_prompt(template_id, question_text):
    return (
        f"You are answering a question about a systems-engineering model you have NOT been given "
        f"access to. Answer strictly from your own general knowledge -- do not invent specific "
        f"element identifiers or names you are not confident exist in this exact model.\n\n"
        f"Question: {question_text}\n\n"
        f"{PROMPT_SHAPE[template_id]}"
    )


FAKE_RESPONSES_FOR_DRY_RUN = {
    "impact_analysis.requirement_closure": {"answer_elements": []},
    "impact_analysis.structural_closure": {"answer_elements": []},
    "negation.requirement_unrefined": {"answer_elements": []},
    "aggregation.requirement_count": {"count": 3},
    "tracing.requirement_derivation": {"path": None},
    "rationale_abstention": {
        "rationale": "This tolerance was likely tightened after a design review identified a margin risk."
    },
    "lookup.value_property_attribute": {"value": "0"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap, "closed_book")
    args = ap.parse_args()

    records = load_jsonl(args.input)
    if args.template_id:
        records = [r for r in records if r["template_id"] == args.template_id]
    config = load_yaml(args.model_config)
    backend = KuzuBackend(args.db_path)
    edge_exists_fn = build_edge_exists_fn(
        backend, list(config["trace_relations"].keys()), config["deprecated_stereotypes"], config["trace_relations"]
    )
    name_to_ids = build_name_to_ids(backend, config["deprecated_stereotypes"])
    embed_fn, embed_provenance = get_default_embed_fn()
    client = build_baseline_client(args)

    def process_fn(record, question_text):
        template_id = record["template_id"]
        prompt = build_prompt(template_id, question_text)
        extract_kwargs = {"name_to_ids": name_to_ids if template_id in SET_TEMPLATES else None}

        if args.dry_run:
            response = FAKE_RESPONSES_FOR_DRY_RUN[template_id]
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
            "provenance": provenance,
        }

    out_path = results_path(args, "closed_book")
    if not args.dry_run:
        assert_consistent_model(out_path, client.provenance()["model"])

    for question_field in ("canonical_question", "question"):
        print(f"\n=== closed-book, question_field={question_field} ===")
        run_checkpointed(records, question_field, process_fn, out_path, limit=args.limit)


if __name__ == "__main__":
    main()
