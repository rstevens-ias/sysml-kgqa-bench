"""G-Retriever baseline (retrieval only): PCST subgraph retrieval plus a frozen LLM, no GNN.

Requires the candidate pool from build_pcst_pool.py and its embeddings from embed_pcst_pool.py.
"""

import argparse
import json

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.baselines.common import (
    add_common_args,
    assert_consistent_model,
    build_baseline_client,
    complete_and_extract,
    gold_dry_run_response,
    local_name_maps,
    results_path,
    run_checkpointed,
)
from sysml_kgqa_bench.baselines.pcst import PCSTRetriever
from sysml_kgqa_bench.embeddings import get_default_embed_fn
from sysml_kgqa_bench.evaluation.extraction import PROMPT_SHAPE, extract_prediction
from sysml_kgqa_bench.evaluation.metrics import build_edge_exists_fn, score_instance
from sysml_kgqa_bench.graph.backend import KuzuBackend
from sysml_kgqa_bench.utils import load_jsonl, load_yaml, record_key


def build_prompt(template_id, question_text, context):
    return (
        f"You are answering a question about a systems-engineering model. Below is a subgraph "
        f"retrieved as likely relevant -- it may be incomplete or include irrelevant elements; use "
        f"only what's actually useful here, do not assume anything beyond it.\n\n"
        f"{context}\n\n"
        f"Question: {question_text}\n\n"
        f"{PROMPT_SHAPE[template_id]}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap, "gretriever")
    ap.add_argument("--pool", default=str(paths.PCST_POOL))
    ap.add_argument("--embedded", default=str(paths.PCST_EMBEDDINGS))
    ap.add_argument("--topk", type=int, default=3, help="PCST: number of top-scoring nodes with prizes")
    ap.add_argument("--topk-e", type=int, default=5, help="PCST: number of top-scoring edges with prizes")
    ap.add_argument("--cost-e", type=float, default=0.5, help="PCST: edge cost")
    args = ap.parse_args()

    records = load_jsonl(args.input)
    if args.template_id:
        records = [r for r in records if r["template_id"] == args.template_id]
    config = load_yaml(args.model_config)
    backend = KuzuBackend(args.db_path)
    edge_exists_fn = build_edge_exists_fn(
        backend, list(config["trace_relations"].keys()), config["deprecated_stereotypes"], config["trace_relations"]
    )
    embed_fn, embed_provenance = get_default_embed_fn()
    retriever = PCSTRetriever(args.pool, args.embedded)
    client = build_baseline_client(args)

    def process_fn(record, question_text):
        template_id = record["template_id"]
        context, node_meta = retriever.build_context(
            record_key(record), question_text, topk=args.topk, topk_e=args.topk_e, cost_e=args.cost_e
        )
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
        provenance["retrieval"] = {
            "topk": args.topk,
            "topk_e": args.topk_e,
            "cost_e": args.cost_e,
            "embedding": retriever.embed_provenance,
        }
        return {
            "template_id": template_id,
            "predicted": predicted,
            "raw_response": raw_response,
            "extraction_meta": extraction_meta,
            "scores": result,
            "context_chars": len(context),
            "n_retrieved_nodes": len(node_meta),
            "provenance": provenance,
        }

    out_path = results_path(args, "gretriever")
    if not args.dry_run:
        assert_consistent_model(out_path, client.provenance()["model"])

    for question_field in ("canonical_question", "question"):
        print(f"\n=== gretriever, question_field={question_field} ===")
        run_checkpointed(records, question_field, process_fn, out_path, limit=args.limit)


if __name__ == "__main__":
    main()
