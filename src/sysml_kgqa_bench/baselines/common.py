"""Shared code for the LLM baselines: CLI options, graph serialization, LLM calls and the resumable runner."""

import json
import os
import re

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.evaluation.extraction import extract_prediction
from sysml_kgqa_bench.evaluation.metrics import EXTRACTION_FAILED
from sysml_kgqa_bench.llm import BACKENDS, build_llm_client
from sysml_kgqa_bench.utils import record_key

DEFAULT_MODEL = "qwen-3-5-397b-a17b"
# Disables Qwen3 thinking mode on vLLM-served endpoints; pass '{}' for other models.
DEFAULT_EXTRA_BODY = '{"chat_template_kwargs": {"enable_thinking": false}}'
SET_TEMPLATES = (
    "impact_analysis.requirement_closure",
    "impact_analysis.structural_closure",
    "negation.requirement_unrefined",
)


def add_common_args(ap, name):
    """CLI options shared by all LLM baselines."""
    ap.add_argument("--input", default=str(paths.VERBALIZED), help="verbalized benchmark")
    ap.add_argument("--db-path", default=str(paths.KUZU_DB))
    ap.add_argument("--model-config", default=str(paths.MODEL_CONFIG))
    ap.add_argument(
        "--outdir",
        default=None,
        help="default: data/results/<baseline> (data/results/dry_run/<baseline> with --dry-run)",
    )
    ap.add_argument("--backend", default="private", choices=BACKENDS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--extra-body", default=DEFAULT_EXTRA_BODY, help="JSON passed as extra_body to the API")
    ap.add_argument("--limit", type=int, default=None, help="only the first N instances per question form")
    ap.add_argument("--template-id", default=None, help="only instances of this template")
    ap.add_argument("--dry-run", action="store_true", help="use canned responses instead of LLM calls")


def results_path(args, name):
    """Path of the results file; dry runs never write into the released results."""
    outdir = args.outdir or str(paths.RESULTS / "dry_run" / name if args.dry_run else paths.RESULTS / name)
    os.makedirs(outdir, exist_ok=True)
    return f"{outdir}/results.jsonl"


def build_baseline_client(args):
    if args.dry_run:
        return None
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    return build_llm_client(args.backend, args.model, timeout=180, max_tokens=args.max_tokens, extra_body=extra_body)


def load_done_keys(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["instance_key"])
    return done


class MixedModelResults(RuntimeError):
    pass


def assert_consistent_model(out_path, expected_model):
    """Refuse to resume into a results file that was produced with a different model."""
    if not os.path.exists(out_path):
        return
    with open(out_path, encoding="utf-8") as f:
        models_seen = set()
        for line in f:
            if not line.strip():
                continue
            models_seen.add(json.loads(line).get("provenance", {}).get("model"))
    if models_seen - {expected_model}:
        raise MixedModelResults(
            f"{out_path} already contains results from {sorted(models_seen)}, but this run uses "
            f"{expected_model!r}. Use a different --outdir."
        )


# Fields rendered for every node whose types make it eligible, even if the value is None.
_DEFAULT_VALUE_ELIGIBLE_TYPES = {"ValueProperty", "Property"}
_RATIONALE_ELIGIBLE_TYPES = {"TMT Requirement"}


def linearize_subgraph(nodes, edges):
    """Serialize a subgraph as text; used identically by the oracle and G-Retriever baselines.

    nodes: list of {"id", "name", "node_types", optionally "default_value", "rationale"}.
    edges: list of {"from", "to", "relation"}.
    """
    lines = ["NODES:"]
    for n in nodes:
        node_types = n.get("node_types") or []
        types = ",".join(node_types)
        line = f'- {n["id"]} | name="{n.get("name")}" | types=[{types}]'
        if _DEFAULT_VALUE_ELIGIBLE_TYPES & set(node_types):
            line += f" | default_value={n.get('default_value')!r}"
        if _RATIONALE_ELIGIBLE_TYPES & set(node_types):
            line += f" | rationale={n.get('rationale')!r}"
        lines.append(line)
    lines.append("EDGES:")
    for e in edges:
        lines.append(f"- {e['from']} --{e['relation']}--> {e['to']}")
    return "\n".join(lines)


def local_name_maps(node_meta):
    """Name lookups for the nodes in a context: name -> [ids] and name -> first id."""
    name_to_ids = {}
    for nid, meta in node_meta.items():
        if meta.get("name"):
            name_to_ids.setdefault(meta["name"].strip().lower(), []).append(nid)
    return name_to_ids, {k: v[0] for k, v in name_to_ids.items()}


def gold_dry_run_response(record):
    """Canned response that returns the gold answer (for --dry-run of context-based baselines)."""
    template_id = record["template_id"]
    if template_id in SET_TEMPLATES:
        return {"answer_elements": record["answer"]}
    if template_id == "aggregation.requirement_count":
        return {"count": record["answer"]}
    if template_id == "lookup.value_property_attribute":
        return {"value": record["answer"]}
    if template_id == "rationale_abstention":
        return {"rationale": record["answer"] or "No rationale is recorded for this requirement."}
    if template_id == "tracing.requirement_derivation":
        wp, wr = record["answer"]["witness_path"], record["answer"]["witness_relations"]
        return {"path": [{"from": wp[i], "to": wp[i + 1], "relation": wr[i]} for i in range(len(wr))]}
    raise ValueError(template_id)


def complete_and_extract(client, prompt, temperature, template_id, extract_kwargs=None):
    """Call the LLM and extract a typed prediction; returns (predicted, extraction_meta, raw_response).

    If the response contains no parseable JSON, the prediction is EXTRACTION_FAILED, except for
    rationale_abstention where the raw text is scored as the answer.
    """
    extract_kwargs = extract_kwargs or {}

    def _attach_generation_meta(meta):
        finish_reason = getattr(client, "last_finish_reason", None)
        if finish_reason is not None:
            meta["finish_reason"] = finish_reason
            meta["content_chars"] = getattr(client, "last_content_chars", None)
            meta["reasoning_chars"] = getattr(client, "last_reasoning_chars", None)
        return meta

    try:
        response = client.complete_json(prompt, temperature)
        raw_response = json.dumps(response)
        predicted, meta = extract_prediction(template_id, response, **extract_kwargs)
        meta["extraction_failed"] = False
        meta = _attach_generation_meta(meta)
        return predicted, meta, raw_response
    except (ValueError, json.JSONDecodeError) as e:
        msg = str(e)
        snippet_match = re.search(r": (.*)$", msg)
        raw_snippet = snippet_match.group(1) if snippet_match else msg
        meta = {"extraction_failed": True, "reason": msg}
        meta = _attach_generation_meta(meta)
        if template_id == "rationale_abstention":
            predicted = raw_snippet
        else:
            predicted = EXTRACTION_FAILED
        return predicted, meta, raw_snippet


def _json_default(obj):
    if obj is EXTRACTION_FAILED:
        return repr(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_checkpointed(records, question_field, process_fn, out_path, limit=None):
    """Run process_fn(record, question_text) for every record and append the results to out_path.

    question_field is "canonical_question" or "question" (the paraphrase). Records already present in
    out_path are skipped, so interrupted runs can be resumed.
    """
    done = load_done_keys(out_path)
    todo = records if limit is None else records[:limit]
    n_done_this_run = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, record in enumerate(todo, start=1):
            key = record_key(record) + "|" + question_field
            if key in done:
                continue
            question_text = record["verbalization"][question_field]
            out = process_fn(record, question_text)
            out["instance_key"] = key
            out["question_field"] = question_field
            f.write(json.dumps(out, default=_json_default) + "\n")
            f.flush()
            n_done_this_run += 1
            if n_done_this_run % 10 == 0:
                print(f"  {n_done_this_run} done this run ({i}/{len(todo)} considered)", flush=True)
    print(
        f"Done. {n_done_this_run} new records written to {out_path} ({len(done)} were already there from a prior run)."
    )
