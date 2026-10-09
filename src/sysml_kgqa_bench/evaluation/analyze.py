"""Aggregate the baseline results into the per-template tables reported in the paper."""

import argparse
import json
import statistics as st
from collections import Counter, defaultdict

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.evaluation.metrics import TEMPLATE_GUARANTEE
from sysml_kgqa_bench.utils import load_jsonl

LABELS = {
    "impact_analysis.requirement_closure": "T1 requirement_closure",
    "negation.requirement_unrefined": "T2 negation",
    "impact_analysis.structural_closure": "T3 structural_closure",
    "aggregation.requirement_count": "T4 count",
    "tracing.requirement_derivation": "T5 tracing",
    "rationale_abstention": "T6 abstention",
    "lookup.value_property_attribute": "T7 lookup",
}
BASELINES = ("oracle", "closed_book", "gretriever", "text_to_cypher")


def primary_metric(guarantee, scored):
    if not scored:
        return None
    if guarantee == "completeness":
        return st.mean(r["scores"]["completeness"] for r in scored)
    if guarantee in ("exactness_set", "exactness_scalar"):
        return st.mean(r["scores"]["exact_match"] for r in scored)
    if guarantee == "faithfulness_path":
        return st.mean(r["scores"]["endpoint_connection"] for r in scored)
    if guarantee == "abstention":
        return st.mean(r["scores"]["correct"] for r in scored)


def summarize(records):
    """Per-template primary metric plus secondary metrics, excluding extraction failures."""
    by_tid = defaultdict(list)
    for r in records:
        by_tid[r["template_id"]].append(r)
    out = {}
    for tid, guarantee in TEMPLATE_GUARANTEE.items():
        rs = by_tid.get(tid, [])
        n = len(rs)
        ef = sum(1 for r in rs if r["scores"].get("extraction_failed"))
        scored = [r for r in rs if not r["scores"].get("extraction_failed")]
        entry = {"n": n, "extraction_failed": ef, "primary": primary_metric(guarantee, scored)}
        if guarantee == "completeness":
            entry["faithfulness"] = st.mean(r["scores"]["faithfulness"] for r in scored) if scored else None
            entry["self_inclusion"] = (
                sum(1 for r in scored if r["scores"].get("self_inclusion")) / len(scored) if scored else None
            )
            entry["stopped_early"] = (
                sum(1 for r in scored if r["scores"].get("stopped_early")) / len(scored) if scored else None
            )
        if guarantee == "faithfulness_path":
            entry["edge_existence"] = st.mean(r["scores"]["edge_existence"] for r in scored) if scored else None
        if guarantee == "abstention":
            unans = [r for r in scored if r["scores"].get("metric") == "abstention_accuracy"]
            ans = [r for r in scored if r["scores"].get("metric") == "lasm_accuracy"]
            entry["abstention_accuracy"] = st.mean(r["scores"]["correct"] for r in unans) if unans else None
            entry["lasm_accuracy"] = st.mean(r["scores"]["correct"] for r in ans) if ans else None
        out[tid] = entry
    return out


def analyze(results_dir):
    baselines = {name: load_jsonl(f"{results_dir}/{name}/results.jsonl") for name in BASELINES}
    summaries = {name: summarize(recs) for name, recs in baselines.items()}
    return baselines, summaries


def print_report(results_dir):
    baselines, summaries = analyze(results_dir)
    order = BASELINES

    print("=" * 110)
    print("MAIN TABLE -- primary guarantee metric per template x baseline")
    print("=" * 110)
    print(
        f"{'Template':28s} {'guarantee':16s} {'oracle':>10s} {'closed_book':>12s} {'gretriever':>11s} {'text2cypher':>12s}"
    )
    for tid, guarantee in TEMPLATE_GUARANTEE.items():
        row = f"{LABELS[tid]:28s} {guarantee:16s}"
        for name in order:
            v = summaries[name][tid]["primary"]
            row += f" {v:10.3f}" if v is not None else f" {'--':>10s}"
        print(row)

    print("\n" + "=" * 110)
    print("EXTRACTION_FAILED rates")
    print("=" * 110)
    for name in order:
        for tid in TEMPLATE_GUARANTEE:
            e = summaries[name][tid]
            if e["n"] and e["extraction_failed"]:
                print(
                    f"  {name:16s} {LABELS[tid]:28s} extraction_failed={e['extraction_failed']}/{e['n']} "
                    f"({e['extraction_failed'] / e['n']:.1%})"
                )

    print("\n" + "=" * 110)
    print("T1/T3 completeness detail: faithfulness, self_inclusion, stopped_early")
    print("=" * 110)
    for tid in ("impact_analysis.requirement_closure", "impact_analysis.structural_closure"):
        print(f"\n-- {LABELS[tid]} --")
        for name in order:
            e = summaries[name][tid]
            if e["primary"] is None:
                continue
            print(
                f"  {name:16s} completeness={e['primary']:.3f}  faithfulness={e['faithfulness']:.3f}  "
                f"self_inclusion={e['self_inclusion']:.1%}  stopped_early={e['stopped_early']:.1%}"
            )

    print("\n" + "=" * 110)
    print("T5 detail: edge_existence vs endpoint_connection")
    print("=" * 110)
    for name in order:
        e = summaries[name]["tracing.requirement_derivation"]
        if e["primary"] is None:
            continue
        print(
            f"  {name:16s} edge_existence={e['edge_existence']:.3f}  endpoint_connection={e['primary']:.3f}  "
            f"extraction_failed={e['extraction_failed']}/{e['n']} ({e['extraction_failed'] / e['n']:.1%})"
        )

    print("\n" + "=" * 110)
    print("T6 detail: abstention_accuracy vs lasm_accuracy")
    print("=" * 110)
    for name in order:
        e = summaries[name]["rationale_abstention"]
        if e["abstention_accuracy"] is None:
            continue
        print(
            f"  {name:16s} abstention_accuracy={e['abstention_accuracy']:.3f}  lasm_accuracy={e['lasm_accuracy']:.3f}"
        )

    print("\n" + "=" * 110)
    print("SANITY CHECK: oracle >= every other baseline, per template, per primary metric")
    print("=" * 110)
    violations = []
    for tid, guarantee in TEMPLATE_GUARANTEE.items():
        ov = summaries["oracle"][tid]["primary"]
        if ov is None:
            continue
        for name in ("closed_book", "gretriever", "text_to_cypher"):
            v = summaries[name][tid]["primary"]
            if v is not None and v > ov + 1e-9:
                violations.append((name, tid, v, ov))
    if violations:
        for name, tid, v, ov in violations:
            print(f"  VIOLATION: {name} beats oracle on {tid}: {v:.3f} > {ov:.3f}")
    else:
        print("  none -- oracle is >= every other baseline on every template's primary metric.")

    gr = baselines["gretriever"]
    sizes = [r.get("n_retrieved_nodes") for r in gr if r.get("n_retrieved_nodes") is not None]
    if sizes:
        ctx = [r.get("context_chars") for r in gr if r.get("context_chars") is not None]
        print("\n" + "=" * 110)
        print("G-Retriever: retrieval size stats")
        print("=" * 110)
        print(
            f"  n_retrieved_nodes: n={len(sizes)} min={min(sizes)} median={st.median(sizes):.1f} "
            f"p90={st.quantiles(sizes, n=10)[8]:.1f} max={max(sizes)}"
        )
        if ctx:
            print(f"  context_chars: median={st.median(ctx):.0f} p90={st.quantiles(ctx, n=10)[8]:.0f} max={max(ctx)}")

    t2c = baselines["text_to_cypher"]
    if t2c:
        print("\n" + "=" * 110)
        print("text-to-Cypher: syntax_error / attempts_used")
        print("=" * 110)
        se = sum(1 for r in t2c if r.get("extraction_meta", {}).get("syntax_error"))
        print(f"  syntax_error (repair loop exhausted): {se}/{len(t2c)} ({se / len(t2c):.1%})")
        attempts = Counter(r.get("extraction_meta", {}).get("attempts_used") for r in t2c)
        print(f"  attempts_used distribution: {dict(sorted((k, v) for k, v in attempts.items() if k is not None))}")

    print("\n" + "=" * 110)
    print("Verbalizer bias (canonical vs paraphrase) -- gretriever & text_to_cypher")
    print("=" * 110)
    for name, recs in (("gretriever", gr), ("text_to_cypher", t2c)):
        if not recs:
            continue
        print(f"\n-- {name} --")
        by_tid = defaultdict(list)
        for r in recs:
            by_tid[r["template_id"]].append(r)
        for tid, guarantee in TEMPLATE_GUARANTEE.items():
            rs = by_tid.get(tid, [])
            for qf in ("canonical_question", "question"):
                sub = [r for r in rs if r["question_field"] == qf]
                scored = [r for r in sub if not r["scores"].get("extraction_failed")]
                v = primary_metric(guarantee, scored)
                if v is not None:
                    print(f"  {LABELS[tid]:28s} {qf:20s} n={len(scored):4d} primary={v:.3f}")


def to_json_summary(results_dir):
    """Flat {"baseline/template_id/metric": value} mapping, as in docs/paper_reported_results.json."""
    _, summaries = analyze(results_dir)
    flat = {}
    for name, by_tid in summaries.items():
        for tid, entry in by_tid.items():
            for metric, value in entry.items():
                if metric in ("n", "extraction_failed") or value is None:
                    continue
                flat[f"{name}/{tid}/{metric}"] = value
    return flat


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(paths.RESULTS))
    ap.add_argument("--json", action="store_true", help="print the flat metric mapping as JSON")
    args = ap.parse_args()
    if args.json:
        print(json.dumps(to_json_summary(args.results_dir), indent=2, sort_keys=True))
    else:
        print_report(args.results_dir)
