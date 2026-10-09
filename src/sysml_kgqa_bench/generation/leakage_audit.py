"""Lexical leakage audit: overlap between the names substituted into a question and its gold answer."""

import argparse
import re
from collections import defaultdict

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.graph.backend import KuzuBackend
from sysml_kgqa_bench.utils import load_jsonl

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "are",
    "from",
    "into",
    "over",
    "under",
    "when",
    "what",
    "which",
    "shall",
    "will",
    "not",
    "requirement",
    "requirements",
}


def tokenize(name, min_len=4):
    """Lower-cased alphanumeric tokens of at least `min_len` characters, without stopwords."""
    if not name:
        return set()
    tokens = re.split(r"[^A-Za-z0-9]+", name.lower())
    return {t for t in tokens if len(t) >= min_len and t not in STOPWORDS}


def token_overlap(name_a, name_b):
    ta, tb = tokenize(name_a), tokenize(name_b)
    return bool(ta & tb)


def fetch_names(backend, ids):
    ids = [i for i in set(ids) if i]
    if not ids:
        return {}
    id_list = ", ".join(f"'{i}'" for i in ids)
    rows = backend.run_rows(f"MATCH (n:Entity) WHERE n.id IN [{id_list}] RETURN n.id AS id, n.name AS name")
    return {r["id"]: r["name"] for r in rows}


def audit_impact_analysis(records, backend, seed_key="seed_name"):
    all_answer_ids = [aid for r in records for aid in r["answer"]]
    names = fetch_names(backend, all_answer_ids)
    n_overlap = 0
    for r in records:
        seed_name = r.get(seed_key)
        answer_names = [names.get(aid) for aid in r["answer"]]
        if any(token_overlap(seed_name, an) for an in answer_names):
            n_overlap += 1
    return n_overlap, len(records)


def audit_structural_closure(records, backend):
    all_answer_ids = [aid for r in records for aid in r["answer"]]
    names = fetch_names(backend, all_answer_ids)
    n_overlap = 0
    for r in records:
        seed_name = r.get("seed_name")
        owning_block_id = r["difficulty_coords"].get("owning_block")
        substituted = [seed_name]
        if owning_block_id:
            ob_names = fetch_names(backend, [owning_block_id])
            substituted.append(ob_names.get(owning_block_id))
        answer_names = [names.get(aid) for aid in r["answer"]]
        if any(token_overlap(s, an) for s in substituted for an in answer_names):
            n_overlap += 1
    return n_overlap, len(records)


def audit_negation(records, backend):
    all_answer_ids = [aid for r in records for aid in r["answer"]]
    names = fetch_names(backend, all_answer_ids)
    n_overlap = 0
    for r in records:
        anchor = r.get("cluster_anchor_name")
        answer_names = [names.get(aid) for aid in r["answer"]]
        if any(token_overlap(anchor, an) for an in answer_names):
            n_overlap += 1
    return n_overlap, len(records)


def audit_aggregation(records):
    n_overlap = 0
    for r in records:
        scope_name = r.get("scope_name") or ""
        count_str = str(r["difficulty_coords"]["strict_count"])
        digit_runs = re.findall(r"\d+", scope_name)
        if count_str in digit_runs:
            n_overlap += 1
    return n_overlap, len(records)


def audit_tracing(records, backend):
    all_intermediate_ids = []
    for r in records:
        path = r["answer"]["witness_path"]
        all_intermediate_ids.extend(path[1:-1])
    names = fetch_names(backend, all_intermediate_ids)
    n_overlap = 0
    for r in records:
        path = r["answer"]["witness_path"]
        intermediate_names = [names.get(nid) for nid in path[1:-1]]
        if not intermediate_names:
            continue
        if any(
            token_overlap(r.get("source_name"), n) or token_overlap(r.get("target_name"), n) for n in intermediate_names
        ):
            n_overlap += 1
    return n_overlap, len(records)


def audit_rationale(records):
    n_overlap, n_checked = 0, 0
    for r in records:
        if not r["difficulty_coords"].get("answerable"):
            continue
        n_checked += 1
        if token_overlap(r.get("seed_name"), r["answer"]):
            n_overlap += 1
    return n_overlap, n_checked


def audit_lookup(records):
    n_overlap = 0
    for r in records:
        seed_name = r.get("seed_name") or ""
        owning_block_name = r["difficulty_coords"].get("owning_block_name") or ""
        answer_str = str(r["answer"])
        if answer_str and (answer_str in seed_name or answer_str in owning_block_name):
            n_overlap += 1
    return n_overlap, len(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=str(paths.CANDIDATES_SAMPLED))
    ap.add_argument("--db-path", default=str(paths.KUZU_DB))
    ap.add_argument("--outdir", default=str(paths.QA))
    args = ap.parse_args()

    backend = KuzuBackend(args.db_path)
    records = load_jsonl(args.candidates)
    by_template = defaultdict(list)
    for r in records:
        by_template[r["template_id"]].append(r)

    lines = ["# Leakage audit (sampled pool)\n"]
    lines.append(
        "Share of instances where a name substituted into the question shares a token (>= 4 characters, "
        "no stopwords) with the gold answer; for scalar answers (T4, T7), a direct substring match.\n"
    )

    checks = [
        ("impact_analysis.requirement_closure", lambda recs: audit_impact_analysis(recs, backend)),
        ("impact_analysis.structural_closure", lambda recs: audit_structural_closure(recs, backend)),
        ("negation.requirement_unrefined", lambda recs: audit_negation(recs, backend)),
        ("aggregation.requirement_count", audit_aggregation),
        ("tracing.requirement_derivation", lambda recs: audit_tracing(recs, backend)),
        ("rationale_abstention", audit_rationale),
        ("lookup.value_property_attribute", audit_lookup),
    ]

    for template_id, fn in checks:
        recs = by_template.get(template_id, [])
        if not recs:
            lines.append(f"{template_id}: 0 records, skipped")
            continue
        n_overlap, n_total = fn(recs)
        pct = n_overlap / n_total if n_total else 0.0
        lines.append(f"{template_id}: {n_overlap}/{n_total} ({pct:.1%}) soft lexical overlap")

    report = "\n".join(lines)
    print(report)
    with open(f"{args.outdir}/leakage_audit_report.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
