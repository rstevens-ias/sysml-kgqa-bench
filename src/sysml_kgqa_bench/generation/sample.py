"""Stratified sampling of the enumerated candidates down to each template's target count."""

import argparse
import random
from collections import Counter, defaultdict

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.utils import TEMPLATE_FILES, load_jsonl, load_yaml, write_jsonl


def sample_stratum(records, target, rng, label):
    if target >= len(records):
        return list(records), f"{label}: {len(records)} available, target {target} -- take-all (pool-limited)"
    return rng.sample(records, target), f"{label}: sampled {target} of {len(records)}"


def reconcile_target(sampled, target_count, report_lines):
    achieved = len(sampled)
    if achieved != target_count:
        report_lines.append(
            f"  RECONCILED: achieved {achieved} vs declared target {target_count} "
            f"(diff {achieved - target_count:+d}) due to proportional bin rounding."
        )
    else:
        report_lines.append(f"  RECONCILED: achieved {achieved}, matches declared target exactly.")


def sample_template(candidates, meta, rng):
    """Sample one template's candidates according to its `target_stratify_by` setting."""
    template_id = meta["template_id"]
    target_count = meta.get("target_count")
    stratify_by = meta.get("target_stratify_by")
    report_lines = [f"=== {template_id} (pool {len(candidates)}, target {target_count}) ==="]

    if stratify_by is None:
        sampled, line = sample_stratum(candidates, target_count, rng, "all")
        report_lines.append(f"  {line}")
        reconcile_target(sampled, target_count, report_lines)
        return sampled, report_lines

    if stratify_by == "difficulty_band":
        band_counts = meta["target_band_counts"]
        by_band = defaultdict(list)
        for r in candidates:
            by_band[r["difficulty_coords"]["difficulty_band"]].append(r)
        sampled = []
        for band, target in band_counts.items():
            group = by_band.get(band, [])
            picked, line = sample_stratum(group, target, rng, band)
            sampled.extend(picked)
            report_lines.append(f"  {line}")
        reconcile_target(sampled, target_count, report_lines)
        return sampled, report_lines

    if stratify_by == "depth":
        depth_counts = meta["target_depth_counts"]
        by_depth = defaultdict(list)
        for r in candidates:
            by_depth[r["difficulty_coords"]["depth"]].append(r)
        sampled = []
        for depth, target in depth_counts.items():
            group = by_depth.get(depth, [])
            picked, line = sample_stratum(group, target, rng, f"depth={depth}")
            sampled.extend(picked)
            report_lines.append(f"  {line}")
        reconcile_target(sampled, target_count, report_lines)
        return sampled, report_lines

    if stratify_by == "answerable":
        answerable_counts = meta["target_answerable_counts"]
        by_answerable = defaultdict(list)
        for r in candidates:
            by_answerable[str(r["difficulty_coords"]["answerable"]).lower()].append(r)
        sampled = []
        for key, target in answerable_counts.items():
            group = by_answerable.get(key, [])
            picked, line = sample_stratum(group, target, rng, f"answerable={key}")
            sampled.extend(picked)
            report_lines.append(f"  {line}")
        reconcile_target(sampled, target_count, report_lines)
        return sampled, report_lines

    if stratify_by == "seed_type":
        seed_type_counts = meta["target_seed_type_counts"]
        by_seed_type = defaultdict(list)
        for r in candidates:
            by_seed_type[r["seed_type"]].append(r)
        sampled = []
        for st, target in seed_type_counts.items():
            group = by_seed_type.get(st, [])
            picked, line = sample_stratum(group, target, rng, st)
            sampled.extend(picked)
            report_lines.append(f"  {line}")
        reconcile_target(sampled, target_count, report_lines)
        return sampled, report_lines

    if stratify_by == "scope_unit_type":
        scope_targets = meta["target_scope_type_counts"]
        by_scope = defaultdict(list)
        for r in candidates:
            by_scope[r["difficulty_coords"]["scope_unit_type"]].append(r)
        sampled = []
        for scope_type, target in scope_targets.items():
            group = by_scope.get(scope_type, [])
            if scope_type != "header":
                picked, line = sample_stratum(group, target, rng, scope_type)
                sampled.extend(picked)
                report_lines.append(f"  {line}")
                continue
            in_band = [r for r in group if 2 <= r["difficulty_coords"]["strict_count"] <= 20]
            bins = {"2-5": [], "6-10": [], "11-20": []}
            for r in in_band:
                c = r["difficulty_coords"]["strict_count"]
                bins["2-5" if c <= 5 else "6-10" if c <= 10 else "11-20"].append(r)
            total_in_band = len(in_band)
            for bin_name, bin_group in bins.items():
                bin_target = round(target * len(bin_group) / total_in_band) if total_in_band else 0
                picked, line = sample_stratum(bin_group, bin_target, rng, f"header[{bin_name}]")
                sampled.extend(picked)
                report_lines.append(f"  {line}")
        reconcile_target(sampled, target_count, report_lines)
        return sampled, report_lines

    raise ValueError(f"unknown target_stratify_by: {stratify_by}")


def trivial_fraction_report(sampled_by_template):
    lines = ["\n=== Trivial-fraction check ==="]
    total_trivial, total_n = 0, 0
    for template_id, records in sampled_by_template.items():
        if not records or "difficulty_band" not in records[0].get("difficulty_coords", {}):
            continue
        bands = Counter(r["difficulty_coords"]["difficulty_band"] for r in records)
        n = len(records)
        trivial = bands.get("trivial", 0)
        total_trivial += trivial
        total_n += n
        lines.append(f"  {template_id}: {trivial}/{n} trivial ({trivial / n:.1%})")
    if total_n:
        lines.append(
            f"  COMBINED (classes with a difficulty_band field): "
            f"{total_trivial}/{total_n} trivial ({total_trivial / total_n:.1%})"
        )
    return lines


def easy_fraction_report(sampled_by_template):
    lines = ["\n=== Combined easy-fraction check (trivial band + lookup class) ==="]
    trivial_total = 0
    for records in sampled_by_template.values():
        if not records or "difficulty_band" not in records[0].get("difficulty_coords", {}):
            continue
        trivial_total += sum(1 for r in records if r["difficulty_coords"]["difficulty_band"] == "trivial")
    lookup_total = len(sampled_by_template.get("lookup.value_property_attribute", []))
    grand_total = sum(len(v) for v in sampled_by_template.values())
    easy_total = trivial_total + lookup_total
    lines.append(f"  trivial-band instances (T1/T3/T5): {trivial_total}")
    lines.append(f"  lookup class (T7): {lookup_total}")
    if grand_total:
        lines.append(f"  COMBINED: {easy_total}/{grand_total} easy ({easy_total / grand_total:.1%})")
    return lines


def abstention_balance_report(sampled_by_template):
    lines = []
    records = sampled_by_template.get("rationale_abstention")
    if not records:
        return lines
    n = len(records)
    n_true = sum(1 for r in records if r["difficulty_coords"].get("answerable"))
    n_false = n - n_true
    lines.append("\n=== Abstention balance check (rationale_abstention) ===")
    lines.append(f"  answerable=true: {n_true}/{n} ({n_true / n:.1%})")
    lines.append(f"  answerable=false: {n_false}/{n} ({n_false / n:.1%})")
    return lines


def annotate(sampled_by_template):
    """Add per-instance attributes computed on the sampled pool.

    T1/T3: out-degree of the seed and maximum out-degree in the proof subgraph.
    T6: whether an answerable instance shares its gold rationale text with another sampled instance.
    """
    for template_id in ("impact_analysis.requirement_closure", "impact_analysis.structural_closure"):
        for r in sampled_by_template.get(template_id, []):
            out_degree = Counter(e["from"] for e in r["proof_subgraph"] or [])
            r["seed_branching_factor"] = out_degree.get(r["bindings"]["X"], 0)
            r["max_internal_branching_factor"] = max(out_degree.values(), default=0)

    rationale_records = sampled_by_template.get("rationale_abstention", [])
    text_counts = Counter(r["answer"] for r in rationale_records if r.get("answerable"))
    for r in rationale_records:
        shared = bool(r.get("answerable")) and text_counts[r["answer"]] > 1
        r["boilerplate_shared"] = shared
        if not r.get("answerable"):
            r["t6_stratum"] = "unanswerable"
        else:
            r["t6_stratum"] = "answerable_boilerplate" if shared else "answerable_distinct"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=str(paths.CANDIDATES))
    ap.add_argument("--templates-dir", default=str(paths.TEMPLATES))
    ap.add_argument("--outdir", default=str(paths.QA))
    ap.add_argument("--seed", type=int, default=20260722)
    args = ap.parse_args()

    all_candidates = load_jsonl(args.candidates)
    by_template = defaultdict(list)
    for r in all_candidates:
        by_template[r["template_id"]].append(r)

    rng = random.Random(args.seed)
    sampled_by_template = {}
    all_report_lines = [f"Sampling with rng_seed={args.seed}\n"]

    for fname in TEMPLATE_FILES:
        meta = load_yaml(f"{args.templates_dir}/{fname}")
        template_id = meta["template_id"]
        candidates = by_template.get(template_id, [])
        sampled, report_lines = sample_template(candidates, meta, rng)
        sampled_by_template[template_id] = sampled
        all_report_lines.extend(report_lines)

    all_report_lines.extend(trivial_fraction_report(sampled_by_template))
    all_report_lines.extend(easy_fraction_report(sampled_by_template))
    all_report_lines.extend(abstention_balance_report(sampled_by_template))

    annotate(sampled_by_template)
    final_pool = [r for records in sampled_by_template.values() for r in records]
    total_pool = sum(len(v) for v in by_template.values())
    all_report_lines.append(
        f"\nTotal sampled pool: {len(final_pool)} (of {total_pool} enumerated across {len(TEMPLATE_FILES)} templates)"
    )

    write_jsonl(f"{args.outdir}/candidates_sampled.jsonl", final_pool)
    report_text = "\n".join(all_report_lines)
    print(report_text)
    with open(f"{args.outdir}/sampling_report.md", "w", encoding="utf-8") as f:
        f.write("# Sampling report\n\n```\n" + report_text + "\n```\n")


if __name__ == "__main__":
    main()
