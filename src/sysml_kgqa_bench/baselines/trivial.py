"""Trivial floors per template (empty answer, majority value, whole component, always abstain).

No LLM calls; used as a sanity check that the metrics do not reward trivial strategies.
"""

import argparse
from collections import Counter, defaultdict

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.evaluation.metrics import _normalize_scalar, build_edge_exists_fn, score_instance
from sysml_kgqa_bench.generation.instantiate import (
    build_components,
    dep_filter_clause,
    load_relation_edges,
    load_structural_edges,
    resolve_type,
)
from sysml_kgqa_bench.graph.backend import KuzuBackend
from sysml_kgqa_bench.utils import load_jsonl, load_yaml


def component_members(uf, adj_undirected):
    out = defaultdict(list)
    for node in adj_undirected:
        out[uf.find(node)].append(node)
    return out


def avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def report(label, records, predict_fn, edge_exists_fn=None, embed_fn=None):
    print(f"\n--- {label} ({len(records)} instances) ---")
    agg = defaultdict(list)
    for r in records:
        predicted = predict_fn(r)
        result = score_instance(r, predicted, edge_exists_fn=edge_exists_fn, embed_fn=embed_fn)
        for k, v in result.items():
            if isinstance(v, (int, float)):
                agg[k].append(v)
    for k, vals in agg.items():
        print(f"  {k}: mean={avg(vals):.3f} (n={len(vals)})")
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(paths.VERBALIZED))
    ap.add_argument("--db-path", default=str(paths.KUZU_DB))
    ap.add_argument("--model-config", default=str(paths.MODEL_CONFIG))
    ap.add_argument("--templates-dir", default=str(paths.TEMPLATES))
    args = ap.parse_args()

    records = load_jsonl(args.input)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["template_id"]].append(r)

    config = load_yaml(args.model_config)
    backend = KuzuBackend(args.db_path)
    dep = config["deprecated_stereotypes"]

    for template_id in ("impact_analysis.requirement_closure", "impact_analysis.structural_closure"):
        recs = by_class[template_id]
        report(f"{template_id}: empty-set floor", recs, lambda r: [])

        if template_id == "impact_analysis.requirement_closure":
            edges, _ = load_relation_edges(backend, list(config["trace_relations"].keys()), dep)
        else:
            raw_edges, _ = load_structural_edges(backend, dep)
            edges = [(a, rel, b) for a, rel, b, *_ in raw_edges]
        uf, _, adj_undirected = build_components(edges)
        members = component_members(uf, adj_undirected)

        def predict_component(r, members=members, uf=uf):
            seed = r["bindings"]["X"]
            comp = uf.find(seed) if seed in uf.parent else None
            return [n for n in members.get(comp, []) if n != seed]

        report(f"{template_id}: 'all elements in seed's component' floor", recs, predict_component)

    neg_recs = by_class["negation.requirement_unrefined"]
    report("negation.requirement_unrefined: empty-set floor", neg_recs, lambda r: [])

    neg_meta = load_yaml(f"{args.templates_dir}/negation.requirement_unrefined.yaml")
    neg_relation = neg_meta["derived_from"]["relation"]
    neg_type = resolve_type(neg_meta, config)
    neg_edges, _ = load_relation_edges(backend, [neg_relation], dep)
    neg_uf, _, neg_adj = build_components(neg_edges)
    domain_rows = backend.run_rows(
        f"MATCH (n:Entity) WHERE '{neg_type}' IN n.node_types AND {dep_filter_clause('n', dep)} RETURN n.id AS id"
    )
    domain_ids = [row["id"] for row in domain_rows]
    domain_by_component = defaultdict(list)
    for nid in domain_ids:
        comp = neg_uf.find(nid) if nid in neg_uf.parent else f"singleton:{nid}"
        domain_by_component[comp].append(nid)

    def predict_all_in_scope(r):
        comp = r["bindings"]["component_id"]
        root = neg_uf.find(comp) if comp in neg_uf.parent else f"singleton:{comp}"
        return domain_by_component.get(root, [])

    report("negation.requirement_unrefined: 'all elements in scope' floor", neg_recs, predict_all_in_scope)

    agg_recs = by_class["aggregation.requirement_count"]
    modal_count = Counter(r["answer"] for r in agg_recs).most_common(1)[0][0]
    print(f"\nmodal aggregation count across the class: {modal_count}")
    report("aggregation.requirement_count: modal-count floor", agg_recs, lambda r: modal_count)

    trace_edge_exists_fn = build_edge_exists_fn(
        backend, list(config["trace_relations"].keys()), dep, config["trace_relations"]
    )
    trace_recs = by_class["tracing.requirement_derivation"]
    report(
        "tracing.requirement_derivation: 'no path' floor",
        trace_recs,
        lambda r: None,
        edge_exists_fn=trace_edge_exists_fn,
    )

    trace_relations = list(config["trace_relations"].keys())

    def predict_one_hop(r):
        source, target = r["bindings"]["source"], r["bindings"]["target"]
        for rel in trace_relations:
            if trace_edge_exists_fn(source, target, rel):
                return [{"from": source, "to": target, "relation": rel}]
        return None

    report(
        "tracing.requirement_derivation: 'one-hop edge if it exists' floor",
        trace_recs,
        predict_one_hop,
        edge_exists_fn=trace_edge_exists_fn,
    )

    rat_recs = by_class["rationale_abstention"]
    n_answerable = sum(1 for r in rat_recs if r["answerable"])
    print(
        f"\nrationale_abstention pool balance: {n_answerable}/{len(rat_recs)} answerable "
        f"({n_answerable / len(rat_recs):.1%})"
    )

    def bow_embed(text):
        return [1.0]

    report(
        "rationale_abstention: always-abstain floor",
        rat_recs,
        lambda r: "No rationale is recorded for this requirement.",
        embed_fn=bow_embed,
    )

    look_recs = by_class["lookup.value_property_attribute"]
    norm_counts = Counter(_normalize_scalar(r["answer"]) for r in look_recs)
    modal_normalized = norm_counts.most_common(1)[0][0]
    modal_value = next(r["answer"] for r in look_recs if _normalize_scalar(r["answer"]) == modal_normalized)
    print(
        f"\nmodal (normalized) lookup value across the class: {modal_normalized!r} "
        f"({norm_counts[modal_normalized]}/{len(look_recs)}), using literal value {modal_value!r}"
    )
    report("lookup.value_property_attribute: most-frequent-value floor", look_recs, lambda r: modal_value)


if __name__ == "__main__":
    main()
