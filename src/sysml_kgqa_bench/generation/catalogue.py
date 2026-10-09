"""Derive an empirical schema catalogue (stereotypes and relations) from the instance graph.

For every stereotype: instance count, coverage, role (semantic, infrastructural, status) and subset-based
subsumption. For every relation: observed (source type, target type) pairs and degree statistics.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict

from sysml_kgqa_bench import paths

INFRASTRUCTURAL = {
    "Class",
    "HierarchyElement",
    "ObjectProperties",
    "Element",
    "NamedElement",
    "PackageableElement",
    "Package",
    "HyperlinkOwner",
    "Model",
    "Diagram",
    "Property",
    "TaggedValue",
    "Slot",
    "InstanceSpecification",
    "Constraint",
    "Dependency",
    "Generalization",
    "Association",
    "Aggregation",
    "View",
    "view",
    "Conform",
    "Expose",
    "DiagramInfo",
    "Diagram Description",
    "DiagramTable",
    "InstanceTable",
    "RequirementTable",
    "SpecificationTable",
    "GenericTable",
    "SequenceDiagramGeneratorConfig",
    "NestedUIConfig",
    "SimulationConfig",
    "Document",
    "ActivityDiagram",
    "BlockDefinitionDiagram",
    "ParametricDiagram",
    "StateMachineDiagram",
    "SequenceDiagram",
    "PackageDiagram",
    "InternalBlockDiagram",
    "ClassDiagram",
    "UseCaseDiagram",
    "Button",
    "GroupBox",
    "TextField",
    "Panel",
    "CheckBox",
    "Slider",
    "Text",
    "Label",
    "LegendItem",
    "Legend",
    "Frame",
    "UI",
    "TimeSeriesChart",
    "TimelineChart",
}
STATUS = {
    "deprecated",
    "deprecated_by_container_name",
    "VerificationStatus",
    "TODO_Owner",
    "deleted_by_text_marker",
    "non_normative_annotation",
}


def load_graph(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_stereotype_index(nodes):
    index = defaultdict(set)
    for n in nodes:
        for t in n.get("node_types") or []:
            index[t].add(n["id"])
    return index


MIN_CONFIDENT_SUBSET_SIZE = 20


def compute_subsumption(stereotype_index):
    """Stereotype A is a child of B if A's instance set is a strict subset of B's."""
    names = sorted(stereotype_index)
    sets = {n: stereotype_index[n] for n in names}
    parents = defaultdict(dict)
    children = defaultdict(dict)

    for a in names:
        sa = sets[a]
        if not sa:
            continue
        confidence = "high" if len(sa) >= MIN_CONFIDENT_SUBSET_SIZE else "low_small_sample"
        for b in names:
            if a == b:
                continue
            sb = sets[b]
            if len(sb) <= len(sa):
                continue
            if sa <= sb:
                parents[a][b] = confidence
                children[b][a] = confidence
    return parents, children


def most_specific_types(node_types, children):
    types = set(node_types)
    result = []
    for t in types:
        if any(children.get(t, {}).get(other) == "high" for other in types if other != t):
            continue
        result.append(t)
    return sorted(result)


def classify_role(name, coverage):
    if name in INFRASTRUCTURAL:
        return "infrastructural"
    if name in STATUS:
        return "status"
    if coverage >= 0.30:
        return "infrastructural"
    return "semantic"


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def degree_stats(values):
    if not values:
        return None
    return {
        "min": min(values),
        "p50": percentile(values, 0.50),
        "mean": round(statistics.mean(values), 2),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "n": len(values),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-json", default=str(paths.GRAPH_JSON))
    ap.add_argument("--provenance", default=str(paths.PROCESSED / "provenance.json"))
    ap.add_argument("--outdir", default=str(paths.GRAPH))
    args = ap.parse_args()

    data = load_graph(args.graph_json)
    nodes = data["nodes"]
    edges = data["edges"]
    node_by_id = {n["id"]: n for n in nodes}
    total_nodes = len(nodes)

    with open(args.provenance, encoding="utf-8") as f:
        provenance = json.load(f)
    graph_version = provenance.get("graph_version", "unknown")

    print(f"Loaded {total_nodes} nodes, {len(edges)} edges (graph_version={graph_version})")

    stereotype_index = build_stereotype_index(nodes)
    print(f"Indexed {len(stereotype_index)} distinct stereotypes")

    print("Computing subsumption lattice (pairwise subset check) ...")
    parents, children = compute_subsumption(stereotype_index)

    stereotypes_catalogue = {}
    for name, ids in stereotype_index.items():
        coverage = len(ids) / total_nodes
        stereotypes_catalogue[name] = {
            "instance_count": len(ids),
            "coverage": round(coverage, 6),
            "origin": "empirical",
            "role": classify_role(name, coverage),
            "subsumption_parents": [{"name": p, "confidence": c} for p, c in sorted(parents.get(name, {}).items())],
            "subsumption_children": [
                {"name": c_name, "confidence": c} for c_name, c in sorted(children.get(name, {}).items())
            ],
        }

    n_low_confidence_edges = sum(
        1
        for v in stereotypes_catalogue.values()
        for p in v["subsumption_parents"]
        if p["confidence"] == "low_small_sample"
    )
    n_high_confidence_edges = sum(
        1 for v in stereotypes_catalogue.values() for p in v["subsumption_parents"] if p["confidence"] == "high"
    )
    print(
        f"  subsumption edges: {n_high_confidence_edges} high-confidence, "
        f"{n_low_confidence_edges} low_small_sample (child < {MIN_CONFIDENT_SUBSET_SIZE} instances)"
    )

    n_semantic = sum(1 for v in stereotypes_catalogue.values() if v["role"] == "semantic")
    n_infra = sum(1 for v in stereotypes_catalogue.values() if v["role"] == "infrastructural")
    n_status = sum(1 for v in stereotypes_catalogue.values() if v["role"] == "status")
    print(f"  role split: {n_semantic} semantic, {n_infra} infrastructural, {n_status} status")

    def node_label(node_id):
        node = node_by_id.get(node_id)
        if node is None:
            return None
        types = node.get("node_types") or []
        semantic_types = [t for t in types if stereotypes_catalogue.get(t, {}).get("role") == "semantic"]
        if not semantic_types:
            return None
        specific = most_specific_types(semantic_types, children)
        return specific

    relation_instances = defaultdict(list)
    for e in edges:
        relation_instances[e["relation"]].append(e)

    relations_catalogue = {}
    for rel, instances in relation_instances.items():
        endpoint_counts = Counter()
        out_degree = defaultdict(Counter)
        in_degree = defaultdict(Counter)

        for e in instances:
            src_labels = node_label(e["source"]) or ["(non-semantic)"]
            tgt_labels = node_label(e["target"]) or ["(non-semantic)"]
            for sl in src_labels:
                out_degree[sl][e["source"]] += 1
            for tl in tgt_labels:
                in_degree[tl][e["target"]] += 1
            for sl in src_labels:
                for tl in tgt_labels:
                    endpoint_counts[(sl, tl)] += 1

        relations_catalogue[rel] = {
            "instance_count": len(instances),
            "origin": "empirical",
            "observed_endpoint_pairs": [
                {"source_type": sl, "target_type": tl, "count": c}
                for (sl, tl), c in sorted(endpoint_counts.items(), key=lambda x: -x[1])
            ],
            "out_degree_by_source_type": {t: degree_stats(list(counts.values())) for t, counts in out_degree.items()},
            "in_degree_by_target_type": {t: degree_stats(list(counts.values())) for t, counts in in_degree.items()},
        }

    print(f"Indexed {len(relations_catalogue)} relation types")

    EXCLUSION_STEREOTYPES = [
        "deprecated",
        "deprecated_by_container_name",
        "deleted_by_text_marker",
        "non_normative_annotation",
    ]
    exclusion_counts = {
        name: stereotypes_catalogue.get(name, {}).get("instance_count", 0) for name in EXCLUSION_STEREOTYPES
    }
    deprecated_semantics = {
        "decision": "Elements carrying any of these stereotypes are excluded from answer sets; none are removed "
        "from the graph.",
        "distractor_eligible": {
            "stereotypes": ["deprecated", "deprecated_by_container_name"],
            "note": "Superseded content that still carries meaningful text.",
        },
        "never_distractor_eligible": {
            "stereotypes": ["deleted_by_text_marker", "non_normative_annotation"],
            "note": "Deleted content or content that is not a requirement.",
        },
        "instance_counts": exclusion_counts,
        "instance_count": exclusion_counts["deprecated"],
        "coverage": stereotypes_catalogue.get("deprecated", {}).get("coverage", 0.0),
    }

    catalogue = {
        "graph_version": graph_version,
        "schema_origin": "empirical",
        "total_nodes": total_nodes,
        "total_edges": len(edges),
        "stereotypes": stereotypes_catalogue,
        "relations": relations_catalogue,
        "deprecated_semantics": deprecated_semantics,
    }

    out_path = f"{args.outdir}/catalogue.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalogue, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
