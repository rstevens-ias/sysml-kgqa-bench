"""Build the TMT property graph from the processed JSONL files.

Every element that takes part in a relationship becomes a node typed by its SysML/UML class and stereotypes;
requirements, functions, blocks and interactions are enriched with attributes from the curated files.
Writes the graph as node-link JSON and GraphML, plus a schema summary in docs/graph_schema.md.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import networkx as nx

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.graph.extract import REPO_SHA

SCHEMA_VERSION = "1.0"


def load_jsonl(path):
    if not os.path.exists(path):
        print(f"  (skipping, not found: {path})", file=sys.stderr)
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def strip_prefix(t):
    if isinstance(t, str):
        if t.startswith("sysml."):
            return t[len("sysml.") :]
        if t.startswith("uml:"):
            return t[len("uml:") :]
    return t


PLACEHOLDER_TYPES = {None, "", "null", "null.null"}


def is_placeholder_type(t):
    return t in PLACEHOLDER_TYPES


def set_attrs(attrs, **kwargs):
    for k, v in kwargs.items():
        if v is not None:
            attrs[k] = v


def add_node_types(attrs, *types):
    existing = attrs.setdefault("node_types", [])
    for t in types:
        if t and not is_placeholder_type(t) and t not in existing:
            existing.append(t)


def ensure_node(G, node_id, node_types=None, name=None):
    if node_id not in G:
        G.add_node(node_id)
    attrs = G.nodes[node_id]
    if node_types:
        if isinstance(node_types, str):
            node_types = [node_types]
        add_node_types(attrs, *node_types)
    if name == node_id and attrs.get("name"):
        name = None
    set_attrs(attrs, name=name)
    return attrs


def build_base_graph(graph_edges, node_types_by_id):
    """Create one node per edge endpoint and one edge per relationship in graph_edges.jsonl."""
    G = nx.MultiDiGraph()

    def seed_types(node_id, fallback_type):
        info = node_types_by_id.get(node_id)
        if info:
            return [strip_prefix(info["type"])] + info.get("stereotypes", [])
        return [strip_prefix(fallback_type)]

    for e in graph_edges:
        sid, tid = e.get("source_id"), e.get("target_id")
        if sid:
            ensure_node(G, sid, node_types=seed_types(sid, e.get("source_type")), name=e.get("source_name"))
        if tid:
            ensure_node(G, tid, node_types=seed_types(tid, e.get("target_type")), name=e.get("target_name"))
        if sid and tid:
            G.add_edge(sid, tid, relation=e.get("relation"), category=e.get("category"))
    return G


def overlay_requirements(G, requirements):
    for r in requirements:
        attrs = ensure_node(G, r["id"], node_types="Requirement", name=r.get("name"))
        set_attrs(
            attrs,
            tmt_id=r.get("tmt_id"),
            legacy_id=r.get("legacy_id"),
            text=r.get("text"),
            rationale=r.get("rationale"),
            description=r.get("description"),
            acceptance_state=r.get("acceptance_state"),
            applicable_subsystems=r.get("applicable_subsystems"),
            se_notes=r.get("se_notes"),
            hierarchy_path=r.get("hierarchy_path"),
        )


def overlay_functions(G, functions):
    for f in functions:
        attrs = ensure_node(G, f["id"], node_types="Function", name=f.get("name"))
        set_attrs(attrs, hierarchy_path=f.get("hierarchy_path"), owner_block_id=f.get("owner_block_id"))
        for call in f.get("calls", []):
            called_id = call.get("called_id")
            if called_id:
                ensure_node(G, called_id, name=call.get("called_name"))
                G.add_edge(f["id"], called_id, relation="calls", category="curated")


def overlay_blocks(G, blocks):
    for b in blocks:
        attrs = ensure_node(G, b["id"], node_types="Block", name=b.get("name"))
        set_attrs(attrs, hierarchy_path=b.get("hierarchy_path"), layer=b.get("layer"), domain=b.get("domain"))
        for prop in b.get("properties", []):
            pid = prop.get("property_id")
            if not pid:
                continue
            pattrs = ensure_node(
                G, pid, node_types=strip_prefix(prop.get("property_kind")) or "Property", name=prop.get("property_name")
            )
            set_attrs(pattrs, typed_by_name=prop.get("typed_by_name"), default_value=prop.get("default_value"))
            G.add_edge(b["id"], pid, relation="has_property", category="curated")


def overlay_interactions(G, interactions):
    for i in interactions:
        attrs = ensure_node(G, i["id"], node_types="Interaction", name=i.get("name"))
        set_attrs(attrs, hierarchy_path=i.get("hierarchy_path"))
        for ll in i.get("lifelines", []):
            lid = ll.get("lifeline_id")
            if not lid:
                continue
            lattrs = ensure_node(G, lid, node_types="Lifeline")
            set_attrs(
                lattrs,
                represents_id=ll.get("represents_id"),
                represents_name=ll.get("represents_name"),
                typed_by_id=ll.get("typed_by_id"),
                typed_by_name=ll.get("typed_by_name"),
            )
            G.add_edge(i["id"], lid, relation="has_lifeline", category="curated")
        for msg in i.get("messages", []):
            sender, receiver = msg.get("sender_lifeline_id"), msg.get("receiver_lifeline_id")
            if sender and receiver:
                G.add_edge(
                    sender,
                    receiver,
                    relation="sends_message",
                    category="curated",
                    message_id=msg.get("message_id"),
                    message_name=msg.get("name"),
                    signal_id=msg.get("signal_id"),
                    signal_name=msg.get("signal_name"),
                    interaction_id=i["id"],
                )


def add_relations(G, relations):
    n_added = 0
    for r in relations:
        sid, tid = r.get("source_id"), r.get("target_id")
        if not (sid and tid):
            continue
        ensure_node(G, sid, node_types=strip_prefix(r.get("source_type")), name=r.get("source_name"))
        ensure_node(G, tid, node_types=strip_prefix(r.get("target_type")), name=r.get("target_name"))
        G.add_edge(sid, tid, relation=r["relation_type"], category="curated_trace")
        n_added += 1
    return n_added


def add_connections(G, connections):
    n_added = 0
    for c in connections:
        sid, tid = c.get("client_part_id"), c.get("supplier_part_id")
        if sid and tid:
            G.add_edge(sid, tid, relation="connects_to", category="curated")
            n_added += 1
    return n_added


def add_duration_constraints(G, owners):
    """Add each duration/time constraint as its own node linked to its owner and constrained elements."""
    n_constraints = 0
    for owner_id, rules in owners:
        for i, rule in enumerate(rules or []):
            dc_id = f"dc::{owner_id}::{i}"
            attrs = ensure_node(G, dc_id, node_types=strip_prefix(rule.get("rule_type")) or "DurationConstraint")
            set_attrs(attrs, min_value=rule.get("min_value"), max_value=rule.get("max_value"))
            G.add_edge(owner_id, dc_id, relation="has_duration_constraint", category="curated")
            n_constraints += 1
            for ce in rule.get("constrained_elements", []):
                ce_id = ce.get("id")
                if ce_id:
                    ensure_node(G, ce_id, name=ce.get("name"))
                    G.add_edge(dc_id, ce_id, relation="constrains", category="curated")
    return n_constraints


def assign_edge_ids(G):
    for i, (u, v, k) in enumerate(G.edges(keys=True)):
        G.edges[u, v, k]["edge_id"] = f"e{i}"


def sanitize_for_graphml(G):
    """Return a copy without None values and with list/dict attributes serialized as JSON strings."""
    H = G.copy()
    for _, data in H.nodes(data=True):
        for k in list(data.keys()):
            v = data[k]
            if v is None:
                del data[k]
            elif isinstance(v, (list, dict)):
                data[k] = json.dumps(v, ensure_ascii=False)
    for _, _, data in H.edges(data=True):
        for k in list(data.keys()):
            v = data[k]
            if v is None:
                del data[k]
            elif isinstance(v, (list, dict)):
                data[k] = json.dumps(v, ensure_ascii=False)
    return H


def write_schema_doc(path, G, repo_sha):
    from collections import Counter

    node_type_counts = Counter()
    n_multi_type_nodes = 0
    for _, d in G.nodes(data=True):
        types = d.get("node_types") or ["(untyped)"]
        if len(types) >= 2:
            n_multi_type_nodes += 1
        for t in types:
            node_type_counts[t] += 1
    edge_relation_counts = Counter(d.get("relation") for _, _, d in G.edges(data=True))
    edge_category_counts = Counter(d.get("category") for _, _, d in G.edges(data=True))

    lines = [
        "# TMT knowledge graph schema",
        "",
        f"Generated by `sysml_kgqa_bench.graph.build` from `data/processed/*.jsonl` (source commit `{repo_sha}`).",
        f"Schema version: `{SCHEMA_VERSION}`.",
        "",
        "Every element that appears in a relationship of the source model is a node. Nodes carry a list",
        "`node_types` (SysML/UML class plus all applied stereotypes), so type checks are membership tests,",
        "e.g. `WHERE 'Requirement' IN n.node_types`. Requirements, functions, blocks, interactions, lifelines",
        "and properties carry additional attributes from the curated files. `DurationConstraint`,",
        "`TimeConstraint`, `VerificationStatus` and `NestedConnectorEnd` nodes are created from the raw XMI.",
        "",
        f"{n_multi_type_nodes} of {G.number_of_nodes()} nodes ({100 * n_multi_type_nodes / G.number_of_nodes():.1f}%) carry 2+ types.",
        "",
        f"## Node types ({len(node_type_counts)} distinct, counted per stereotype application -- a multi-typed node is counted once per type it carries)",
        "",
        "| node_type | count |",
        "|---|---|",
    ]
    for t, c in node_type_counts.most_common():
        lines.append(f"| {t} | {c} |")

    lines += [
        "",
        f"## Edge relations ({len(edge_relation_counts)} distinct)",
        "",
        "| relation | count | category |",
        "|---|---|---|",
    ]
    relation_to_category = {}
    for _, _, d in G.edges(data=True):
        relation_to_category[d.get("relation")] = d.get("category")
    for rel, c in edge_relation_counts.most_common():
        lines.append(f"| {rel} | {c} | {relation_to_category.get(rel)} |")

    lines += [
        "",
        "## Edge categories",
        "",
        "| category | count | meaning |",
        "|---|---|---|",
        f"| hierarchy | {edge_category_counts.get('hierarchy', 0)} | `hasParent` containment from the source model. |",
        f"| semantic | {edge_category_counts.get('semantic', 0)} | All other relations from `graph_edges.jsonl` (typedBy, classifiedBy, constrainedElement, ...). |",
        f"| diagram_view | {edge_category_counts.get('diagram_view', 0)} | Elements shown on a diagram (`element`, `diagramConnector`). |",
        f"| curated | {edge_category_counts.get('curated', 0)} | Edges derived from the curated files (calls, has_property, has_lifeline, sends_message, connects_to, has_duration_constraint, constrains). |",
        f"| curated_trace | {edge_category_counts.get('curated_trace', 0)} | Trace relations from `relations.jsonl` (Refines, Refine, Expose, Allocate, Specifies, Abstraction, InterfaceRealization). |",
        "",
        f"Totals: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-dir", default=str(paths.PROCESSED))
    ap.add_argument("--outdir", default=str(paths.GRAPH))
    ap.add_argument("--schema-doc", default=str(paths.GRAPH_SCHEMA_DOC))
    ap.add_argument("--repo-sha", default=REPO_SHA, help="commit of Open-MBEE/TMT-SysML-Model")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pd = args.processed_dir

    print("Loading processed JSONL files ...", file=sys.stderr)
    graph_edges = load_jsonl(f"{pd}/graph_edges.jsonl")
    node_types_list = load_jsonl(f"{pd}/node_types.jsonl")
    node_types_by_id = {n["id"]: n for n in node_types_list}
    requirements = load_jsonl(f"{pd}/requirements.jsonl")
    functions = load_jsonl(f"{pd}/functions.jsonl")
    blocks = load_jsonl(f"{pd}/blocks.jsonl")
    interactions = load_jsonl(f"{pd}/interactions.jsonl")
    relations = load_jsonl(f"{pd}/relations.jsonl")
    connections = load_jsonl(f"{pd}/connections.jsonl")
    print(
        f"  {len(graph_edges)} graph_edges, {len(node_types_list)} node_types, {len(requirements)} requirements, "
        f"{len(functions)} functions, {len(blocks)} blocks, {len(interactions)} interactions, "
        f"{len(relations)} relations, {len(connections)} connections",
        file=sys.stderr,
    )

    print("Building base layer from graph_edges.jsonl ...", file=sys.stderr)
    G = build_base_graph(graph_edges, node_types_by_id)
    print(f"  base graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)

    print("Overlaying curated attributes + shortcut edges ...", file=sys.stderr)
    overlay_requirements(G, requirements)
    overlay_functions(G, functions)
    overlay_blocks(G, blocks)
    overlay_interactions(G, interactions)
    n_relations = add_relations(G, relations)
    n_connections = add_connections(G, connections)
    n_duration_owners = (
        [(f["id"], f.get("duration_constraints")) for f in functions if f.get("duration_constraints")]
        + [(b["id"], b.get("duration_constraints")) for b in blocks if b.get("duration_constraints")]
        + [(i["id"], i.get("duration_constraints")) for i in interactions if i.get("duration_constraints")]
    )
    n_duration_constraints = add_duration_constraints(G, n_duration_owners)
    print(
        f"  +{n_relations} trace relations, +{n_connections} connections, "
        f"+{n_duration_constraints} duration constraints (reified as nodes)",
        file=sys.stderr,
    )
    print(f"  final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)

    G.graph["schema_version"] = SCHEMA_VERSION
    G.graph["source_repo"] = "https://github.com/Open-MBEE/TMT-SysML-Model"
    G.graph["source_commit"] = args.repo_sha
    G.graph["generated_at"] = datetime.now(timezone.utc).isoformat()

    assign_edge_ids(G)

    print("Writing outputs ...", file=sys.stderr)
    graphml_path = f"{args.outdir}/tmt_graph.graphml"
    nx.write_graphml(sanitize_for_graphml(G), graphml_path, edge_id_from_attribute="edge_id")
    print(f"  {graphml_path}", file=sys.stderr)

    json_path = f"{args.outdir}/tmt_graph.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f, ensure_ascii=False)
    print(f"  {json_path}", file=sys.stderr)

    schema_path = args.schema_doc
    write_schema_doc(schema_path, G, args.repo_sha)
    print(f"  {schema_path}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
