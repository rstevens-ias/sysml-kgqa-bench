"""Enumerate all candidate instances of the seven question templates over the Kuzu graph.

Each candidate stores its bindings, gold answer, gold query and difficulty coordinates. Candidates that
violate a template's validity rules are written to rejections.csv instead.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.graph.backend import KuzuBackend
from sysml_kgqa_bench.utils import load_yaml, write_jsonl


def assert_bound_names_resolved(candidates, name_fields, template_id):
    """Fail if any candidate lacks a name for an entity referenced in its question."""
    missing = [(r["bindings"], f) for r in candidates for f in name_fields if not r.get(f)]
    if missing:
        raise RuntimeError(
            f"{template_id}: {len(missing)} candidate(s) have an unresolved name field ({name_fields}), "
            f"e.g. {missing[0]}."
        )


def difficulty_band(closure_size, max_depth):
    if closure_size == 1 and max_depth == 1:
        return "trivial"
    if closure_size <= 4:
        return "small"
    return "medium"


OVERSIZED_THRESHOLD = 500
NEAR_TOTAL_FRACTION = 0.6


def dep_filter_clause(alias, deprecated_stereotypes):
    """Cypher condition excluding nodes that carry any of the given stereotypes."""
    return " AND ".join(
        f"({alias}.node_types IS NULL OR NOT '{s}' IN {alias}.node_types)" for s in deprecated_stereotypes
    )


def is_deprecated(node_types, deprecated_stereotypes):
    return any(t in (node_types or []) for t in deprecated_stereotypes)


def resolve_type(template_meta, config):
    """Map a template's generic type role to the model-specific stereotype."""
    role = template_meta["domain_type_role"]
    return config["type_roles"].get(role, role)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def load_relation_edges(backend, relations, deprecated_stereotypes):
    """Return (edges, n_dropped): (source, relation, target) triples between non-deprecated nodes."""
    rows = backend.run_rows(
        """
        MATCH (a:Entity)-[r:Relation]->(b:Entity)
        WHERE r.relation IN $rels
        RETURN a.id AS aid, a.node_types AS atypes,
               r.relation AS rel,
               b.id AS bid, b.node_types AS btypes
        """,
        {"rels": relations},
    )
    edges = []
    n_dropped = 0
    for row in rows:
        if is_deprecated(row["atypes"], deprecated_stereotypes) or is_deprecated(row["btypes"], deprecated_stereotypes):
            n_dropped += 1
            continue
        edges.append((row["aid"], row["rel"], row["bid"]))
    return edges, n_dropped


def build_components(edges):
    """Weakly connected components of the edge set."""
    uf = UnionFind()
    adj_undirected = defaultdict(set)
    for aid, _rel, bid in edges:
        uf.union(aid, bid)
        adj_undirected[aid].add(bid)
        adj_undirected[bid].add(aid)
    component_size = Counter(uf.find(n) for n in adj_undirected)
    return uf, component_size, adj_undirected


def build_directed_adjacency(edges, direction_map):
    """Adjacency oriented in each relation's downstream direction ("forward" or "reverse")."""
    adj = defaultdict(list)
    for aid, rel, bid in edges:
        direction = direction_map[rel]
        if direction == "forward":
            adj[aid].append((bid, rel))
        else:
            adj[bid].append((aid, rel))
    return adj


def bfs_downstream_closure(adj, uf, component_size, seed_id):
    """All nodes reachable downstream of the seed, with depth, edge types and the BFS proof edges."""
    comp_id = uf.find(seed_id) if seed_id in uf.parent else f"singleton:{seed_id}"
    comp_size = component_size.get(comp_id, 1)

    if seed_id not in adj:
        return [], 0, Counter(), [], comp_id, comp_size

    depth = {seed_id: 0}
    proof_edges = []
    frontier = [seed_id]
    while frontier:
        next_frontier = []
        for node in frontier:
            for neighbor, rel in adj[node]:
                if neighbor not in depth:
                    depth[neighbor] = depth[node] + 1
                    proof_edges.append({"from": node, "to": neighbor, "relation": rel})
                    next_frontier.append(neighbor)
        frontier = next_frontier

    closure = [n for n in depth if n != seed_id]
    max_depth = max((depth[n] for n in closure), default=0)
    edge_types = Counter(e["relation"] for e in proof_edges)
    return closure, max_depth, edge_types, proof_edges, comp_id, comp_size


def hop_query_for(relation, direction, seed_id, deprecated_stereotypes):
    dep = dep_filter_clause("b" if direction == "forward" else "a", deprecated_stereotypes)
    if direction == "forward":
        cypher = (
            f"MATCH (a:Entity {{id: '{seed_id}'}})-[r:Relation {{relation: '{relation}'}}]->(b:Entity) "
            f"WHERE {dep} RETURN b.id AS id"
        )
    else:
        cypher = (
            f"MATCH (a:Entity)-[r:Relation {{relation: '{relation}'}}]->(b:Entity {{id: '{seed_id}'}}) "
            f"WHERE {dep} RETURN a.id AS id"
        )
    return cypher


def instantiate_impact_analysis(backend, meta, config):
    """T1: downstream closure of a requirement along the trace relations."""
    template_id = meta["template_id"]
    print(f"\n=== {template_id} ===")

    deprecated_stereotypes = config["deprecated_stereotypes"]
    downstream_direction = config["trace_relations"]
    trace_relations = list(downstream_direction.keys())
    requirement_type = resolve_type(meta, config)

    edges, n_dropped = load_relation_edges(backend, trace_relations, deprecated_stereotypes)
    uf, component_size, _ = build_components(edges)
    adj = build_directed_adjacency(edges, downstream_direction)
    print(
        f"  trace graph: {len(edges)} non-deprecated edges, {len(component_size)} components "
        f"({n_dropped} edges dropped for a deprecated endpoint)"
    )

    domain_rows = backend.run_rows(
        f"""
        MATCH (n:Entity)
        WHERE '{requirement_type}' IN n.node_types AND {dep_filter_clause("n", deprecated_stereotypes)}
        RETURN n.id AS id, n.name AS name
        """
    )
    print(f"  domain: {len(domain_rows)} non-deprecated {requirement_type} seeds")

    candidates, rejections = [], []
    n_invariant_violations = 0
    for row in domain_rows:
        seed_id, seed_name = row["id"], row["name"]
        closure, max_depth, edge_types, proof_edges, comp_id, comp_size = bfs_downstream_closure(
            adj, uf, component_size, seed_id
        )
        closure_size = len(closure)

        if closure_size > max(comp_size - 1, 0):
            n_invariant_violations += 1
            print(
                f"  WARNING: directed closure_size {closure_size} exceeds "
                f"component_size-1 ({comp_size - 1}) for {seed_id}"
            )

        gold_query = [
            {
                "relation": rel,
                "direction": direction,
                "cypher": hop_query_for(rel, direction, seed_id, deprecated_stereotypes),
            }
            for rel, direction in downstream_direction.items()
        ]

        record = {
            "template_id": template_id,
            "graph_version": meta["graph_version"],
            "sampling": "exhaustive",
            "rng_seed": None,
            "bindings": {"X": seed_id},
            "seed_name": seed_name,
            "gold_query": {
                "procedure": "directed_bfs_over_hops",
                "note": (
                    "Run each hop query against the current frontier (starting {X}), "
                    "union new ids into the next frontier, repeat until no new ids "
                    "appear. Literal ids substituted below for this record's seed."
                ),
                "hops": gold_query,
            },
            "answer": closure,
            "proof_subgraph": proof_edges,
            "difficulty_coords": {
                "closure_size": closure_size,
                "max_depth": max_depth,
                "edge_type_count": len(edge_types),
                "component_id": comp_id,
                "component_size": comp_size,
                "difficulty_band": difficulty_band(closure_size, max_depth),
            },
        }

        if closure_size == 0:
            rejections.append({**record, "reject_reason": "empty_answer"})
        elif closure_size > OVERSIZED_THRESHOLD:
            rejections.append({**record, "reject_reason": "oversized"})
        else:
            candidates.append(record)

    if n_invariant_violations:
        print(f"  ERROR: {n_invariant_violations} directed-closure invariant violations")

    if candidates:
        n_degenerate = sum(
            1
            for r in candidates
            if r["difficulty_coords"]["closure_size"] == r["difficulty_coords"]["component_size"] - 1
        )
        degenerate_fraction = n_degenerate / len(candidates)
        if degenerate_fraction > 0.9:
            raise RuntimeError(
                f"Degeneracy check failed: {n_degenerate}/{len(candidates)} candidates "
                f"({degenerate_fraction:.1%}) have closure_size == component_size - 1; check the relation directions."
            )
        print(
            f"  degeneracy check: {n_degenerate}/{len(candidates)} candidates ({degenerate_fraction:.1%}) "
            f"have closure_size == component_size - 1 (threshold 90%)"
        )

    verify_gold_query_equivalence(backend, candidates)
    assert_bound_names_resolved(candidates, ["seed_name"], template_id)

    return candidates, rejections


TRACING_TRIVIAL_MAX_DEPTH = 1


def find_all_simple_paths_from_source(adj, source, max_nodes_reachable=None):
    """Map every node reachable from `source` to the list of all simple paths leading to it."""
    paths_to = defaultdict(list)
    stack = [(source, [], {source})]
    while stack:
        node, path_so_far, visited = stack.pop()
        if path_so_far:
            paths_to[node].append(path_so_far)
        for neighbor, rel in adj.get(node, []):
            if neighbor not in visited:
                stack.append((neighbor, path_so_far + [(node, rel, neighbor)], visited | {neighbor}))
    return paths_to


def instantiate_tracing(backend, meta, config):
    """T5: (source, target) requirement pairs connected by a directed chain of trace relations."""
    template_id = meta["template_id"]
    print(f"\n=== {template_id} ===")

    deprecated_stereotypes = config["deprecated_stereotypes"]
    downstream_direction = config["trace_relations"]
    trace_relations = list(downstream_direction.keys())
    requirement_type = resolve_type(meta, config)

    edges, n_dropped = load_relation_edges(backend, trace_relations, deprecated_stereotypes)
    uf, component_size, _ = build_components(edges)
    adj = build_directed_adjacency(edges, downstream_direction)
    print(
        f"  trace graph: {len(edges)} non-deprecated edges, {len(component_size)} components "
        f"({n_dropped} edges dropped for a deprecated endpoint)"
    )

    cycle_free = _confirm_acyclic(adj)
    if not cycle_free:
        raise RuntimeError("The directed trace graph contains a cycle; simple-path enumeration requires a DAG.")
    print("  acyclic check: passed")

    domain_rows = backend.run_rows(
        f"""
        MATCH (n:Entity)
        WHERE '{requirement_type}' IN n.node_types AND {dep_filter_clause("n", deprecated_stereotypes)}
        RETURN n.id AS id, n.name AS name
        """
    )
    name_of = {r["id"]: r["name"] for r in domain_rows}
    requirement_ids = set(name_of.keys())
    print(f"  domain: {len(domain_rows)} non-deprecated {requirement_type} sources")

    candidates = []
    n_sources_with_targets = 0
    for row in domain_rows:
        source_id = row["id"]
        if source_id not in adj:
            continue
        paths_to = find_all_simple_paths_from_source(adj, source_id)
        if not paths_to:
            continue
        n_sources_with_targets += 1
        comp_id = uf.find(source_id) if source_id in uf.parent else f"singleton:{source_id}"
        comp_size = component_size.get(comp_id, 1)

        for target_id, all_paths in paths_to.items():
            shortest = min(all_paths, key=len)
            depth = len(shortest)
            witness_node_path = [source_id] + [hop[2] for hop in shortest]
            witness_relations = [hop[1] for hop in shortest]
            relation_type_count = len({hop[1] for path in all_paths for hop in path})

            gold_hops = [
                {
                    "from": hop[0],
                    "relation": hop[1],
                    "to": hop[2],
                    "cypher": _edge_existence_cypher(hop[0], hop[1], hop[2], downstream_direction[hop[1]]),
                }
                for hop in shortest
            ]

            record = {
                "template_id": template_id,
                "graph_version": meta["graph_version"],
                "sampling": "exhaustive",
                "rng_seed": None,
                "bindings": {"source": source_id, "target": target_id},
                "source_name": name_of.get(source_id),
                "target_name": name_of.get(target_id),
                "target_is_requirement": target_id in requirement_ids,
                "target_type": "Requirement",
                "gold_query": {
                    "procedure": "path_predicate_check",
                    "note": (
                        "A proposed path is correct if every asserted edge exists with the stated relation and "
                        "direction (edge existence) and the asserted edges connect `source` to `target` "
                        "(endpoint connection); both are reported separately. `witness_path` is one shortest "
                        "path, `all_paths` lists all valid paths. Each hop below checks the existence of one "
                        "edge of the witness path."
                    ),
                    "hops": gold_hops,
                },
                "answer": {
                    "witness_path": witness_node_path,
                    "witness_relations": witness_relations,
                    "all_paths": [
                        {"nodes": [source_id] + [hop[2] for hop in p], "relations": [hop[1] for hop in p]}
                        for p in all_paths
                    ],
                },
                "difficulty_coords": {
                    "depth": depth,
                    "alternative_path_count": len(all_paths),
                    "relation_type_count": relation_type_count,
                    "component_id": comp_id,
                    "component_size": comp_size,
                    "source_id": source_id,
                    "difficulty_band": "trivial"
                    if depth <= TRACING_TRIVIAL_MAX_DEPTH
                    else ("small" if depth == 2 else "medium"),
                },
            }
            candidates.append(record)

    print(f"  sources with >=1 reachable target: {n_sources_with_targets}")
    print(f"  total (source, target) pairs: {len(candidates)}")
    depth_dist = Counter(r["difficulty_coords"]["depth"] for r in candidates)
    print(f"  depth distribution: {dict(sorted(depth_dist.items()))}")
    alt_path_dist = Counter(r["difficulty_coords"]["alternative_path_count"] > 1 for r in candidates)
    print(
        f"  pairs with >1 alternative path: {alt_path_dist.get(True, 0)}/{len(candidates)} "
        f"({alt_path_dist.get(True, 0) / len(candidates):.1%})"
    )

    n_cross_layer = sum(1 for r in candidates if not r["target_is_requirement"])
    print(f"  cross-layer targets (non-requirement elements): {n_cross_layer}/{len(candidates)}")

    from sysml_kgqa_bench.generation.leakage_audit import fetch_names

    missing_ids = {r["bindings"]["target"] for r in candidates if r["target_name"] is None}
    if missing_ids:
        broader_names = fetch_names(backend, missing_ids)
        id_list = ", ".join(f"'{i}'" for i in missing_ids)
        type_rows = backend.run_rows(
            f"MATCH (n:Entity) WHERE n.id IN [{id_list}] RETURN n.id AS id, n.node_types AS node_types"
        )
        types_of = {r["id"]: (r["node_types"] or ["Element"])[0] for r in type_rows}
        for r in candidates:
            if r["target_name"] is None:
                r["target_name"] = broader_names.get(r["bindings"]["target"])
                r["target_type"] = types_of.get(r["bindings"]["target"], "Element")

    assert_bound_names_resolved(candidates, ["source_name", "target_name"], template_id)

    verify_tracing_gold_query_equivalence(backend, candidates)

    return candidates, []


def _confirm_acyclic(adj):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)

    def dfs(u):
        color[u] = GRAY
        for v, _rel in adj.get(u, []):
            if color[v] == GRAY:
                return False
            if color[v] == WHITE and not dfs(v):
                return False
        color[u] = BLACK
        return True

    return all(dfs(node) for node in list(adj.keys()) if color[node] == WHITE)


def _edge_existence_cypher(from_id, relation, to_id, direction):
    return (
        f"MATCH (a:Entity {{id: '{from_id if direction == 'forward' else to_id}'}})"
        f"-[r:Relation {{relation: '{relation}'}}]->"
        f"(b:Entity {{id: '{to_id if direction == 'forward' else from_id}'}}) "
        f"RETURN count(r) > 0 AS edge_exists"
    )


def verify_tracing_gold_query_equivalence(backend, candidates, sample_size=40, seed=20260722, max_queries=400):
    if not candidates:
        return
    rng = random.Random(seed)
    sample = rng.sample(candidates, min(sample_size, len(candidates)))

    n_checked, n_mismatch, n_queries_used = 0, 0, 0
    for r in sample:
        if n_queries_used >= max_queries:
            break
        all_exist = True
        for hop in r["gold_query"]["hops"]:
            n_queries_used += 1
            rows = backend.run_rows(hop["cypher"])
            if not (rows and rows[0]["edge_exists"]):
                all_exist = False
        n_checked += 1
        if not all_exist:
            n_mismatch += 1
            print(
                f"  MISMATCH source={r['bindings']['source']} target={r['bindings']['target']}: "
                f"stored witness_path hop failed to verify against the live graph"
            )

    print(
        f"  gold_query equivalence check: {n_checked - n_mismatch}/{n_checked} sampled candidates' "
        f"stored witness_path hops all verified present ({n_queries_used} queries used)"
    )
    if n_mismatch:
        raise RuntimeError(
            f"tracing gold_query equivalence check failed: {n_mismatch}/{n_checked} sampled "
            f"candidates' stored witness_path could not be fully re-verified against the live graph."
        )


def verify_gold_query_equivalence(backend, candidates, sample_per_size=1, seed=12345, max_queries=800):
    """Re-execute the stored hop queries for a sample of closures and compare with the stored answers."""
    if not candidates:
        return
    by_size = defaultdict(list)
    for r in candidates:
        by_size[r["difficulty_coords"]["closure_size"]].append(r)
    rng = random.Random(seed)
    sample = []
    for size, group in by_size.items():
        k = min(sample_per_size, len(group))
        sample.extend(rng.sample(group, k))
    sample.sort(key=lambda r: r["difficulty_coords"]["closure_size"])

    n_mismatch = 0
    n_checked = 0
    n_queries_used = 0
    for r in sample:
        if n_queries_used >= max_queries:
            break
        seed_id = r["bindings"]["X"]
        stored_answer = set(r["answer"])
        depth = {seed_id: 0}
        frontier = [seed_id]
        while frontier:
            next_frontier = []
            for node in frontier:
                for hop in r["gold_query"]["hops"]:
                    cypher = hop["cypher"].replace(f"'{seed_id}'", f"'{node}'")
                    n_queries_used += 1
                    for row in backend.run_rows(cypher):
                        nid = row["id"]
                        if nid not in depth:
                            depth[nid] = depth[node] + 1
                            next_frontier.append(nid)
            frontier = next_frontier
        recomputed = {n for n in depth if n != seed_id}
        n_checked += 1
        if recomputed != stored_answer:
            n_mismatch += 1
            print(
                f"  MISMATCH seed={seed_id}: stored {len(stored_answer)} vs "
                f"recomputed {len(recomputed)} via literal gold_query hops"
            )

    n_skipped = len(sample) - n_checked
    print(
        f"  gold_query equivalence check: {n_checked - n_mismatch}/{n_checked} sampled "
        f"candidates' stored `answer` reproduced by re-executing their literal `gold_query` "
        f"hops (spanning {len(by_size)} distinct closure sizes, {n_queries_used} queries used"
        f"{f', {n_skipped} candidates skipped once the {max_queries}-query budget was spent' if n_skipped else ''})"
    )
    if n_mismatch:
        raise RuntimeError(
            f"gold_query equivalence check failed: {n_mismatch}/{n_checked} sampled candidates' "
            f"gold_query hops do not reproduce the stored answer."
        )


HASPARENT_STRUCTURAL_CHILD_TYPES = {
    "PartProperty",
    "Port",
    "ProxyPort",
    "Connector",
    "BindingConnector",
    "Generalization",
    "Signal",
    "FlowProperty",
    "ConstraintProperty",
}
HASPARENT_STRUCTURAL_PARENT_TYPES = {"Block", "InterfaceBlock"}
STRUCTURAL_SEED_TYPES = ["Block", "ValueProperty", "PartProperty"]


def load_structural_edges(backend, deprecated_stereotypes):
    """Structural edges (has_property, containment, typedBy, connects_to) with their traversal direction."""
    edges = []
    n_dropped = 0

    hp_rows = backend.run_rows(
        "MATCH (a:Entity)-[r:Relation {relation: 'has_property'}]->(b:Entity) "
        "RETURN a.id AS aid, a.node_types AS atypes, b.id AS bid, b.node_types AS btypes"
    )
    hp_pairs = set()
    for row in hp_rows:
        hp_pairs.add((row["aid"], row["bid"]))
        if is_deprecated(row["atypes"], deprecated_stereotypes) or is_deprecated(row["btypes"], deprecated_stereotypes):
            n_dropped += 1
            continue
        edges.append((row["aid"], "has_property", row["bid"], "bidirectional"))

    hasparent_rows = backend.run_rows(
        "MATCH (a:Entity)-[r:Relation {relation: 'hasParent'}]->(b:Entity) "
        "RETURN a.id AS aid, a.node_types AS atypes, b.id AS bid, b.node_types AS btypes"
    )
    for row in hasparent_rows:
        at, bt = set(row["atypes"] or []), set(row["btypes"] or [])
        if not (at & HASPARENT_STRUCTURAL_CHILD_TYPES and bt & HASPARENT_STRUCTURAL_PARENT_TYPES):
            continue
        if (row["aid"], row["bid"]) in hp_pairs:
            continue
        if is_deprecated(row["atypes"], deprecated_stereotypes) or is_deprecated(row["btypes"], deprecated_stereotypes):
            n_dropped += 1
            continue
        edges.append((row["aid"], "hasParent_structural", row["bid"], "reverse"))

    typedby_rows = backend.run_rows(
        "MATCH (a:Entity)-[r:Relation {relation: 'typedBy'}]->(b:Entity) "
        "RETURN a.id AS aid, a.node_types AS atypes, b.id AS bid, b.node_types AS btypes"
    )
    for row in typedby_rows:
        if is_deprecated(row["atypes"], deprecated_stereotypes) or is_deprecated(row["btypes"], deprecated_stereotypes):
            n_dropped += 1
            continue
        edges.append((row["aid"], "typedBy", row["bid"], "reverse"))

    connects_rows = backend.run_rows(
        "MATCH (a:Entity)-[r:Relation {relation: 'connects_to'}]->(b:Entity) "
        "RETURN a.id AS aid, a.node_types AS atypes, b.id AS bid, b.node_types AS btypes"
    )
    for row in connects_rows:
        if is_deprecated(row["atypes"], deprecated_stereotypes) or is_deprecated(row["btypes"], deprecated_stereotypes):
            n_dropped += 1
            continue
        edges.append((row["aid"], "connects_to", row["bid"], "bidirectional"))

    return edges, n_dropped


def build_structural_adjacency(edges):
    adj = defaultdict(list)
    for aid, rel, bid, direction in edges:
        if direction == "forward":
            adj[aid].append((bid, rel))
        elif direction == "reverse":
            adj[bid].append((aid, rel))
        else:
            adj[aid].append((bid, rel))
            adj[bid].append((aid, rel))
    return adj


def type_membership_clause(alias, types):
    return "(" + " OR ".join(f"'{t}' IN {alias}.node_types" for t in sorted(types)) + ")"


def structural_hop_queries_for(
    rel_key, direction, raw_relation, seed_id, deprecated_stereotypes, child_type_filter=None, parent_type_filter=None
):
    def forward_cypher():
        dep = dep_filter_clause("b", deprecated_stereotypes)
        extra = f" AND {type_membership_clause('b', child_type_filter)}" if child_type_filter else ""
        extra += f" AND {type_membership_clause('a', parent_type_filter)}" if parent_type_filter else ""
        return (
            f"MATCH (a:Entity {{id: '{seed_id}'}})-[r:Relation {{relation: '{raw_relation}'}}]->(b:Entity) "
            f"WHERE {dep}{extra} RETURN b.id AS id"
        )

    def reverse_cypher():
        dep = dep_filter_clause("a", deprecated_stereotypes)
        extra = f" AND {type_membership_clause('a', child_type_filter)}" if child_type_filter else ""
        extra += f" AND {type_membership_clause('b', parent_type_filter)}" if parent_type_filter else ""
        return (
            f"MATCH (a:Entity)-[r:Relation {{relation: '{raw_relation}'}}]->(b:Entity {{id: '{seed_id}'}}) "
            f"WHERE {dep}{extra} RETURN a.id AS id"
        )

    if direction == "forward":
        return [forward_cypher()]
    elif direction == "reverse":
        return [reverse_cypher()]
    else:
        return [forward_cypher(), reverse_cypher()]


def bfs_structural_closure(adj, uf, component_size, seed_id):
    comp_id = uf.find(seed_id) if seed_id in uf.parent else f"singleton:{seed_id}"
    comp_size = component_size.get(comp_id, 1)
    if seed_id not in adj:
        return [], 0, Counter(), [], comp_id, comp_size
    depth = {seed_id: 0}
    proof_edges = []
    frontier = [seed_id]
    while frontier:
        next_frontier = []
        for node in frontier:
            for neighbor, rel in adj[node]:
                if neighbor not in depth:
                    depth[neighbor] = depth[node] + 1
                    proof_edges.append({"from": node, "to": neighbor, "relation": rel})
                    next_frontier.append(neighbor)
        frontier = next_frontier
    closure = [n for n in depth if n != seed_id]
    max_depth = max((depth[n] for n in closure), default=0)
    edge_types = Counter(e["relation"] for e in proof_edges)
    return closure, max_depth, edge_types, proof_edges, comp_id, comp_size


def find_owning_block(backend, seed_id, seed_type, proof_edges):
    """The block that owns a value or part property (the seed itself for blocks)."""
    if seed_type == "Block":
        return seed_id
    if seed_type == "ValueProperty":
        for e in proof_edges:
            if e["from"] == seed_id and e["relation"] == "has_property":
                return e["to"]
        return None
    rows = backend.run_rows(
        f"MATCH (a:Entity {{id: '{seed_id}'}})-[r:Relation {{relation: 'hasParent'}}]->(b:Entity) "
        f"WHERE 'Block' IN b.node_types OR 'InterfaceBlock' IN b.node_types "
        f"RETURN b.id AS id"
    )
    return rows[0]["id"] if rows else None


def instantiate_structural_closure(backend, meta, config):
    """T3: structural closure of a block, value property or part property."""
    template_id = meta["template_id"]
    print(f"\n=== {template_id} ===")

    deprecated_stereotypes = config["deprecated_stereotypes"]
    structural_relations = config["structural_relations"]
    raw_relation_for = {
        "has_property": "has_property",
        "hasParent_structural": "hasParent",
        "typedBy": "typedBy",
        "connects_to": "connects_to",
    }

    edges, n_dropped = load_structural_edges(backend, deprecated_stereotypes)
    uf, component_size, _ = build_components([(a, r, b) for a, r, b, _d in edges])
    adj = build_structural_adjacency(edges)
    print(
        f"  structural graph: {len(edges)} non-deprecated edges, {len(component_size)} components "
        f"({n_dropped} edges dropped for a deprecated endpoint)"
    )

    candidates, rejections = [], []
    for seed_type in STRUCTURAL_SEED_TYPES:
        domain_rows = backend.run_rows(
            f"""
            MATCH (n:Entity)
            WHERE '{seed_type}' IN n.node_types AND {dep_filter_clause("n", deprecated_stereotypes)}
            RETURN n.id AS id, n.name AS name
            """
        )
        print(f"  domain: {len(domain_rows)} non-deprecated {seed_type} seeds")

        n_seed_type_candidates = 0
        for row in domain_rows:
            seed_id, seed_name = row["id"], row["name"]
            closure, max_depth, edge_types, proof_edges, comp_id, comp_size = bfs_structural_closure(
                adj, uf, component_size, seed_id
            )
            closure_size = len(closure)
            owning_block = find_owning_block(backend, seed_id, seed_type, proof_edges)

            gold_query = []
            for rel_key, rel_conf in structural_relations.items():
                child_filter = HASPARENT_STRUCTURAL_CHILD_TYPES if rel_key == "hasParent_structural" else None
                parent_filter = HASPARENT_STRUCTURAL_PARENT_TYPES if rel_key == "hasParent_structural" else None
                for cypher in structural_hop_queries_for(
                    rel_key,
                    rel_conf["direction"],
                    raw_relation_for[rel_key],
                    seed_id,
                    deprecated_stereotypes,
                    child_type_filter=child_filter,
                    parent_type_filter=parent_filter,
                ):
                    gold_query.append({"relation": rel_key, "direction": rel_conf["direction"], "cypher": cypher})

            record = {
                "template_id": template_id,
                "graph_version": meta["graph_version"],
                "sampling": "exhaustive",
                "rng_seed": None,
                "bindings": {"X": seed_id},
                "seed_name": seed_name,
                "seed_type": seed_type,
                "gold_query": {
                    "procedure": "structural_bfs_over_hops",
                    "note": (
                        "Run each hop query against the current frontier (starting {X}), "
                        "union new ids into the next frontier, repeat until no new ids "
                        "appear. Bidirectional relations contribute two hop cyphers, both "
                        "run every round. Literal ids substituted below for this record's seed."
                    ),
                    "hops": gold_query,
                },
                "answer": closure,
                "proof_subgraph": proof_edges,
                "difficulty_coords": {
                    "closure_size": closure_size,
                    "max_depth": max_depth,
                    "edge_type_count": len(edge_types),
                    "component_id": comp_id,
                    "component_size": comp_size,
                    "seed_type": seed_type,
                    "owning_block": owning_block,
                    "difficulty_band": difficulty_band(closure_size, max_depth),
                },
            }

            if closure_size == 0:
                rejections.append({**record, "reject_reason": "empty_answer"})
            elif closure_size > OVERSIZED_THRESHOLD:
                rejections.append({**record, "reject_reason": "oversized"})
            else:
                candidates.append(record)
                n_seed_type_candidates += 1
        print(f"    {seed_type}: {n_seed_type_candidates} candidates")

    if candidates:
        n_degenerate = sum(
            1
            for r in candidates
            if r["difficulty_coords"]["closure_size"] == r["difficulty_coords"]["component_size"] - 1
        )
        degenerate_fraction = n_degenerate / len(candidates)
        if degenerate_fraction > 0.9:
            raise RuntimeError(
                f"Degeneracy check failed: {n_degenerate}/{len(candidates)} candidates "
                f"({degenerate_fraction:.1%}) have closure_size == component_size - 1; check the relation directions."
            )
        print(
            f"  degeneracy check: {n_degenerate}/{len(candidates)} candidates ({degenerate_fraction:.1%}) "
            f"have closure_size == component_size - 1 (threshold 90%)"
        )
        by_seed_type = defaultdict(lambda: [0, 0])
        for r in candidates:
            st = r["difficulty_coords"]["seed_type"]
            by_seed_type[st][1] += 1
            if r["difficulty_coords"]["closure_size"] == r["difficulty_coords"]["component_size"] - 1:
                by_seed_type[st][0] += 1
        for st, (n_deg, n_tot) in by_seed_type.items():
            print(f"    {st}: {n_deg}/{n_tot} ({n_deg / n_tot:.1%}) degenerate")

    verify_gold_query_equivalence(backend, candidates)
    assert_bound_names_resolved(candidates, ["seed_name"], template_id)

    return candidates, rejections


NEAR_TOTAL_SCOPE_FRACTION = 0.6


def build_hasparent_forest(backend):
    rows = backend.run_rows(
        "MATCH (a:Entity)-[r:Relation {relation: 'hasParent'}]->(b:Entity) RETURN a.id AS aid, b.id AS bid"
    )
    children_of = defaultdict(list)
    for row in rows:
        children_of[row["bid"]].append(row["aid"])
    return children_of


TMT_DOC_ROOT_ANCHOR_NAME = "TMT Requirements imported from DOORS"


def root_document_of(backend, node_id, cache):
    """Name of the top-level imported requirements document that contains the node."""
    if "_anchor_id" not in cache:
        rows = backend.run_rows(f"MATCH (n:Entity {{name: '{TMT_DOC_ROOT_ANCHOR_NAME}'}}) RETURN n.id AS id")
        cache["_anchor_id"] = rows[0]["id"] if rows else None
    anchor_id = cache["_anchor_id"]
    if anchor_id is None:
        return None

    current = node_id
    seen = set()
    while current is not None and current not in seen:
        seen.add(current)
        if current not in cache:
            rows = backend.run_rows(
                f"MATCH (a:Entity {{id: '{current}'}})-[r:Relation {{relation: 'hasParent'}}]->(b:Entity) "
                f"RETURN b.id AS id, b.name AS name"
            )
            cache[current] = (rows[0]["id"], rows[0]["name"]) if rows else (None, None)
        parent_id, _parent_name = cache[current]
        if parent_id is None:
            return None
        if parent_id == anchor_id:
            name_rows = backend.run_rows(f"MATCH (n:Entity {{id: '{current}'}}) RETURN n.name AS name")
            return name_rows[0]["name"] if name_rows else None
        current = parent_id
    return None


def scope_descendants(children_of, root):
    seen = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(children_of.get(n, []))
    seen.discard(root)
    return seen


def instantiate_aggregation(backend, meta, config):
    """T4: number of active requirements below a package or document section."""
    template_id = meta["template_id"]
    print(f"\n=== {template_id} ===")

    always_exclude = config["always_exclude_stereotypes"]
    retired_but_real = config["retired_but_real_stereotypes"]
    requirement_type = resolve_type(meta, config)

    children_of = build_hasparent_forest(backend)

    req_rows = backend.run_rows(
        f"MATCH (n:Entity) WHERE '{requirement_type}' IN n.node_types RETURN n.id AS id, n.node_types AS nt"
    )
    req_types = {r["id"]: set(r["nt"] or []) for r in req_rows}
    domain_size = sum(
        1 for t in req_types.values() if not (t & set(always_exclude)) and not (t & set(retired_but_real))
    )
    print(f"  requirement domain (all exclusions applied): {domain_size}")

    scope_specs = [
        ("package", "MATCH (n:Entity) WHERE 'Package' IN n.node_types RETURN n.id AS id, n.name AS name"),
        (
            "header",
            "MATCH (n:Entity) WHERE 'non_normative_annotation' IN n.node_types RETURN n.id AS id, n.name AS name",
        ),
    ]

    root_doc_cache = {}
    candidates, rejections = [], []
    for scope_unit_type, query in scope_specs:
        scope_rows = backend.run_rows(query)
        print(f"  scope_unit_type={scope_unit_type}: {len(scope_rows)} candidate scopes enumerated")

        n_kept = 0
        for row in scope_rows:
            scope_id, scope_name = row["id"], row["name"]
            if not scope_name:
                continue
            descendants = scope_descendants(children_of, scope_id)
            req_descendants = [d for d in descendants if d in req_types]
            if not req_descendants:
                continue

            raw_count = len(req_descendants)
            bc_excluded_count = sum(1 for d in req_descendants if not (req_types[d] & set(always_exclude)))
            strict_count = sum(
                1
                for d in req_descendants
                if not (req_types[d] & set(always_exclude)) and not (req_types[d] & set(retired_but_real))
            )
            b_only_count = sum(1 for d in req_descendants if "deleted_by_text_marker" not in req_types[d])
            c_only_count = sum(1 for d in req_descendants if "non_normative_annotation" not in req_types[d])
            matters_a = bc_excluded_count != strict_count
            matters_b = raw_count != b_only_count
            matters_c = raw_count != c_only_count
            qualifier_count = int(matters_a) + int(matters_b) + int(matters_c)
            root_document = root_document_of(backend, scope_id, root_doc_cache)

            record = {
                "template_id": template_id,
                "graph_version": meta["graph_version"],
                "sampling": "exhaustive",
                "rng_seed": None,
                "bindings": {"scope": scope_id},
                "scope_name": scope_name,
                "scope_unit_type": scope_unit_type,
                "gold_query": {
                    "procedure": "hasparent_subtree_count",
                    "note": (
                        "Descendants of {scope} via hasParent that are "
                        f"'{requirement_type}'-typed and carry none of model_config.yaml's "
                        "always_exclude_stereotypes or retired_but_real_stereotypes. "
                        "Literal scope id substituted below."
                    ),
                    "cypher": (
                        f"MATCH (d:Entity)-[r:Relation* (r, n | WHERE r.relation = 'hasParent')]->"
                        f"(s:Entity {{id: '{scope_id}'}}) "
                        f"WHERE '{requirement_type}' IN d.node_types "
                        f"AND {dep_filter_clause('d', always_exclude + retired_but_real)} "
                        f"RETURN count(d) AS n"
                    ),
                },
                "answer": strict_count,
                "difficulty_coords": {
                    "raw_count": raw_count,
                    "strict_count": strict_count,
                    "scope_unit_type": scope_unit_type,
                    "matters_a": matters_a,
                    "matters_b": matters_b,
                    "matters_c": matters_c,
                    "qualifier_count": qualifier_count,
                    "root_document": root_document,
                },
            }

            if strict_count == 0:
                rejections.append({**record, "reject_reason": "empty_answer"})
            elif strict_count >= NEAR_TOTAL_SCOPE_FRACTION * domain_size:
                rejections.append({**record, "reject_reason": "near_total"})
            else:
                candidates.append(record)
                n_kept += 1
        print(f"    {scope_unit_type}: {n_kept} candidates")

    qc_dist = Counter(r["difficulty_coords"]["qualifier_count"] for r in candidates)
    print(f"  qualifier_count distribution: {dict(sorted(qc_dist.items()))}")

    root_doc_dist = Counter(r["difficulty_coords"]["root_document"] for r in candidates)
    print(f"  root_document distribution: {dict(root_doc_dist.most_common())}")

    header_candidates = [r for r in candidates if r["difficulty_coords"]["scope_unit_type"] == "header"]
    if header_candidates:
        hp_rows = backend.run_rows(
            "MATCH (n:Entity) WHERE 'non_normative_annotation' IN n.node_types "
            "RETURN n.id AS id, n.hierarchy_path AS hp"
        )
        hp_of = {r["id"]: r["hp"] for r in hp_rows}
        ids = [r["bindings"]["scope"] for r in header_candidates]
        n_ancestor_pairs = 0
        n_unordered_pairs = len(ids) * (len(ids) - 1) // 2
        for i in range(len(ids)):
            hp_i = hp_of.get(ids[i])
            if not hp_i:
                continue
            for j in range(len(ids)):
                if i == j:
                    continue
                hp_j = hp_of.get(ids[j])
                if hp_j and hp_j.startswith(hp_i + " > "):
                    n_ancestor_pairs += 1
        print(
            f"  header nesting check: {n_ancestor_pairs}/{n_unordered_pairs} unordered pairs "
            f"({n_ancestor_pairs / n_unordered_pairs:.1%}) are ancestor/descendant"
        )

    verify_aggregation_gold_query_equivalence(backend, candidates)
    assert_bound_names_resolved(candidates, ["scope_name"], template_id)

    return candidates, rejections


def verify_aggregation_gold_query_equivalence(backend, candidates, sample_per_type=15, seed=20260722, max_queries=200):
    if not candidates:
        return
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for r in candidates:
        by_type[r["difficulty_coords"]["scope_unit_type"]].append(r)
    sample = []
    for scope_type, group in by_type.items():
        k = min(sample_per_type, len(group))
        sample.extend(rng.sample(group, k))

    n_checked, n_mismatch, n_queries_used = 0, 0, 0
    for r in sample:
        if n_queries_used >= max_queries:
            break
        n_queries_used += 1
        rows = backend.run_rows(r["gold_query"]["cypher"])
        recomputed = rows[0]["n"] if rows else 0
        n_checked += 1
        if recomputed != r["answer"]:
            n_mismatch += 1
            print(f"  MISMATCH scope={r['bindings']['scope']}: stored {r['answer']} vs recomputed {recomputed}")

    n_skipped = len(sample) - n_checked
    print(
        f"  gold_query equivalence check: {n_checked - n_mismatch}/{n_checked} sampled candidates' "
        f"stored `answer` reproduced by re-executing their literal `gold_query` cypher "
        f"({n_queries_used} queries used"
        f"{f', {n_skipped} candidates skipped once the {max_queries}-query budget was spent' if n_skipped else ''})"
    )
    if n_mismatch:
        raise RuntimeError(
            f"aggregation gold_query equivalence check failed: {n_mismatch}/{n_checked} sampled "
            f"candidates' gold_query does not reproduce the stored answer."
        )


def instantiate_negation_scoped(backend, meta, config):
    """T2: requirements within a Refines cluster that are not refined by another requirement."""
    template_id = meta["template_id"]
    print(f"\n=== {template_id} ===")

    deprecated_stereotypes = config["deprecated_stereotypes"]
    type_name = resolve_type(meta, config)
    relation = meta["derived_from"]["relation"]
    direction = "outgoing" if config["trace_relations"][relation] == "forward" else "incoming"

    edges, n_dropped = load_relation_edges(backend, [relation], deprecated_stereotypes)
    uf, component_size, _ = build_components(edges)
    print(
        f"  {relation} graph: {len(edges)} non-deprecated edges, {len(component_size)} components "
        f"({n_dropped} edges dropped for a deprecated endpoint)"
    )

    domain_rows = backend.run_rows(
        f"""
        MATCH (n:Entity)
        WHERE '{type_name}' IN n.node_types AND {dep_filter_clause("n", deprecated_stereotypes)}
        RETURN n.id AS id, n.name AS name, n.hierarchy_path AS hierarchy_path
        """
    )
    domain_ids = [r["id"] for r in domain_rows]
    name_of = {r["id"]: r["name"] for r in domain_rows}
    hierarchy_path_of = {r["id"]: r["hierarchy_path"] for r in domain_rows}
    print(f"  domain: {len(domain_ids)} non-deprecated {type_name} elements")

    has_outgoing = defaultdict(set)
    has_incoming = defaultdict(set)
    for aid, rel, bid in edges:
        has_outgoing[aid].add(bid)
        has_incoming[bid].add(aid)
    ref_map = has_outgoing if direction == "outgoing" else has_incoming

    by_component = defaultdict(list)
    for nid in domain_ids:
        comp_id = uf.find(nid) if nid in uf.parent else f"singleton:{nid}"
        by_component[comp_id].append(nid)

    print(f"  {len(by_component)} components contain >=1 non-deprecated {type_name}")

    candidates, rejections = [], []
    for comp_id, members in by_component.items():
        domain_size = len(members)
        unrefined = [nid for nid in members if not ref_map.get(nid)]
        answer_size = len(unrefined)

        anchor_pool = [nid for nid in members if nid not in set(unrefined)] or members

        def _depth(nid):
            hp = hierarchy_path_of.get(nid)
            return len(hp.split(" > ")) if hp else float("inf")

        anchor_id = min(anchor_pool, key=lambda nid: (_depth(nid), name_of.get(nid) or ""))
        cluster_anchor_name = name_of.get(anchor_id)

        member_list = ", ".join(f"'{m}'" for m in members)
        edge_pattern = (
            "(t)-[r:Relation {{relation: '{rel}'}}]->(u:Entity)"
            if direction == "outgoing"
            else "(u:Entity)-[r:Relation {{relation: '{rel}'}}]->(t)"
        )
        edge_pattern = edge_pattern.format(rel=relation)
        gold_query = (
            f"MATCH (t:Entity) WHERE '{type_name}' IN t.node_types AND {dep_filter_clause('t', deprecated_stereotypes)} "
            f"AND t.id IN [{member_list}] "
            f"AND NOT EXISTS {{ MATCH {edge_pattern} "
            f"WHERE {dep_filter_clause('u', deprecated_stereotypes)} }} "
            f"RETURN t.id AS id, t.name AS name"
        )

        record = {
            "template_id": template_id,
            "graph_version": meta["graph_version"],
            "sampling": "exhaustive",
            "rng_seed": None,
            "bindings": {"component_id": comp_id},
            "cluster_anchor_name": cluster_anchor_name,
            "gold_query": gold_query,
            "answer": unrefined,
            "proof_subgraph": None,
            "difficulty_coords": {
                "answer_size": answer_size,
                "domain_size": domain_size,
                "answer_fraction": round(answer_size / domain_size, 4) if domain_size else 0.0,
                "component_id": comp_id,
            },
        }

        if answer_size == 0:
            rejections.append({**record, "reject_reason": "empty_answer"})
        elif answer_size == domain_size:
            rejections.append({**record, "reject_reason": "total_answer"})
        elif answer_size > OVERSIZED_THRESHOLD:
            rejections.append({**record, "reject_reason": "oversized"})
        elif domain_size > 0 and answer_size / domain_size > NEAR_TOTAL_FRACTION:
            rejections.append({**record, "reject_reason": "near_total"})
        else:
            candidates.append(record)

    assert_bound_names_resolved(candidates, ["cluster_anchor_name"], template_id)

    return candidates, rejections


RATIONALE_DEGENERATE_VALUES = {
    "tbd",
    "n/a",
    "na",
    "none",
    ".",
    "-",
    "tbd.",
    "n.a.",
}
RATIONALE_MIN_LENGTH = 5


def instantiate_rationale_abstention(backend, meta, config):
    """T6: rationale of a requirement; unanswerable if no rationale is recorded."""
    template_id = meta["template_id"]
    print(f"\n=== {template_id} ===")

    deprecated_stereotypes = config["deprecated_stereotypes"]
    type_name = resolve_type(meta, config)

    rows = backend.run_rows(
        f"""
        MATCH (n:Entity)
        WHERE '{type_name}' IN n.node_types AND {dep_filter_clause("n", deprecated_stereotypes)}
        RETURN n.id AS id, n.name AS name, n.rationale AS rationale
        """
    )
    print(f"  domain: {len(rows)} non-deprecated {type_name} elements")

    candidates, rejections = [], []
    n_answerable, n_unanswerable = 0, 0
    for row in rows:
        seed_id, seed_name = row["id"], row["name"]
        raw_rationale = row["rationale"]
        stripped = raw_rationale.strip() if raw_rationale else ""

        gold_query = f"MATCH (n:Entity {{id: '{seed_id}'}}) RETURN n.rationale AS rationale"
        record = {
            "template_id": template_id,
            "graph_version": meta["graph_version"],
            "sampling": "exhaustive",
            "rng_seed": None,
            "bindings": {"X": seed_id},
            "seed_name": seed_name,
            "gold_query": gold_query,
            "proof_subgraph": None,
        }

        if not stripped:
            record["answer"] = None
            record["answerable"] = False
            record["difficulty_coords"] = {"answerable": False, "rationale_length": 0}
            candidates.append(record)
            n_unanswerable += 1
        elif stripped.lower() in RATIONALE_DEGENERATE_VALUES or len(stripped) < RATIONALE_MIN_LENGTH:
            record["answer"] = stripped
            record["difficulty_coords"] = {"rationale_length": len(stripped)}
            rejections.append({**record, "reject_reason": "degenerate_rationale"})
        else:
            record["answer"] = stripped
            record["answerable"] = True
            record["difficulty_coords"] = {"answerable": True, "rationale_length": len(stripped)}
            candidates.append(record)
            n_answerable += 1

    print(
        f"  {n_answerable} answerable (rationale present), {n_unanswerable} unanswerable "
        f"(rationale absent), {len(rejections)} rejected (degenerate stub content)"
    )

    verify_rationale_gold_query_equivalence(backend, candidates)
    assert_bound_names_resolved(candidates, ["seed_name"], template_id)

    return candidates, rejections


def verify_rationale_gold_query_equivalence(backend, candidates, sample_size=30, seed=20260723, max_queries=200):
    rng = random.Random(seed)
    sample = rng.sample(candidates, min(sample_size, len(candidates), max_queries))
    n_checked, n_mismatch = 0, 0
    for record in sample:
        rows = backend.run_rows(record["gold_query"])
        live_rationale = rows[0]["rationale"] if rows else None
        live_stripped = live_rationale.strip() if live_rationale else None
        stored = record["answer"]
        n_checked += 1
        if live_stripped != stored:
            n_mismatch += 1
            print(f"  MISMATCH: {record['bindings']['X']} stored={stored!r} live={live_stripped!r}")
    if n_mismatch:
        raise RuntimeError(f"gold_query equivalence check failed: {n_mismatch}/{n_checked} mismatches")
    print(f"  gold_query equivalence check: {n_checked}/{n_checked} match (0 mismatches)")


def instantiate_lookup(backend, meta, config):
    """T7: default value of a named value property, qualified by its owning block."""
    template_id = meta["template_id"]
    print(f"\n=== {template_id} ===")

    deprecated_stereotypes = config["deprecated_stereotypes"]
    dep = dep_filter_clause("n", deprecated_stereotypes)

    edges, n_dropped = load_structural_edges(backend, deprecated_stereotypes)
    undirected_edges = [(a, rel, b) for a, rel, b, _direction in edges]
    uf, component_size, _ = build_components(undirected_edges)
    adj = build_structural_adjacency(edges)

    rows = backend.run_rows(
        f"""
        MATCH (n:Entity)
        WHERE 'ValueProperty' IN n.node_types AND {dep}
          AND n.name IS NOT NULL AND n.default_value IS NOT NULL
        RETURN n.id AS id, n.name AS name, n.default_value AS default_value
        """
    )
    print(f"  domain: {len(rows)} non-deprecated, named ValueProperty seeds with a populated default_value")

    resolved = []
    n_no_owner = 0
    for row in rows:
        _closure, _max_depth, _edge_types, proof_edges, _comp_id, _comp_size = bfs_structural_closure(
            adj, uf, component_size, row["id"]
        )
        owning_block = find_owning_block(backend, row["id"], "ValueProperty", proof_edges)
        if owning_block is None:
            n_no_owner += 1
            continue
        resolved.append({**row, "owning_block": owning_block})
    print(f"  {len(resolved)} have a resolvable owning_block ({n_no_owner} excluded, no owning_block)")

    owning_block_names = {}
    for row in resolved:
        ob = row["owning_block"]
        if ob not in owning_block_names:
            block_rows = backend.run_rows(f"MATCH (b:Entity {{id: '{ob}'}}) RETURN b.name AS name")
            owning_block_names[ob] = block_rows[0]["name"] if block_rows else ob

    by_name = defaultdict(list)
    for row in resolved:
        by_name[row["name"]].append(row)

    candidates, rejections = [], []
    for name, members in by_name.items():
        owners = [m["owning_block"] for m in members]
        residually_ambiguous = len(set(owners)) != len(owners)
        for m in members:
            record = {
                "template_id": template_id,
                "graph_version": meta["graph_version"],
                "sampling": "exhaustive",
                "rng_seed": None,
                "bindings": {"X": m["id"]},
                "seed_name": m["name"],
                "gold_query": f"MATCH (n:Entity {{id: '{m['id']}'}}) RETURN n.default_value AS default_value",
                "answer": m["default_value"],
                "proof_subgraph": None,
                "difficulty_coords": {
                    "ambiguous": len(members) > 1,
                    "name_collision_count": len(members),
                    "owning_block": m["owning_block"],
                    "owning_block_name": owning_block_names[m["owning_block"]],
                },
            }
            if residually_ambiguous:
                rejections.append({**record, "reject_reason": "residual_ambiguity"})
            else:
                candidates.append(record)

    n_ambiguous = sum(1 for r in candidates if r["difficulty_coords"]["ambiguous"])
    print(
        f"  {len(candidates)} candidates ({n_ambiguous} ambiguous-name / "
        f"{len(candidates) - n_ambiguous} unique-name), {len(rejections)} rejected (residual_ambiguity)"
    )

    verify_lookup_gold_query_equivalence(backend, candidates)
    assert_bound_names_resolved(candidates, ["seed_name"], template_id)

    return candidates, rejections


def verify_lookup_gold_query_equivalence(backend, candidates, sample_size=30, seed=20260723, max_queries=200):
    rng = random.Random(seed)
    sample = rng.sample(candidates, min(sample_size, len(candidates), max_queries))
    n_checked, n_mismatch = 0, 0
    for record in sample:
        rows = backend.run_rows(record["gold_query"])
        live_value = rows[0]["default_value"] if rows else None
        n_checked += 1
        if live_value != record["answer"]:
            n_mismatch += 1
            print(f"  MISMATCH: {record['bindings']['X']} stored={record['answer']!r} live={live_value!r}")
    if n_mismatch:
        raise RuntimeError(f"gold_query equivalence check failed: {n_mismatch}/{n_checked} mismatches")
    print(f"  gold_query equivalence check: {n_checked}/{n_checked} match (0 mismatches)")


def write_rejections_csv(path, rejections):
    fieldnames = [
        "template_id",
        "bindings",
        "reject_reason",
        "closure_size",
        "answer_size",
        "domain_size",
        "answer_fraction",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rejections:
            dc = r.get("difficulty_coords", {})
            writer.writerow(
                {
                    "template_id": r["template_id"],
                    "bindings": json.dumps(r["bindings"]),
                    "reject_reason": r["reject_reason"],
                    "closure_size": dc.get("closure_size", ""),
                    "answer_size": dc.get("answer_size", ""),
                    "domain_size": dc.get("domain_size", ""),
                    "answer_fraction": dc.get("answer_fraction", ""),
                }
            )


INSTANTIATORS = [
    ("impact_analysis.requirement_closure.yaml", instantiate_impact_analysis),
    ("negation.requirement_unrefined.yaml", instantiate_negation_scoped),
    ("impact_analysis.structural_closure.yaml", instantiate_structural_closure),
    ("aggregation.requirement_count.yaml", instantiate_aggregation),
    ("tracing.requirement_derivation.yaml", instantiate_tracing),
    ("rationale_abstention.yaml", instantiate_rationale_abstention),
    ("lookup.value_property_attribute.yaml", instantiate_lookup),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-path", default=str(paths.KUZU_DB))
    ap.add_argument("--templates-dir", default=str(paths.TEMPLATES))
    ap.add_argument("--outdir", default=str(paths.QA))
    ap.add_argument("--model-config", default=str(paths.MODEL_CONFIG))
    args = ap.parse_args()

    backend = KuzuBackend(args.db_path)
    config = load_yaml(args.model_config)

    all_candidates, all_rejections, summary = [], [], []
    for fname, instantiate in INSTANTIATORS:
        meta = load_yaml(f"{args.templates_dir}/{fname}")
        candidates, rejections = instantiate(backend, meta, config)
        all_candidates += candidates
        all_rejections += rejections
        summary.append((meta["template_id"], candidates, rejections))

    write_jsonl(f"{args.outdir}/candidates.jsonl", all_candidates)
    write_rejections_csv(f"{args.outdir}/rejections.csv", all_rejections)

    print("\n=== Summary ===")
    for template_id, candidates, rejections in summary:
        print(
            f"{template_id}: {len(candidates)} candidates, {len(rejections)} rejected "
            f"(of {len(candidates) + len(rejections)} enumerated)"
        )
        for reason, count in Counter(r["reject_reason"] for r in rejections).most_common():
            print(f"    {reason}: {count}")
    print(f"\nWrote {len(all_candidates)} candidates to {args.outdir}/candidates.jsonl")
    print(f"Wrote {len(all_rejections)} rejections to {args.outdir}/rejections.csv")


if __name__ == "__main__":
    main()
