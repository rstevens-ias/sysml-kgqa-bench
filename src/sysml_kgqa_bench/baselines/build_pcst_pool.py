"""Build the G-Retriever candidate pool: the k-hop neighborhood of each instance's seed entities.

Writes the union of all nodes (with name, types, default value and rationale), the edges induced on it, and
the node ids per instance.
"""

import argparse
import json
import os
import time

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.graph.backend import KuzuBackend
from sysml_kgqa_bench.utils import load_jsonl, record_key


def seeds_for_record(record):
    b = record["bindings"]
    t = record["template_id"]
    if t == "tracing.requirement_derivation":
        return [b["source"], b["target"]]
    if t == "aggregation.requirement_count":
        return [b["scope"]]
    if t == "negation.requirement_unrefined":
        return [b["component_id"]]
    return [b["X"]]


def khop_neighbor_ids(backend, seed_id, k, cache):
    if seed_id in cache:
        return cache[seed_id]
    rows = backend.run_rows(
        f"MATCH (n:Entity {{id: $sid}})-[:Relation*1..{k}]-(m:Entity) RETURN DISTINCT m.id AS id", {"sid": seed_id}
    )
    ids = {seed_id} | {r["id"] for r in rows}
    cache[seed_id] = ids
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(paths.VERBALIZED))
    ap.add_argument("--db-path", default=str(paths.KUZU_DB))
    ap.add_argument("--k", type=int, default=2, help="neighborhood radius")
    ap.add_argument("--outpath", default=str(paths.PCST_POOL))
    args = ap.parse_args()

    records = load_jsonl(args.input)
    backend = KuzuBackend(args.db_path)

    cache = {}
    per_instance = {}
    global_node_ids = set()

    t0 = time.monotonic()
    n_seed_queries = 0
    for i, r in enumerate(records, start=1):
        seeds = seeds_for_record(r)
        instance_ids = set()
        for s in seeds:
            if s not in cache:
                n_seed_queries += 1
            instance_ids |= khop_neighbor_ids(backend, s, args.k, cache)
        per_instance[record_key(r)] = sorted(instance_ids)
        global_node_ids |= instance_ids
        if i % 100 == 0:
            print(
                f"  {i}/{len(records)} records processed, "
                f"{len(cache)} unique seeds queried so far, "
                f"{len(global_node_ids)} nodes in union so far, "
                f"{time.monotonic() - t0:.0f}s elapsed",
                flush=True,
            )

    print(
        f"Done: {len(records)} records, {n_seed_queries} unique seed k-hop queries, "
        f"{len(global_node_ids)} nodes in the global union, {time.monotonic() - t0:.0f}s total."
    )

    global_node_ids = sorted(global_node_ids)
    print("Fetching name/types/default_value/rationale for the global node union...")
    node_meta = {}
    CHUNK = 2000
    for i in range(0, len(global_node_ids), CHUNK):
        chunk = global_node_ids[i : i + CHUNK]
        rows = backend.run_rows(
            "MATCH (n:Entity) WHERE n.id IN $ids RETURN n.id AS id, n.name AS name, n.node_types AS node_types, "
            "n.default_value AS default_value, n.rationale AS rationale",
            {"ids": chunk},
        )
        for row in rows:
            node_meta[row["id"]] = {
                "id": row["id"],
                "name": row["name"],
                "node_types": row["node_types"] or [],
                "default_value": row["default_value"],
                "rationale": row["rationale"],
            }
    missing = [nid for nid in global_node_ids if nid not in node_meta]
    if missing:
        raise RuntimeError(f"{len(missing)} node ids from the k-hop union were not found, e.g. {missing[:5]!r}.")

    print("Fetching induced edges among the global node union...")
    global_node_id_set = set(global_node_ids)
    global_edges = []
    seen_edges = set()
    for i in range(0, len(global_node_ids), CHUNK):
        chunk = global_node_ids[i : i + CHUNK]
        rows = backend.run_rows(
            "MATCH (a:Entity)-[r:Relation]->(b:Entity) WHERE a.id IN $ids "
            "RETURN a.id AS from_id, b.id AS to_id, r.relation AS relation",
            {"ids": chunk},
        )
        for row in rows:
            if row["to_id"] not in global_node_id_set:
                continue
            key = (row["from_id"], row["to_id"], row["relation"])
            if key not in seen_edges:
                seen_edges.add(key)
                global_edges.append({"from": row["from_id"], "to": row["to_id"], "relation": row["relation"]})

    print(f"Global pool: {len(global_node_ids)} nodes, {len(global_edges)} induced edges.")

    out = {
        "k": args.k,
        "global_nodes": [node_meta[nid] for nid in global_node_ids],
        "global_edges": global_edges,
        "per_instance": per_instance,
    }
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
    with open(args.outpath, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"Wrote {args.outpath}")


if __name__ == "__main__":
    main()
