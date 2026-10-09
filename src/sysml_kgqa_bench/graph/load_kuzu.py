"""Load the node-link graph JSON into a Kuzu database (one Entity node table, one Relation edge table)."""

import argparse
import csv
import json
import os
import shutil
import sys

import kuzu
import networkx as nx

from sysml_kgqa_bench import paths

NODE_COLUMNS = [
    "name",
    "hierarchy_path",
    "text",
    "rationale",
    "tmt_id",
    "legacy_id",
    "acceptance_state",
    "applicable_subsystems",
    "se_notes",
    "layer",
    "domain",
    "owner_block_id",
    "represents_id",
    "represents_name",
    "typed_by_id",
    "typed_by_name",
    "default_value",
    "min_value",
    "max_value",
]
EDGE_COLUMNS = [
    "edge_id",
    "relation",
    "category",
    "interaction_id",
    "message_id",
    "message_name",
    "signal_id",
    "signal_name",
]


def serialize_node_types(attrs):
    types = attrs.get("node_types") or []
    if not types:
        return ""
    return "[" + ",".join(types) + "]"


def write_node_csv(G, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "node_types"] + NODE_COLUMNS)
        for node_id, attrs in G.nodes(data=True):
            w.writerow([node_id, serialize_node_types(attrs)] + [attrs.get(c, "") for c in NODE_COLUMNS])


def write_edge_csv(G, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_id", "to_id"] + EDGE_COLUMNS)
        for u, v, attrs in G.edges(data=True):
            w.writerow([u, v] + [attrs.get(c, "") for c in EDGE_COLUMNS])


def build_schema(conn):
    node_cols = ", ".join(f"{c} STRING" for c in NODE_COLUMNS)
    conn.execute(f"CREATE NODE TABLE Entity(id STRING, node_types STRING[], {node_cols}, PRIMARY KEY(id))")
    edge_cols = ", ".join(f"{c} STRING" for c in EDGE_COLUMNS)
    conn.execute(f"CREATE REL TABLE Relation(FROM Entity TO Entity, {edge_cols})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-json", default=str(paths.GRAPH_JSON))
    ap.add_argument("--db-path", default=str(paths.KUZU_DB))
    ap.add_argument("--csv-dir", default=str(paths.KUZU_CSV), help="directory for the intermediate CSV files")
    args = ap.parse_args()

    csv_dir = args.csv_dir
    os.makedirs(csv_dir, exist_ok=True)

    print(f"Loading {args.graph_json} ...", file=sys.stderr)
    with open(args.graph_json, encoding="utf-8") as f:
        data = json.load(f)
    G = nx.node_link_graph(data, directed=True, multigraph=True)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)

    nodes_csv = f"{csv_dir}/nodes.csv"
    edges_csv = f"{csv_dir}/edges.csv"
    print(f"Writing {nodes_csv} and {edges_csv} ...", file=sys.stderr)
    write_node_csv(G, nodes_csv)
    write_edge_csv(G, edges_csv)

    if os.path.exists(args.db_path):
        print(f"Removing existing database at {args.db_path} for a clean rebuild ...", file=sys.stderr)
        if os.path.isdir(args.db_path):
            shutil.rmtree(args.db_path)
        else:
            os.remove(args.db_path)
            wal_path = f"{args.db_path}.wal"
            if os.path.exists(wal_path):
                os.remove(wal_path)

    print(f"Creating Kuzu database at {args.db_path} ...", file=sys.stderr)
    db = kuzu.Database(args.db_path)
    conn = kuzu.Connection(db)
    build_schema(conn)

    def to_kuzu_path(p):
        return os.path.abspath(p).replace("\\", "/")

    print("Copying nodes ...", file=sys.stderr)
    conn.execute(f"COPY Entity FROM '{to_kuzu_path(nodes_csv)}' (HEADER=true, PARALLEL=FALSE)")
    print("Copying edges ...", file=sys.stderr)
    conn.execute(f"COPY Relation FROM '{to_kuzu_path(edges_csv)}' (HEADER=true, PARALLEL=FALSE)")

    n_nodes = conn.execute("MATCH (n:Entity) RETURN COUNT(*)").get_next()[0]
    n_edges = conn.execute("MATCH ()-[r:Relation]->() RETURN COUNT(*)").get_next()[0]
    print(f"Loaded: {n_nodes} nodes, {n_edges} edges", file=sys.stderr)

    print("Sample query -- a Requirement --Refines--> another Requirement:", file=sys.stderr)
    res = conn.execute(
        "MATCH (a:Entity)-[r:Relation {relation: 'Refines'}]->(b:Entity) "
        "WHERE 'Requirement' IN a.node_types "
        "RETURN a.tmt_id, b.name LIMIT 3"
    )
    while res.has_next():
        print("  ", res.get_next(), file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
