"""Embed all nodes and relation types of the G-Retriever candidate pool and cache them as .npz."""

import argparse
import json
import os
import time

import numpy as np

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.embeddings import get_client


def node_text(node):
    types = "/".join(node.get("node_types") or [])
    return f"{node.get('name')} ({types})" if types else str(node.get("name"))


def embed_batched(embed_fn_batch, texts, batch_size=256, label=""):
    out = []
    t0 = time.monotonic()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        out.extend(embed_fn_batch(chunk))
        elapsed = time.monotonic() - t0
        n_done = min(i + batch_size, len(texts))
        rate = n_done / elapsed if elapsed > 0 else float("inf")
        eta = (len(texts) - n_done) / rate if rate > 0 else float("inf")
        print(f"  [{label}] {n_done}/{len(texts)} embedded, {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=str(paths.PCST_POOL))
    ap.add_argument("--outpath", default=str(paths.PCST_EMBEDDINGS))
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N nodes (timing test, no output)")
    args = ap.parse_args()

    with open(args.pool, encoding="utf-8") as f:
        pool = json.load(f)
    nodes = pool["global_nodes"]
    edges = pool["global_edges"]
    print(f"Pool: {len(nodes)} nodes, {len(edges)} edges.")

    relation_strings = sorted({e["relation"] for e in edges})
    print(f"Distinct relation strings to embed: {len(relation_strings)} (from {len(edges)} edges).")

    if args.limit:
        nodes = nodes[: args.limit]
        print(f"--limit set: embedding only the first {len(nodes)} nodes, no output is written.")

    client, provenance = get_client()
    print(f"Embedding provenance: {provenance}")

    node_texts = [node_text(n) for n in nodes]
    node_embeddings = embed_batched(client.embed, node_texts, label="nodes")
    relation_embeddings = embed_batched(client.embed, relation_strings, label="relations")

    if args.limit:
        return

    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
    np.savez(
        args.outpath,
        node_ids=np.array([n["id"] for n in nodes], dtype=object),
        node_embeddings=np.array(node_embeddings, dtype=np.float32),
        relation_strings=np.array(relation_strings, dtype=object),
        relation_embeddings=np.array(relation_embeddings, dtype=np.float32),
        embedding_provenance=json.dumps(provenance),
    )
    print(f"Wrote {args.outpath}")


if __name__ == "__main__":
    main()
