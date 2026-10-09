"""Name-similarity floor: predict every element whose name shares a token with the seed's name.

Scored on T1/T3 closures, T5 target retrieval, and exact name lookup for T7. No LLM calls.
"""

import argparse
from collections import defaultdict

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.evaluation.metrics import exact_scalar_match, score_instance
from sysml_kgqa_bench.generation.instantiate import dep_filter_clause
from sysml_kgqa_bench.generation.leakage_audit import tokenize
from sysml_kgqa_bench.graph.backend import KuzuBackend
from sysml_kgqa_bench.utils import load_jsonl, load_yaml


def load_all_domain_names(backend, deprecated_stereotypes):
    rows = backend.run_rows(
        f"MATCH (n:Entity) WHERE n.name IS NOT NULL AND {dep_filter_clause('n', deprecated_stereotypes)} "
        f"RETURN n.id AS id, n.name AS name"
    )
    return {r["id"]: r["name"] for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(paths.VERBALIZED))
    ap.add_argument("--db-path", default=str(paths.KUZU_DB))
    ap.add_argument("--model-config", default=str(paths.MODEL_CONFIG))
    args = ap.parse_args()

    records = load_jsonl(args.input)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["template_id"]].append(r)

    config = load_yaml(args.model_config)
    backend = KuzuBackend(args.db_path)
    dep = config["deprecated_stereotypes"]

    print("Loading domain name pool (all non-deprecated named entities)...")
    id_to_name = load_all_domain_names(backend, dep)
    print(f"  {len(id_to_name)} candidate elements.")

    print("Tokenizing names...")
    id_to_tokens = {nid: tokenize(name) for nid, name in id_to_name.items()}
    token_to_ids = defaultdict(set)
    for nid, toks in id_to_tokens.items():
        for t in toks:
            token_to_ids[t].add(nid)

    def name_similarity_retrieve(seed_name, exclude_id=None):
        seed_tokens = tokenize(seed_name)
        retrieved = set()
        for t in seed_tokens:
            retrieved |= token_to_ids.get(t, set())
        retrieved.discard(exclude_id)
        return list(retrieved)

    for template_id in ("impact_analysis.requirement_closure", "impact_analysis.structural_closure"):
        recs = by_class[template_id]
        agg = defaultdict(list)
        pool_sizes = []
        for r in recs:
            seed = r["bindings"]["X"]
            predicted = name_similarity_retrieve(r["seed_name"], exclude_id=seed)
            pool_sizes.append(len(predicted))
            result = score_instance(r, predicted)
            for k, v in result.items():
                agg[k].append(v)
        print(f"\n--- {template_id}: name-similarity floor ({len(recs)} instances) ---")
        for k, vals in agg.items():
            print(f"  {k}: mean={sum(vals) / len(vals):.3f}")
        print(
            f"  mean retrieved-set size: {sum(pool_sizes) / len(pool_sizes):.1f} "
            f"(min {min(pool_sizes)}, max {max(pool_sizes)})"
        )

    trace_recs = by_class["tracing.requirement_derivation"]
    n_hit = 0
    n_empty_retrieval = 0
    n_nonempty_wrong = 0
    retrieved_sizes = []
    for r in trace_recs:
        source, target = r["bindings"]["source"], r["bindings"]["target"]
        retrieved = name_similarity_retrieve(r["source_name"], exclude_id=source)
        retrieved_sizes.append(len(retrieved))
        if target in retrieved:
            n_hit += 1
        elif not retrieved:
            n_empty_retrieval += 1
        else:
            n_nonempty_wrong += 1
    print(f"\n--- tracing.requirement_derivation: name-similarity retrieval floor ({len(trace_recs)} instances) ---")
    print(
        f"  target found via token-overlap retrieval from source: {n_hit}/{len(trace_recs)} = "
        f"{n_hit / len(trace_recs):.1%}"
    )
    print(f"  mean retrieved-set size: {sum(retrieved_sizes) / len(retrieved_sizes):.1f}")
    print(f"  empty retrieval: {n_empty_retrieval}/{len(trace_recs)} = {n_empty_retrieval / len(trace_recs):.1%}")
    print(
        f"  non-empty retrieval without target: {n_nonempty_wrong}/{len(trace_recs)} = "
        f"{n_nonempty_wrong / len(trace_recs):.1%}"
    )

    print("\n--- lookup.value_property_attribute: exact name+owning_block lookup floor ---")
    hp_rows = backend.run_rows(
        "MATCH (block:Entity)-[r:Relation {relation: 'has_property'}]->(vp:Entity) "
        "RETURN block.name AS block_name, vp.id AS id, vp.name AS vp_name, vp.default_value AS default_value"
    )
    name_block_index = defaultdict(list)
    for row in hp_rows:
        if row["vp_name"] and row["block_name"]:
            name_block_index[(row["vp_name"].strip().lower(), row["block_name"].strip().lower())].append(
                (row["id"], row["default_value"])
            )

    look_recs = by_class["lookup.value_property_attribute"]

    n_found = n_correct = n_ambiguous = 0
    for r in look_recs:
        key = (r["seed_name"].strip().lower(), r["difficulty_coords"]["owning_block_name"].strip().lower())
        candidates = name_block_index.get(key, [])
        if not candidates:
            continue
        n_found += 1
        if len(candidates) > 1:
            n_ambiguous += 1
        predicted_value = candidates[0][1]
        if exact_scalar_match(predicted_value, r["answer"]) == 1.0:
            n_correct += 1
    print(f"  found by exact name+owning_block match: {n_found}/{len(look_recs)} = {n_found / len(look_recs):.1%}")
    print(
        f"  of those, exact_match correct: {n_correct}/{n_found} = {n_correct / n_found:.1%}"
        if n_found
        else "  none found"
    )
    print(f"  ambiguous (>1 candidate under the same name+owning_block key): {n_ambiguous}")


if __name__ == "__main__":
    main()
