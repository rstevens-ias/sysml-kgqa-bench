"""Identity baseline: feeding the gold answer through scoring must give 1.0 everywhere, and controlled
corruptions of it must give the analytically expected scores. Requires the Kuzu graph and the benchmark."""

from collections import defaultdict

import pytest

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.evaluation.metrics import TEMPLATE_GUARANTEE, build_edge_exists_fn, score_instance
from sysml_kgqa_bench.utils import load_jsonl, load_yaml

pytestmark = pytest.mark.skipif(
    not (paths.KUZU_DB.exists() and paths.VERBALIZED.exists()), reason="needs data/graph/tmt_kuzu and the benchmark"
)

ABSTENTION_GOLD_TEXT = "No rationale is recorded for this requirement."
SCORE_KEYS = {
    "completeness",
    "faithfulness",
    "exact_match",
    "edge_existence",
    "endpoint_connection",
    "correct",
    "similarity",
}


def _bow_embed(text):
    text = (text or "").lower()
    counts = defaultdict(int)
    for i in range(len(text) - 1):
        counts[text[i : i + 2]] += 1
    keys = sorted(counts)
    return [counts[k] for k in keys] or [0.0]


def identity_predicted(record, corruption=None):
    template_id = record["template_id"]
    guarantee = TEMPLATE_GUARANTEE[template_id]

    if guarantee in ("completeness", "exactness_set"):
        gold = list(record["answer"])
        if corruption == "minus_one":
            return gold[:-1] if gold else gold
        if corruption == "plus_one_fabricated":
            return gold + ["__FABRICATED_ELEMENT__"]
        if corruption == "empty":
            return []
        return gold

    if guarantee == "exactness_scalar":
        gold = record["answer"]
        if corruption == "empty":
            return None
        if isinstance(gold, int) or (isinstance(gold, str) and gold.strip().lstrip("-").isdigit()):
            gold_int = int(gold)
            if corruption == "minus_one":
                return gold_int - 1
            if corruption == "plus_one_fabricated":
                return gold_int + 1
            return gold
        if corruption in ("minus_one", "plus_one_fabricated"):
            return f"{gold}__CORRUPTED__"
        return gold

    if guarantee == "faithfulness_path":
        witness_path = record["answer"]["witness_path"]
        witness_relations = record["answer"]["witness_relations"]
        edges = [
            {"from": witness_path[i], "to": witness_path[i + 1], "relation": witness_relations[i]}
            for i in range(len(witness_relations))
        ]
        if corruption == "minus_one":
            return edges[:-1]
        if corruption == "plus_one_fabricated":
            return edges + [{"from": witness_path[-1], "to": "__FAKE_NODE__", "relation": "__FAKE_RELATION__"}]
        if corruption == "empty":
            return None
        return edges

    if guarantee == "abstention":
        if corruption is not None:
            return None
        if not record["answerable"]:
            return ABSTENTION_GOLD_TEXT
        return record["answer"]

    raise ValueError(f"unhandled guarantee {guarantee!r}")


def expected_metric_for_corruption(record, corruption):
    guarantee = TEMPLATE_GUARANTEE[record["template_id"]]
    if guarantee == "completeness":
        n = len(record["answer"])
        if corruption is None:
            return {"completeness": 1.0, "faithfulness": 1.0}
        if corruption == "minus_one":
            return {"completeness": (n - 1) / n, "faithfulness": 1.0}
        if corruption == "plus_one_fabricated":
            return {"completeness": 1.0, "faithfulness": n / (n + 1)}
        if corruption == "empty":
            return {"completeness": 0.0, "faithfulness": 1.0}
    if guarantee in ("exactness_set", "exactness_scalar"):
        if corruption is None:
            return {"exact_match": 1.0}
        return {"exact_match": 0.0}
    if guarantee == "faithfulness_path":
        n = len(record["answer"]["witness_relations"])
        if corruption is None:
            return {"edge_existence": 1.0, "endpoint_connection": 1.0}
        if corruption == "minus_one":
            if n <= 1:
                return {"edge_existence": 0.0, "endpoint_connection": 0.0}
            return {"edge_existence": 1.0, "endpoint_connection": 0.0}
        if corruption == "plus_one_fabricated":
            # The real edges still connect source and target; the fabricated edge only lowers edge_existence.
            return {"edge_existence": n / (n + 1), "endpoint_connection": 1.0}
        if corruption == "empty":
            return {"edge_existence": 0.0, "endpoint_connection": 0.0}
    return None


@pytest.fixture(scope="module")
def records():
    return load_jsonl(paths.VERBALIZED)


@pytest.fixture(scope="module")
def edge_exists_fn():
    from sysml_kgqa_bench.graph.backend import KuzuBackend

    config = load_yaml(paths.MODEL_CONFIG)
    backend = KuzuBackend(paths.KUZU_DB)
    return build_edge_exists_fn(
        backend, list(config["trace_relations"].keys()), config["deprecated_stereotypes"], config["trace_relations"]
    )


def test_identity_scores_one(records, edge_exists_fn):
    failures = defaultdict(list)
    for r in records:
        is_abstention = TEMPLATE_GUARANTEE[r["template_id"]] == "abstention"
        result = score_instance(
            r,
            identity_predicted(r),
            edge_exists_fn=edge_exists_fn,
            embed_fn=_bow_embed if is_abstention else None,
        )
        for k, v in result.items():
            if k in SCORE_KEYS and v is not None and abs(v - 1.0) > 1e-9:
                failures[r["template_id"]].append((k, v, r.get("bindings")))
    assert not failures


@pytest.mark.parametrize("corruption", ["minus_one", "plus_one_fabricated", "empty"])
def test_corruption_ladder(records, edge_exists_fn, corruption):
    mismatches = []
    for r in records:
        if TEMPLATE_GUARANTEE[r["template_id"]] == "abstention":
            continue
        result = score_instance(r, identity_predicted(r, corruption=corruption), edge_exists_fn=edge_exists_fn)
        for k, expected in expected_metric_for_corruption(r, corruption).items():
            got = result.get(k)
            if got is None or abs(got - expected) > 1e-9:
                mismatches.append((r["template_id"], k, expected, got, r.get("bindings")))
    assert not mismatches, mismatches[:10]
