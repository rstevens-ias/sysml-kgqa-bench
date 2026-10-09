import pytest

from sysml_kgqa_bench.evaluation.metrics import (
    LASMNotWired,
    abstention_metrics,
    cosine_similarity,
    exact_scalar_match,
    exact_set_match,
    looks_like_abstention,
    path_predicate_metrics,
    set_metrics,
)

GOLD = ["a", "b", "c", "d"]


def test_set_metrics_exact():
    assert set_metrics(GOLD, GOLD) == {"completeness": 1.0, "faithfulness": 1.0}


def test_set_metrics_missing_element():
    r = set_metrics(["b", "c", "d"], GOLD)
    assert r["completeness"] == 3 / 4
    assert r["faithfulness"] == 1.0


def test_set_metrics_fabricated_element():
    r = set_metrics(GOLD + ["fabricated"], GOLD)
    assert r["completeness"] == 1.0
    assert r["faithfulness"] == 4 / 5


def test_set_metrics_empty_prediction():
    assert set_metrics([], GOLD)["completeness"] == 0.0


def test_set_metrics_order_independent():
    assert set_metrics(list(reversed(GOLD)), GOLD) == {"completeness": 1.0, "faithfulness": 1.0}


def test_set_metrics_empty_gold_raises():
    with pytest.raises(ValueError):
        set_metrics(["x"], [])


def test_exact_set_match():
    assert exact_set_match(["a", "b"], ["b", "a"]) == 1.0
    assert exact_set_match(["a"], ["a", "b"]) == 0.0
    assert exact_set_match(["a", "b", "c"], ["a", "b"]) == 0.0


@pytest.mark.parametrize(
    "predicted, gold, expected",
    [
        (39, 39, 1.0),
        (38, 39, 0.0),
        (40, 39, 0.0),
        ("39", 39, 1.0),
        ("39.0", 39, 1.0),
        ("Block", "block", 1.0),
        ("  Block  ", "Block", 1.0),
        ("Block", "Enumeration", 0.0),
    ],
)
def test_exact_scalar_match(predicted, gold, expected):
    assert exact_scalar_match(predicted, gold) == expected


GRAPH = {("S", "M", "Refines"), ("M", "T", "Refines"), ("S", "T", "Refines")}


def edge_exists(a, b, rel):
    return (a, b, rel) in GRAPH


def test_path_alternative_valid_path():
    r = path_predicate_metrics([{"from": "S", "to": "T", "relation": "Refines"}], "S", "T", edge_exists)
    assert r == {"edge_existence": 1.0, "endpoint_connection": 1.0}


def test_path_with_fabricated_edge():
    path = [{"from": "S", "to": "M", "relation": "Refines"}, {"from": "M", "to": "T", "relation": "FAKE"}]
    r = path_predicate_metrics(path, "S", "T", edge_exists)
    assert r == {"edge_existence": 0.5, "endpoint_connection": 0.0}


def test_path_none():
    assert path_predicate_metrics(None, "S", "T", edge_exists) == {"edge_existence": 0.0, "endpoint_connection": 0.0}


def test_path_not_reaching_target():
    r = path_predicate_metrics([{"from": "S", "to": "M", "relation": "Refines"}], "S", "T", edge_exists)
    assert r == {"edge_existence": 1.0, "endpoint_connection": 0.0}


def test_path_hops_out_of_order():
    path = [{"from": "M", "to": "T", "relation": "Refines"}, {"from": "S", "to": "M", "relation": "Refines"}]
    r = path_predicate_metrics(path, "S", "T", edge_exists)
    assert r == {"edge_existence": 1.0, "endpoint_connection": 1.0}


def test_abstention_on_unanswerable():
    assert abstention_metrics("no rationale is recorded for this requirement", answerable=False)["correct"] == 1.0
    plausible = (
        "This requirement was relaxed because the original tolerance was found to be "
        "unachievable within the mechanical budget, per a change-request review."
    )
    assert abstention_metrics(plausible, answerable=False)["correct"] == 0.0


def test_abstention_on_answerable():
    assert abstention_metrics("no rationale", answerable=True)["correct"] == 0.0
    r = abstention_metrics(
        "some rationale text", answerable=True, gold_text="some rationale text", embed_fn=lambda t: [1.0, 0.0]
    )
    assert r["correct"] == 1.0
    r = abstention_metrics(
        "some rationale text",
        answerable=True,
        gold_text="unrelated text",
        embed_fn=lambda t: [1.0, 0.0] if t == "some rationale text" else [0.0, 1.0],
    )
    assert r["correct"] == 0.0


def test_abstention_without_embed_fn_raises():
    with pytest.raises(LASMNotWired):
        abstention_metrics("some rationale text", answerable=True, gold_text="gold text", embed_fn=None)


@pytest.mark.parametrize(
    "text",
    [
        "No rationale is recorded for this requirement.",
        "Not available.",
        "Unknown.",
        "The model does not document a justification for this requirement.",
        "",
        None,
        "After checking the specific data provided, there is no stated rationale recorded; the field is empty.",
        "After checking the specific data provided for requirement REQ-1-ORD-5235, there is no documented "
        "rationale recorded; the field is explicitly noted as empty or absent.",
        "After checking the specific data provided for requirement REQ-1-OAD-1755, there is no stated reasoning "
        "recorded; the rationale field is explicitly empty.",
    ],
)
def test_detects_abstention(text):
    assert looks_like_abstention(text)


@pytest.mark.parametrize(
    "text",
    [
        "This tolerance was tightened following the CDR review of thermal drift margins.",
        "Per CCR-142, the sensor was relocated to reduce vibration coupling.",
        "The original derivation assumed a 10-year mission life; this was revised to 15.",
    ],
)
def test_detects_substantive_answer(text):
    assert not looks_like_abstention(text)


def test_cosine_similarity():
    assert abs(cosine_similarity([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
