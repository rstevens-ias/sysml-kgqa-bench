import pytest

from sysml_kgqa_bench.baselines.common import complete_and_extract
from sysml_kgqa_bench.evaluation.extraction import _extract_scalar_value, extract_prediction
from sysml_kgqa_bench.evaluation.metrics import (
    EXTRACTION_FAILED,
    abstention_metrics,
    looks_like_abstention,
    score_instance,
)
from sysml_kgqa_bench.llm import extract_json

GOLD_IDS = ["_abc123", "_def456"]
NAME_TO_IDS = {"acquisition camera sensitivity": ["_abc123"], "chopping mirror assembly": ["_def456"]}


def extract_from_raw_text(template_id, raw_text, **kwargs):
    return extract_prediction(template_id, extract_json(raw_text), **kwargs)


@pytest.mark.parametrize(
    "raw",
    [
        'Based on my analysis of the model, here is the answer:\n\n{"answer_elements": ["_abc123", "_def456"]}\n\n'
        "Let me know if you need anything else.",
        '```json\n{"answer_elements": ["_abc123", "_def456"]}\n```',
    ],
)
def test_set_answer_wrapped_in_text(raw):
    predicted, _ = extract_from_raw_text("impact_analysis.requirement_closure", raw)
    assert sorted(predicted) == GOLD_IDS


@pytest.mark.parametrize(
    "names",
    [
        '["Acquisition Camera Sensitivity", "Chopping Mirror Assembly"]',
        '["ACQUISITION CAMERA SENSITIVITY", "chopping mirror assembly"]',
    ],
)
def test_set_answer_with_names(names):
    raw = f'{{"answer_elements": {names}}}'
    predicted, meta = extract_from_raw_text("impact_analysis.requirement_closure", raw, name_to_ids=NAME_TO_IDS)
    assert sorted(predicted) == GOLD_IDS
    assert meta["n_unresolved"] == 0


def test_empty_set_answer():
    raw = 'Sure, I can help with that.\n\n{"answer_elements": []}\n\nThe closure appears to be empty.'
    predicted, _ = extract_from_raw_text("negation.requirement_unrefined", raw)
    assert predicted == []


@pytest.mark.parametrize(
    "word, value",
    [
        ("seventeen", 17),
        ("two", 2),
        ("one hundred and twenty-three", 123),
        ("three hundred eighty-six", 386),
        ("Seventeen", 17),
    ],
)
def test_count_as_word(word, value):
    raw = f'The count is {word}.\n\n{{"count": "{word}"}}'
    predicted, _ = extract_from_raw_text("aggregation.requirement_count", raw)
    assert predicted == value


def test_count_as_numeral_string():
    predicted, _ = extract_from_raw_text("aggregation.requirement_count", '{"count": "17"}')
    assert predicted == 17


def test_path_with_names():
    node_name_to_id = {"acquisition camera sensitivity": "_abc123", "chopping mirror assembly": "_def456"}
    raw = (
        "Here is the derivation chain I found:\n\n"
        '{"path": [{"from": "Acquisition Camera Sensitivity", "to": "Chopping Mirror Assembly", '
        '"relation": "Refines"}]}'
    )
    predicted, _ = extract_from_raw_text(
        "tracing.requirement_derivation", raw, node_name_to_id_for_path=node_name_to_id
    )
    assert predicted == [{"from": "_abc123", "to": "_def456", "relation": "Refines"}]


def test_path_null():
    predicted, _ = extract_from_raw_text("tracing.requirement_derivation", '{"path": null}')
    assert predicted is None


@pytest.mark.parametrize(
    "text",
    [
        "Is there a rationale recorded for this requirement? It doesn't appear so.",
        "Does the model record a rationale here? Not that I can find.",
        "Was a rationale given? I don't see one in the available information.",
    ],
)
def test_question_phrased_abstention(text):
    assert looks_like_abstention(text)


def test_question_phrased_abstention_in_json():
    raw = '{"rationale": "Is there a rationale recorded for this requirement? It doesn\'t appear so."}'
    predicted, _ = extract_from_raw_text("rationale_abstention", raw)
    assert looks_like_abstention(predicted)


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("The default value is 4.", "4"),
        ("Its default value is 0.", "0"),
        ('The value of this property is "Block".', "Block"),
        ("4", "4"),
    ],
)
def test_scalar_value_in_sentence(raw_value, expected):
    assert _extract_scalar_value(raw_value) == expected


class RefusingClient:
    def __init__(self, refusal_text):
        self.refusal_text = refusal_text

    def complete_json(self, prompt, temperature):
        return extract_json(self.refusal_text)


REFUSALS = [
    "I cannot determine this from the information available in the graph.",
    "Based on the provided context, there is no way to answer this question with confidence.",
]


@pytest.mark.parametrize("refusal", REFUSALS)
def test_refusal_on_rationale_is_scored_as_abstention(refusal):
    predicted, meta, _ = complete_and_extract(RefusingClient(refusal), "prompt", 0.0, "rationale_abstention")
    assert meta["extraction_failed"] is True
    assert looks_like_abstention(predicted)


@pytest.mark.parametrize("refusal", REFUSALS)
@pytest.mark.parametrize(
    "template_id",
    [
        "impact_analysis.requirement_closure",
        "negation.requirement_unrefined",
        "aggregation.requirement_count",
        "tracing.requirement_derivation",
    ],
)
def test_refusal_is_extraction_failure(refusal, template_id):
    predicted, _, _ = complete_and_extract(RefusingClient(refusal), "prompt", 0.0, template_id)
    assert predicted is EXTRACTION_FAILED


def test_extraction_failure_is_not_scored():
    gold_set = {"template_id": "impact_analysis.requirement_closure", "answer": ["_abc123"]}
    gold_count = {"template_id": "aggregation.requirement_count", "answer": 17}
    assert score_instance(gold_set, EXTRACTION_FAILED) == {"extraction_failed": True}
    assert score_instance(gold_count, EXTRACTION_FAILED) == {"extraction_failed": True}


def test_explicit_null_differs_from_missing_key():
    predicted, _ = extract_prediction("aggregation.requirement_count", {"count": None})
    assert predicted is None
    predicted, _ = extract_prediction("aggregation.requirement_count", {})
    assert predicted is EXTRACTION_FAILED


def test_refusal_on_answerable_rationale_is_wrong():
    client = RefusingClient("I cannot determine this from the information available.")
    predicted, _, _ = complete_and_extract(client, "prompt", 0.0, "rationale_abstention")
    result = abstention_metrics(
        predicted, answerable=True, gold_text="some gold rationale", embed_fn=lambda t: [1.0, 0.0]
    )
    assert result["correct"] == 0.0
