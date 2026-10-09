"""Answer extraction: turn a parsed model response into the typed prediction expected by the metrics.

PROMPT_SHAPE holds the response format requested for each template.
"""

from sysml_kgqa_bench.evaluation.metrics import EXTRACTION_FAILED

# Answer format requested by the closed-book, oracle and G-Retriever prompts.
PROMPT_SHAPE = {
    "impact_analysis.requirement_closure": (
        'Respond with a JSON object: {"answer_elements": [<list of element '
        "IDS if you were given them in context, otherwise element NAMES>]}. "
        "List every element you believe is downstream-affected -- omissions "
        "and fabrications are both penalized independently, so include "
        "everything you are confident about and nothing you are guessing at. "
        "If there are many such elements, list every single one of them "
        "individually in the array -- do not summarize, truncate, or name "
        "only a representative subset; a long correct list scores no worse "
        "than a short one, and a long list that stops early is scored as "
        "missing every element it left out."
    ),
    "impact_analysis.structural_closure": (
        'Respond with a JSON object: {"answer_elements": [<list of element '
        "IDS if given, otherwise NAMES>]}, every element you believe is "
        "structurally/parametrically affected. If there are many such "
        "elements, list every single one of them individually in the array "
        "-- do not summarize, truncate, or name only a representative "
        "subset; a long correct list scores no worse than a short one, and "
        "a long list that stops early is scored as missing every element "
        "it left out."
    ),
    "negation.requirement_unrefined": (
        'Respond with a JSON object: {"answer_elements": [<list of element '
        "IDS if given, otherwise NAMES>]}, every requirement you believe has "
        "no refining requirement. If there are many such requirements, list "
        "every single one of them individually in the array -- do not "
        "summarize, truncate, or name only a representative subset; a long "
        "correct list scores no worse than a short one, and a long list "
        "that stops early is scored as missing every element it left out."
    ),
    "aggregation.requirement_count": (
        'Respond with a JSON object: {"count": <integer>}. Actually work out '
        "this specific count from what you were given (enumerate or reason "
        "through the elements that satisfy the question's conditions) "
        "rather than defaulting to zero or an unconsidered guess -- there is "
        "no partial credit for a close answer, so a considered wrong count "
        "and an unconsidered one score identically, but only a considered "
        "one has a chance of being right."
    ),
    "tracing.requirement_derivation": (
        'Respond with a JSON object: {"path": [{"from": <id or name>, "to": '
        '<id or name>, "relation": <relation name>}, ...]}, describing one '
        "valid chain of edges connecting the two elements named in the "
        "question, in order. Check what you were actually given before "
        "answering: if and only if, after checking, you find no such chain, "
        'set "path" to the JSON value null (not a description of a chain, '
        "not any other placeholder -- the literal value null) -- but only "
        "after actually checking the specific elements in THIS question, "
        "never as a default answer. Only assert edges you are confident are "
        "real -- a fabricated edge anywhere in the chain invalidates the "
        "whole answer."
    ),
    "rationale_abstention": (
        'Respond with a JSON object: {"rationale": <string>}. Read whatever '
        "you were given about THIS SPECIFIC requirement and report what its "
        "own rationale/justification actually says, in your own words, tied "
        "to its specific content (not a generic disclaimer). Only if, after "
        "actually checking, this specific requirement has no rationale "
        "recorded at all, say so in your own words explicitly referencing "
        "that you checked for it -- never output a stock phrase as a "
        "default, and never invent a plausible-sounding rationale that "
        "was not actually given to you."
    ),
    "lookup.value_property_attribute": (
        'Respond with a JSON object: {"value": <the attribute value, as a '
        "string or number>}. Find the specific element named in the "
        "question and report its actual recorded value -- do not respond "
        "with null or a placeholder unless you have actually looked and "
        "confirmed no value is recorded for THIS specific element."
    ),
}


_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def _word_to_int(text):
    """Parse English number words ("one hundred and twenty-three"); None if not a number."""
    import re

    words = re.split(r"[\s-]+", text.strip().lower().replace(",", ""))
    words = [w for w in words if w and w != "and"]
    if not words:
        return None
    total = 0
    current = 0
    matched_any = False
    for w in words:
        if w in _ONES:
            current += _ONES[w]
            matched_any = True
        elif w in _TENS:
            current += _TENS[w]
            matched_any = True
        elif w == "hundred":
            current = (current or 1) * 100
            matched_any = True
        else:
            return None
    total += current
    return total if matched_any else None


def _extract_scalar_value(text):
    """Recover a bare value from a prose answer such as "The default value is 4."."""
    import re

    if not isinstance(text, str):
        return text
    stripped = text.strip().rstrip(".")
    if re.fullmatch(r"-?\d+(\.\d+)?", stripped):
        return stripped
    quoted = re.search(r'"([^"]+)"', text)
    if quoted:
        return quoted.group(1)
    number_match = re.search(r"-?\d+(\.\d+)?", text)
    if number_match:
        return number_match.group(0)
    return stripped


def _resolve_elements(raw_list, name_to_ids, known_ids=None):
    """Map returned element ids or names to ids; returns (ids, number of unresolved names)."""
    if not raw_list:
        return [], 0
    resolved, n_unresolved = [], 0
    for item in raw_list:
        item_str = str(item)
        if known_ids is not None and item_str in known_ids:
            resolved.append(item_str)
            continue
        if name_to_ids is None:
            resolved.append(item_str)
            continue
        candidates = name_to_ids.get(item_str.strip().lower())
        if candidates:
            resolved.extend(candidates)
        else:
            n_unresolved += 1
    return resolved, n_unresolved


def extract_prediction(template_id, response, name_to_ids=None, node_name_to_id_for_path=None, known_ids=None):
    """Return (prediction, extraction_meta) for a parsed response; EXTRACTION_FAILED if the answer key is missing."""
    response = response or {}
    meta = {}

    if template_id in (
        "impact_analysis.requirement_closure",
        "impact_analysis.structural_closure",
        "negation.requirement_unrefined",
    ):
        if "answer_elements" not in response:
            meta["missing_key"] = "answer_elements"
            return EXTRACTION_FAILED, meta
        raw = response.get("answer_elements") or []
        resolved, n_unresolved = _resolve_elements(raw, name_to_ids, known_ids=known_ids)
        meta["n_raw"] = len(raw)
        meta["n_unresolved"] = n_unresolved
        return resolved, meta

    if template_id == "aggregation.requirement_count":
        if "count" not in response:
            meta["missing_key"] = "count"
            return EXTRACTION_FAILED, meta
        raw = response.get("count")
        if raw is None:
            return None, meta
        try:
            return int(raw), meta
        except (TypeError, ValueError):
            word_value = _word_to_int(str(raw)) if isinstance(raw, str) else None
            if word_value is not None:
                meta["parsed_from_words"] = raw
                return word_value, meta
            meta["unparseable_count"] = raw
            return None, meta

    if template_id == "tracing.requirement_derivation":
        if "path" not in response:
            meta["missing_key"] = "path"
            return EXTRACTION_FAILED, meta
        raw = response.get("path")
        if not raw:
            return None, meta
        edges = []
        for hop in raw:
            frm, to, rel = hop.get("from"), hop.get("to"), hop.get("relation")
            if node_name_to_id_for_path is not None:
                frm = node_name_to_id_for_path.get(str(frm).strip().lower(), frm)
                to = node_name_to_id_for_path.get(str(to).strip().lower(), to)
            edges.append({"from": frm, "to": to, "relation": rel})
        return edges, meta

    if template_id == "rationale_abstention":
        if "rationale" not in response:
            meta["missing_key"] = "rationale"
            return EXTRACTION_FAILED, meta
        raw = response.get("rationale")
        if raw is None:
            return None, meta
        return str(raw), meta

    if template_id == "lookup.value_property_attribute":
        if "value" not in response:
            meta["missing_key"] = "value"
            return EXTRACTION_FAILED, meta
        raw = response.get("value")
        if raw is None:
            return None, meta
        extracted = _extract_scalar_value(raw)
        if extracted != raw:
            meta["extracted_from_prose"] = raw
        return extracted, meta

    raise ValueError(f"no extraction contract for template_id {template_id!r}")
