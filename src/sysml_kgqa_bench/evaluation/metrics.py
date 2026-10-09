"""Per-template metrics, keyed by each template's correctness guarantee.

Metrics compare an already extracted prediction with a gold record; answer extraction lives in extraction.py.
"""

import math
import re
from collections import defaultdict

TEMPLATE_GUARANTEE = {
    "impact_analysis.requirement_closure": "completeness",  # T1
    "negation.requirement_unrefined": "exactness_set",  # T2
    "impact_analysis.structural_closure": "completeness",  # T3
    "aggregation.requirement_count": "exactness_scalar",  # T4
    "tracing.requirement_derivation": "faithfulness_path",  # T5
    "rationale_abstention": "abstention",  # T6
    "lookup.value_property_attribute": "exactness_scalar",  # T7
}

# A closure answer "stopped early" if the gold set has >= 10 elements but at most 3 were predicted.
EARLY_STOP_MAX_ELEMENTS = 3
EARLY_STOP_MIN_GOLD_SIZE = 10


def set_metrics(predicted, gold):
    """Completeness (recall) and faithfulness (precision) of a predicted set against a non-empty gold set."""
    predicted, gold = set(predicted), set(gold)
    if not gold:
        raise ValueError("set_metrics requires a non-empty gold set")
    completeness = len(predicted & gold) / len(gold)
    faithfulness = 1.0 if not predicted else len(predicted & gold) / len(predicted)
    return {"completeness": completeness, "faithfulness": faithfulness}


def exact_set_match(predicted, gold):
    """1.0 if the sets are equal, else 0.0."""
    return 1.0 if set(predicted) == set(gold) else 0.0


def _normalize_scalar(x):
    if x is None:
        return None
    s = str(x).strip()
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f.is_integer() else repr(f)
    except ValueError:
        return s.lower()


def exact_scalar_match(predicted, gold):
    """1.0 if the values are equal after numeric and case normalization, else 0.0."""
    return 1.0 if _normalize_scalar(predicted) == _normalize_scalar(gold) else 0.0


def path_predicate_metrics(predicted_edges, source, target, edge_exists_fn):
    """Edge existence (share of asserted edges present in the graph) and endpoint connection (whether the
    existing asserted edges connect `source` to `target`, in any order).

    predicted_edges: list of {"from", "to", "relation"} or None.
    """
    if not predicted_edges:
        return {"edge_existence": 0.0, "endpoint_connection": 0.0}

    real_flags = [edge_exists_fn(e["from"], e["to"], e["relation"]) for e in predicted_edges]
    edge_existence = sum(real_flags) / len(real_flags)

    adjacency = defaultdict(list)
    for e, real in zip(predicted_edges, real_flags):
        if real:
            adjacency[e["from"]].append(e["to"])

    visited = {source}
    frontier = [source]
    while frontier:
        next_frontier = []
        for node in frontier:
            for neighbor in adjacency.get(node, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    connected = target in visited

    return {"edge_existence": edge_existence, "endpoint_connection": 1.0 if connected else 0.0}


def build_edge_exists_fn(backend, relations, deprecated_stereotypes, downstream_direction):
    """Load the trace graph once and return edge_exists(from_id, to_id, relation) -> bool."""
    from sysml_kgqa_bench.generation.instantiate import build_directed_adjacency, load_relation_edges

    edges, _ = load_relation_edges(backend, relations, deprecated_stereotypes)
    adj = build_directed_adjacency(edges, downstream_direction)
    edge_set = {(node, neighbor, rel) for node, neighs in adj.items() for neighbor, rel in neighs}

    def edge_exists(from_id, to_id, relation):
        return (from_id, to_id, relation) in edge_set

    return edge_exists


# Phrases that mark an answer as an abstention.
_ABSTENTION_MARKERS = (
    "no rationale",
    "not recorded",
    "not available",
    "not stated",
    "not provided",
    "no reason",
    "cannot determine",
    "cannot be determined",
    "unknown",
    "not specified",
    "no information",
    "unable to determine",
    "not documented",
    "no justification",
    "does not document",
    "does not provide",
    "does not state",
    "does not specify",
    "i don't know",
    "i do not know",
    "n/a",
    "doesn't appear so",
    "does not appear so",
    "not that i can find",
    "don't see one",
    "do not see one",
    "not that i can tell",
    "no such rationale",
    "no way to answer",
    "no way to determine",
    "cannot be answered",
    "can't be answered",
    "insufficient information",
    "not enough information",
    "cannot answer",
)

# Matches "no rationale", "no stated rationale", "no documented reasoning", ...
_NO_FIELD_PATTERN = re.compile(r"\bno\s+(?:\w+\s+){0,1}(?:rationale|reasons?|reasoning|justification)\b")


def _matches_no_field_pattern(t):
    return bool(_NO_FIELD_PATTERN.search(t))


def looks_like_abstention(text):
    """Keyword-based detection of an abstaining answer."""
    if text is None:
        return True
    t = text.strip().lower()
    if not t:
        return True
    return any(marker in t for marker in _ABSTENTION_MARKERS) or _matches_no_field_pattern(t)


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class LASMNotWired(RuntimeError):
    """Raised when an answerable T6 instance is scored without an embedding function."""


# Cosine-similarity threshold for LASM, calibrated for Qwen3-Embedding-0.6B.
LASM_THRESHOLD_DEFAULT = 0.68


def abstention_metrics(
    predicted_text, answerable, gold_text=None, embed_fn=None, lasm_threshold=LASM_THRESHOLD_DEFAULT
):
    """T6: abstention accuracy on unanswerable instances, LASM accuracy (embedding similarity to the gold
    rationale) on answerable ones."""
    abstained = looks_like_abstention(predicted_text)
    if not answerable:
        return {"metric": "abstention_accuracy", "correct": 1.0 if abstained else 0.0, "abstained": abstained}

    if abstained:
        return {"metric": "lasm_accuracy", "correct": 0.0, "similarity": 0.0, "abstained": True}
    if embed_fn is None:
        raise LASMNotWired(
            "Answerable T6 instances need an embed_fn for LASM scoring (see sysml_kgqa_bench.embeddings)."
        )
    sim = cosine_similarity(embed_fn(predicted_text), embed_fn(gold_text))
    return {
        "metric": "lasm_accuracy",
        "correct": 1.0 if sim >= lasm_threshold else 0.0,
        "similarity": sim,
        "abstained": False,
    }


class _ExtractionFailedType:
    """Sentinel for a response that lacks the requested answer field (distinct from an explicit null)."""

    def __repr__(self):
        return "EXTRACTION_FAILED"

    def __bool__(self):
        return False


EXTRACTION_FAILED = _ExtractionFailedType()


def score_instance(record, predicted, edge_exists_fn=None, embed_fn=None, lasm_threshold=LASM_THRESHOLD_DEFAULT):
    """Score one prediction against its gold record with the metric of the record's template."""
    if predicted is EXTRACTION_FAILED:
        return {"extraction_failed": True}

    template_id = record["template_id"]
    guarantee = TEMPLATE_GUARANTEE.get(template_id)
    if guarantee is None:
        raise ValueError(f"no metric registered for template_id {template_id!r}")

    if guarantee == "completeness":
        # The seed itself is removed from the prediction and reported as `self_inclusion`.
        seed_id = record.get("bindings", {}).get("X")
        n_predicted_raw = len(set(predicted))
        predicted_set = set(predicted)
        self_inclusion = seed_id is not None and seed_id in predicted_set
        if self_inclusion:
            predicted_set = predicted_set - {seed_id}
        result = set_metrics(predicted_set, record["answer"])
        result["self_inclusion"] = self_inclusion
        result["stopped_early"] = (
            len(record["answer"]) >= EARLY_STOP_MIN_GOLD_SIZE and n_predicted_raw <= EARLY_STOP_MAX_ELEMENTS
        )
        return result
    if guarantee == "exactness_set":
        return {"exact_match": exact_set_match(predicted, record["answer"])}
    if guarantee == "exactness_scalar":
        return {"exact_match": exact_scalar_match(predicted, record["answer"])}
    if guarantee == "faithfulness_path":
        if edge_exists_fn is None:
            raise ValueError("tracing.requirement_derivation scoring requires edge_exists_fn")
        source, target = record["bindings"]["source"], record["bindings"]["target"]
        return path_predicate_metrics(predicted, source, target, edge_exists_fn)
    if guarantee == "abstention":
        return abstention_metrics(
            predicted,
            record["answerable"],
            gold_text=record["answer"],
            embed_fn=embed_fn,
            lasm_threshold=lasm_threshold,
        )
    raise ValueError(f"unhandled guarantee {guarantee!r} for {template_id!r}")
