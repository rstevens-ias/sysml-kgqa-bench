"""Shared I/O helpers."""

import json

import yaml

TEMPLATE_FILES = [
    "impact_analysis.requirement_closure.yaml",
    "negation.requirement_unrefined.yaml",
    "impact_analysis.structural_closure.yaml",
    "aggregation.requirement_count.yaml",
    "tracing.requirement_derivation.yaml",
    "rationale_abstention.yaml",
    "lookup.value_property_attribute.yaml",
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records, ensure_ascii=True):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=ensure_ascii) + "\n")


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def record_key(record):
    """Unique key of a benchmark instance: template id plus its bindings."""
    return record["template_id"] + "|" + json.dumps(record["bindings"], sort_keys=True)
