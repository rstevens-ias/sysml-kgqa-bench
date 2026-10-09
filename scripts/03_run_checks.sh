#!/usr/bin/env bash
# Checks without LLM calls: unit and integration tests, trivial and name-similarity floors, and dry runs of
# the LLM baselines with canned responses.
set -euo pipefail
cd "$(dirname "$0")/.."

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

python -m pytest
python -m sysml_kgqa_bench.baselines.trivial
python -m sysml_kgqa_bench.baselines.name_similarity
for baseline in closed_book oracle text_to_cypher; do
    python -m "sysml_kgqa_bench.baselines.${baseline}" --dry-run --limit 20 --outdir "$TMP_DIR/$baseline"
done
