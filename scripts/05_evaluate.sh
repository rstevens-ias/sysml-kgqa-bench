#!/usr/bin/env bash
# Print the result tables and compare them with the numbers reported in the paper.
# Extra arguments are passed to compare_to_paper (e.g. --baseline oracle, --html report.html).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m sysml_kgqa_bench.evaluation.analyze
python -m sysml_kgqa_bench.evaluation.compare_to_paper "$@"
