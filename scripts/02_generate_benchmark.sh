#!/usr/bin/env bash
# Generate the benchmark: schema catalogue, template instantiation, sampling, leakage audit and verbalization.
# Verbalization calls the OpenAI API (gpt-4.1-mini generator, gpt-4.1 judge) and needs OPENAI_API_KEY.
# Extra arguments are passed to the verbalization step.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m sysml_kgqa_bench.generation.catalogue
python -m sysml_kgqa_bench.generation.instantiate
python -m sysml_kgqa_bench.generation.sample
python -m sysml_kgqa_bench.generation.leakage_audit
python -m sysml_kgqa_bench.generation.verbalize "$@"
