#!/usr/bin/env bash
# Download the TMT model, extract the knowledge base, build the graph and load it into Kuzu.
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/download_tmt.sh
python -m sysml_kgqa_bench.graph.extract
python -m sysml_kgqa_bench.graph.build
python -m sysml_kgqa_bench.graph.load_kuzu
