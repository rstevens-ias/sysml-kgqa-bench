#!/usr/bin/env bash
# Run the LLM baselines on the full benchmark (both question forms).
# Needs PRIVATE_API_KEY and PRIVATE_ENDPOINT for an OpenAI-compatible endpoint serving the model below.
# Usage: scripts/04_run_baselines.sh [closed_book|oracle|gretriever|text_to_cypher|all]
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-all}"
MODEL="${BASELINE_MODEL:-qwen-3-5-397b-a17b}"

run() {
    local baseline="$1"
    if [ "$baseline" = gretriever ]; then
        [ -f data/pcst/pool.json ] || python -m sysml_kgqa_bench.baselines.build_pcst_pool
        [ -f data/pcst/pool_embeddings.npz ] || python -m sysml_kgqa_bench.baselines.embed_pcst_pool
    fi
    python -m "sysml_kgqa_bench.baselines.${baseline}" --backend private --model "$MODEL"
}

case "$TARGET" in
    closed_book | oracle | gretriever | text_to_cypher) run "$TARGET" ;;
    all) for b in closed_book oracle gretriever text_to_cypher; do run "$b"; done ;;
    *)
        echo "Unknown baseline: $TARGET" >&2
        exit 1
        ;;
esac
