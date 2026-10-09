#!/usr/bin/env bash
# Download the released data (benchmark, intermediate files, baseline results) from Zenodo into data/.
set -euo pipefail
cd "$(dirname "$0")/.."

ZENODO_DOI="${ZENODO_DOI:-}"
ARCHIVE="sysml-kgqa-bench-data-v1.0.0.zip"
SHA256="6438b1979c84b9c20e08fce457b5fb5999454a8c5c379a086def3f31ae83ce9b"

if [ -z "$ZENODO_DOI" ]; then
    echo "Set ZENODO_DOI to the DOI of the dataset (e.g. 10.5281/zenodo.1234567)." >&2
    exit 1
fi
RECORD="${ZENODO_DOI##*zenodo.}"

mkdir -p data
curl -fL -o "data/$ARCHIVE" "https://zenodo.org/records/${RECORD}/files/${ARCHIVE}?download=1"
if [ "$(sha256sum "data/$ARCHIVE" | cut -d' ' -f1)" != "$SHA256" ]; then
    echo "Checksum mismatch for data/$ARCHIVE" >&2
    exit 1
fi
unzip -o -q "data/$ARCHIVE"
rm "data/$ARCHIVE"
echo "Data extracted to data/. Build the graph database with: python -m sysml_kgqa_bench.graph.load_kuzu"
