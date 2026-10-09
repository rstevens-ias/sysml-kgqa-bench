#!/usr/bin/env bash
# Download the TMT SysML model (Open-MBEE/TMT-SysML-Model, Apache-2.0) at a pinned commit and verify checksums.
#   EA/TMT_mtip.zip  MTIP export, primary source of the extraction
#   TMT.mdzip        Cameo project; only the raw XMI member is used
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_SHA="b2a33b71c65678498b538195dcb947a7962ca75b"
BASE_URL="https://raw.githubusercontent.com/Open-MBEE/TMT-SysML-Model/${REPO_SHA}"
OUT_DIR="data/source"
mkdir -p "$OUT_DIR"

fetch() {
    local url="$1" path="$2" sha256="$3"
    if [ ! -f "$path" ] || [ "$(sha256sum "$path" | cut -d' ' -f1)" != "$sha256" ]; then
        curl -fsSL -o "$path" "$url"
    fi
    if [ "$(sha256sum "$path" | cut -d' ' -f1)" != "$sha256" ]; then
        echo "Checksum mismatch for $path" >&2
        exit 1
    fi
}

fetch "${BASE_URL}/EA/TMT_mtip.zip" "$OUT_DIR/TMT_mtip.zip" \
    "ca065bfb6019f86303c667205d80bc3d8b0b90b26b8b570c50a2ab13186095e4"
unzip -o -q "$OUT_DIR/TMT_mtip.zip" -d "$OUT_DIR"

fetch "${BASE_URL}/TMT.mdzip" "$OUT_DIR/TMT.mdzip" \
    "9ffd7a2c3b7f3fca8f64092949309012bb75cad5f86da2edfee140ab49d20581"
unzip -o -q "$OUT_DIR/TMT.mdzip" "com.nomagic.magicdraw.uml_model.model" -d "$OUT_DIR"

echo "TMT model ready in $OUT_DIR (commit $REPO_SHA)."
