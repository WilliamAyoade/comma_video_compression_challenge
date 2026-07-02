#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  VIDEO="${DATA_DIR}/${BASE}.mkv"
  SC="${DATA_DIR}/${BASE}.sidecar"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  for f in "$VIDEO" "$SC"; do
    [ ! -f "$f" ] && echo "ERROR: ${f} not found" >&2 && exit 1
  done
  cd "$ROOT"
  python -m "submissions.${SUB_NAME}.inflate" "$VIDEO" "$SC" "$DST"
done < "$FILE_LIST"
