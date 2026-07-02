#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
TMP_DIR="${PD}/tmp/pair_sidecar"
IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
DEVICE="${PAIR_SIDECAR_DEVICE:-mps}"

rm -rf "$ARCHIVE_DIR" "$TMP_DIR"
mkdir -p "$ARCHIVE_DIR" "$TMP_DIR"

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"
  PRE="${TMP_DIR}/${BASE}.pre.mkv"
  SC="${ARCHIVE_DIR}/${BASE}.sidecar"

  cd "$PD"
  .venv/bin/python -m submissions.pair_sidecar.preprocess --input "$IN" --output "$PRE"
  .venv/bin/python -m submissions.pair_sidecar.encode_segments \
    --pre "$PRE" --out "$OUT" --work "${TMP_DIR}/${BASE}_seg"

  .venv/bin/python -m submissions.pair_sidecar.tune \
    --source "$IN" \
    --encoded "$OUT" \
    --sidecar-out "$SC" \
    --device "$DEVICE"

  rm -f "$PRE"
  rm -rf "${TMP_DIR}/${BASE}_seg"
done

rm -f "${HERE}/archive.zip"
cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
