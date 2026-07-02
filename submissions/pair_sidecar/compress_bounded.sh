#!/usr/bin/env bash
# Primary stall-free path: classical encode + sidecar tune with hard budgets.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"
TMP_DIR="${PD}/tmp/pair_sidecar"
IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
DEVICE="${PAIR_SIDECAR_DEVICE:-mps}"
PY="${PD}/.venv/bin/python"

BUDGET_PREPROCESS="${BUDGET_PREPROCESS:-120}"
BUDGET_ENCODE="${BUDGET_ENCODE:-900}"
BUDGET_TUNE="${BUDGET_TUNE:-3600}"
TUNE_FAST="${TUNE_FAST:-1}"

run_phase() {
  local budget="$1" label="$2"
  shift 2
  "$PY" "${PD}/submissions/stall_guard.py" "$budget" "$label" "$@"
}

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
  run_phase "$BUDGET_PREPROCESS" preprocess \
    "$PY" -m submissions.pair_sidecar.preprocess --input "$IN" --output "$PRE"

  run_phase "$BUDGET_ENCODE" encode \
    "$PY" -m submissions.pair_sidecar.encode_segments \
    --pre "$PRE" --out "$OUT" --work "${TMP_DIR}/${BASE}_seg"

  TUNE_ARGS=(--source "$IN" --encoded "$OUT" --sidecar-out "$SC" --device "$DEVICE")
  if [ "$TUNE_FAST" = "1" ]; then
    TUNE_ARGS+=(--fast)
  fi
  run_phase "$BUDGET_TUNE" tune \
    "$PY" -m submissions.pair_sidecar.tune "${TUNE_ARGS[@]}"

  rm -f "$PRE"
  rm -rf "${TMP_DIR}/${BASE}_seg"
done

rm -f "${HERE}/archive.zip"
cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip ($(wc -c < "${HERE}/archive.zip") bytes)"
