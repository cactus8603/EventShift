#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CONDA_ENV:=mask2former}"
: "${EVENT_EDGE_CACHE_DIR:=work_dirs/diagnostics/cosec_event_edge_cache_p80_r25_50}"

conda run --no-capture-output -n "${CONDA_ENV}" \
  python tools/build_cosec_event_edge_cache.py \
  --datasets cosec_train_event,cosec_day_val_event,cosec_night_val_event \
  --out-dir "${EVENT_EDGE_CACHE_DIR}" \
  --window-radii-ms 25 50 \
  --percentile 80 \
  --reuse

echo "Event-edge cache ready: ${EVENT_EDGE_CACHE_DIR}"
