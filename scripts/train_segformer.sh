#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${1:-$BUNDLE_ROOT/configs/mmseg/SegFormer_B5_FullCoSEC_FromNightBest_Floor546453.py}"
WORK_DIR="${WORK_DIR:-$BUNDLE_ROOT/work_dirs/train_segformer_bundle}"

shift || true

export PYTHONPATH="$BUNDLE_ROOT:$BUNDLE_ROOT/tools:$BUNDLE_ROOT/third_party/mmsegmentation:${PYTHONPATH:-}"

cd "$BUNDLE_ROOT"
"$PYTHON_BIN" third_party/mmsegmentation/tools/train.py \
  "$CONFIG" \
  --work-dir "$WORK_DIR" \
  "$@"
