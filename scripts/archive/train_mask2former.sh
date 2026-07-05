#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${1:-$BUNDLE_ROOT/configs/Mask2Former_SwinL_FullCoSEC_FromDayBest_Floor816070_LR5e-7.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$BUNDLE_ROOT/work_dirs/train_mask2former_bundle}"

shift || true

export SKIP_CODE_BACKUP="${SKIP_CODE_BACKUP:-1}"
export PYTHONPATH="$BUNDLE_ROOT/tools:$BUNDLE_ROOT/third_party/Mask2Former:$BUNDLE_ROOT/third_party/detectron2:${PYTHONPATH:-}"

cd "$BUNDLE_ROOT"
"$PYTHON_BIN" tools/train_mask2former_cosec.py \
  --config-file "$CONFIG" \
  --num-gpus "$NUM_GPUS" \
  OUTPUT_DIR "$OUTPUT_DIR" \
  "$@"
