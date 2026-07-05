#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="$BUNDLE_ROOT/training/maskdino_swinl_runtime"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${1:-$RUNTIME_ROOT/configs/cosec/semantic-segmentation/maskdino_swinl_full_dsec_cosec_acdc_unified_classcover_v1.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$BUNDLE_ROOT/work_dirs/train_maskdino_bundle}"

shift || true

export SKIP_CODE_BACKUP="${SKIP_CODE_BACKUP:-1}"
export PYTHONPATH="$RUNTIME_ROOT:$BUNDLE_ROOT/training/swin_l/tools:$BUNDLE_ROOT/third_party/detectron2:${PYTHONPATH:-}"

cd "$RUNTIME_ROOT"
"$PYTHON_BIN" tools/train_maskdino_cosec.py \
  --config-file "$CONFIG" \
  --num-gpus "$NUM_GPUS" \
  OUTPUT_DIR "$OUTPUT_DIR" \
  "$@"
