#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export EVENTSHIFT_ROOT="${EVENTSHIFT_ROOT:-${ROOT_DIR}}"
mkdir -p \
  "${EVENTSHIFT_ROOT}/data" \
  "${EVENTSHIFT_ROOT}/checkpoints" \
  "${EVENTSHIFT_ROOT}/outputs" \
  "${EVENTSHIFT_ROOT}/work_dirs/cache" \
  "${EVENTSHIFT_ROOT}/work_dirs/diagnostics" \
  "${EVENTSHIFT_ROOT}/work_dirs/manifests" \
  "${EVENTSHIFT_ROOT}/work_dirs/mmseg"
cat <<'MSG'
EventShift workspace directories are ready.

Raw datasets are not stored in this repository. See:

  docs/dataset_preparation.md

Common path args:

  --test-root        CoSEC test root for inference and submission export
  --cosec-root       CoSEC training root containing Day_* and Night_* sequences
  --brenet-root      BRENet root containing CoSEC event assets for event-based training
  --cosec-manifest   CoSEC event manifest JSON for event-based training
  --dsec-root        DSEC dataset root for auxiliary training data
  --acdc-root        ACDC dataset root for auxiliary training data

Build leakage-free CoSEC sequence-level k-fold splits:

  python tools/data/build_cosec_kfold_splits.py \
    --root /path/to/cosec/train \
    --folds 3 \
    --write-splits \
    --split-dir /path/to/cosec/splits

Build a CoSEC frame-list prefix split for domain/class coverage:

  python tools/data/build_cosec_domain_cover_frame_split.py \
    --root /path/to/cosec/train \
    --prefix domaincover20 \
    --val-fraction 0.20 \
    --write-splits \
    --split-dir /path/to/cosec/splits

Train with args-first config selection:

  bash scripts/train.sh \
    --model mask2former \
    --variant eventshift \
    --cosec-root /path/to/cosec/train \
    --brenet-root /path/to/BRENet \
    --cosec-manifest /path/to/cosec_train_bidir_50ms.json

To rebuild the 0.4111 submission:

  bash scripts/rebuild_04111.sh \
    --test-root /path/to/cosec/test \
    --conda /root/miniconda3/bin/conda \
    --m2f-env ebmv_seg \
    --mmseg-env ebmv_seg
MSG
