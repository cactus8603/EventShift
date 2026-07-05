#!/usr/bin/env bash
set -euo pipefail

cd .

conda run --no-capture-output -n mask2former \
  python tools/train_mask2former_cosec.py \
  --config-file configs/Mask2Former_SwinL_CoSEC_KFold3_Fold0_DayHeadOnly_Smoke.yaml \
  --num-gpus 1
