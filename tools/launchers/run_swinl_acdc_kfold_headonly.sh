#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

condition="${1:-night}"
fold="${2:-0}"
folds="${FOLDS:-3}"
batch_size="${IMS_PER_BATCH:-3}"
base_lr="${BASE_LR:-0.00000005}"
epochs="${EPOCHS:-3}"
weights="${WEIGHTS:-work_dirs/swinL_seqholdout305_from_latest_night/best_model_cosec_night.pth}"
output_dir="${OUTPUT_DIR:-work_dirs/acdc_${condition}_kfold${folds}_fold${fold}_headonly_from_night50_lr5e-8_bs${batch_size}}"
conda_env="${CONDA_ENV:-mask2former}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

train_dataset="acdc_${condition}_kfold${folds}_fold${fold}_train"
val_dataset="acdc_${condition}_kfold${folds}_fold${fold}_val"

num_train_samples="$(
conda run --no-capture-output -n "${conda_env}" python - "${condition}" "${folds}" "${fold}" <<'PY'
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tools"))

from acdc_dataset import load_acdc_kfold_dicts

condition = sys.argv[1]
folds = int(sys.argv[2])
fold = int(sys.argv[3])
print(len(load_acdc_kfold_dicts(condition, folds, fold, "train")))
PY
)"

conda run --no-capture-output -n "${conda_env}" \
  python tools/train_mask2former_cosec.py \
  --num-gpus 1 \
  --config-file configs/Mask2Former_SwinL_ACDC_KFold3_HeadOnly.yaml \
    DATASETS.TRAIN "(\"${train_dataset}\",)" \
    DATASETS.TEST "(\"${val_dataset}\", \"cosec_night_val\")" \
    MODEL.WEIGHTS "${weights}" \
    SOLVER.IMS_PER_BATCH "${batch_size}" \
    SOLVER.BASE_LR "${base_lr}" \
    TRAIN.EPOCHS "${epochs}" \
    TRAIN.NUM_TRAIN_SAMPLES "${num_train_samples}" \
    OUTPUT_DIR "${output_dir}"
