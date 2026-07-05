#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
cd "${ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CONDA_ENV="${CONDA_ENV:-mask2former}"

condition="${CONDITION:-all}"
fold_a="${FOLD_A:-1}"
fold_b="${FOLD_B:-2}"
gpu_a="${GPU_A:-0}"
gpu_b="${GPU_B:-1}"
batch_size="${IMS_PER_BATCH:-3}"
base_lr="${BASE_LR:-0.00000005}"
epochs="${EPOCHS:-3}"
weights="${WEIGHTS:-work_dirs/swinL_seqholdout305_from_latest_night/best_model_cosec_night.pth}"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required for persistent detached launches." >&2
  exit 1
fi

for path in \
  tools/run_swinl_acdc_kfold_train_export_real.sh \
  tools/run_swinl_acdc_kfold_headonly.sh \
  tools/export_acdc_kfold_real_candidate.sh \
  "${weights}"
do
  if [ ! -e "${path}" ]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

launch_fold() {
  local fold="$1"
  local gpu="$2"
  local name="swinl_acdc_${condition}_kfold${fold}_realonly"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_path="work_dirs/launch_logs/${name}_${stamp}.log"
  local output_dir="work_dirs/acdc_${condition}_kfold3_fold${fold}_headonly_from_night50_lr5e-8_bs${batch_size}"

  mkdir -p work_dirs/launch_logs
  CUDA_VISIBLE_DEVICES="${gpu}" screen -dmS "${name}" bash -lc \
    "cd '${ROOT}' && export CONDA_ENV='${CONDA_ENV}' PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 IMS_PER_BATCH='${batch_size}' BASE_LR='${base_lr}' EPOCHS='${epochs}' WEIGHTS='${weights}' OUTPUT_DIR='${output_dir}' DEVICE='cuda:0' EXPORT_RAW=0 EXPORT_TTA=1 && bash tools/run_swinl_acdc_kfold_train_export_real.sh '${condition}' '${fold}' > '${log_path}' 2>&1"
  echo "${name}: GPU ${gpu}, condition=${condition}, fold=${fold}, log=${log_path}"
}

launch_fold "${fold_a}" "${gpu_a}"
launch_fold "${fold_b}" "${gpu_b}"
