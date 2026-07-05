#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE=1
: "${CONDA_ENV:=mask2former}"
: "${GPU_TRAIN_ROUTE_DAY:=0}"
: "${GPU_TRAIN_ROUTE_NIGHT:=1}"

cuda_count="$(
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
)"

if [ "${cuda_count}" -lt 2 ]; then
  echo "Need 2 visible CUDA GPUs, but found ${cuda_count}." >&2
  exit 1
fi

log_dir="work_dirs/launch_logs"
mkdir -p "${log_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"

config_file="configs/Mask2Former_SwinL_CoSEC_DayNight_Finetune.yaml"
weights="work_dirs/swinL_cosec_dayonly_from_day65_freeze_backbone_lr5e-7/best_model_cosec_day.pth"
branches="base624flip=s624:flip,highres768flip=s768:flip,tta3flip=s512+s768+s1024:flip,tta4flip=s512+s624+s768+s1024:flip"

day_log="${log_dir}/train-learned-scale-router-day_${stamp}.log"
night_log="${log_dir}/train-learned-scale-router-night_${stamp}.log"
day_session="train_route_day_${stamp}"
night_session="train_route_night_${stamp}"

screen -dmS "${day_session}" bash -lc "
cd '${PWD}'
export PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES='${GPU_TRAIN_ROUTE_DAY}' conda run --no-capture-output -n '${CONDA_ENV}' \
  python tools/diagnose_train_learned_scale_branch_router.py \
  --config-file '${config_file}' \
  --weights '${weights}' \
  --train-dataset cosec_day_train \
  --eval-dataset cosec_day_val \
  --train-limit 64 \
  --eval-limit 175 \
  --chunk-size 32 \
  --device cuda:0 \
  --branches '${branches}' \
  --anchor tta4flip \
  --basis tta4flip \
  --min-delta 0.01 \
  --min-routed-pixels 1000 \
  --out work_dirs/diagnostics/train_learned_scale_router_day_train64_chunk32_evalfull.json \
  > '${day_log}' 2>&1
"

screen -dmS "${night_session}" bash -lc "
cd '${PWD}'
export PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES='${GPU_TRAIN_ROUTE_NIGHT}' conda run --no-capture-output -n '${CONDA_ENV}' \
  python tools/diagnose_train_learned_scale_branch_router.py \
  --config-file '${config_file}' \
  --weights '${weights}' \
  --train-dataset cosec_night_train \
  --eval-dataset cosec_night_val \
  --train-limit 64 \
  --eval-limit 130 \
  --chunk-size 32 \
  --device cuda:0 \
  --branches '${branches}' \
  --anchor tta4flip \
  --basis tta4flip \
  --min-delta 0.01 \
  --min-routed-pixels 1000 \
  --out work_dirs/diagnostics/train_learned_scale_router_night_train64_chunk32_evalfull.json \
  > '${night_log}' 2>&1
"

echo "Launched train-learned scale router Day on GPU ${GPU_TRAIN_ROUTE_DAY}: session=${day_session}, log=${day_log}"
echo "Launched train-learned scale router Night on GPU ${GPU_TRAIN_ROUTE_NIGHT}: session=${night_session}, log=${night_log}"
