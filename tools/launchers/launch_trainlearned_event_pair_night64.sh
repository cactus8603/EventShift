#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
cd "${ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
: "${CONDA_ENV:=mask2former}"
: "${GPU_EVENT_PAIR_NIGHT:=1}"

log_dir="work_dirs/launch_logs"
mkdir -p "${log_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"
session="trainlearned_event_pair_night64_${stamp}"
log_path="${log_dir}/${session}.log"
out_path="work_dirs/diagnostics/trainlearned_tta_event_pair_router_night_train64_evalfull_strict.json"

screen -dmS "${session}" bash -lc "
  cd '${ROOT}' &&
  export PYTHONNOUSERSITE=1 &&
  export PYTHONUNBUFFERED=1 &&
  CUDA_VISIBLE_DEVICES='${GPU_EVENT_PAIR_NIGHT}' conda run --no-capture-output -n '${CONDA_ENV}' \
    python -u tools/diagnose_train_learned_tta_event_pair_router.py \
    --base-config configs/Mask2Former_SwinL_CoSEC_DayNight_Finetune.yaml \
    --base-weights work_dirs/swinL_cosec_dayonly_from_day65_freeze_backbone_lr5e-7/best_model_cosec_day.pth \
    --event-config configs/Mask2Former_SwinL_CoSEC_FullCoSEC_Exp11B_Night50EventActiveUncertainScore.yaml \
    --event-weights work_dirs/fullcosec-exp11b_night50_event_uncertain_score_bs6/best_model_cosec_night.pth \
    --train-dataset cosec_night_train_event \
    --eval-dataset cosec_night_val_event \
    --train-limit 64 \
    --device cuda:0 \
    --scale-set s512+s624+s768+s1024 \
    --flip \
    --regions raw,support,event_union \
    --boundary-radii 0,1,3 \
    --base-conf-thresholds 0.0,0.6,0.8 \
    --event-conf-thresholds 0.0,0.4,0.6 \
    --margin-modes none,event_ge_base \
    --min-net 100 \
    --min-precision 0.7 \
    --min-changed 100 \
    --max-pairs 12 \
    --out '${out_path}' \
    > '${log_path}' 2>&1
"

echo "Launched ${session} on GPU ${GPU_EVENT_PAIR_NIGHT}"
echo "Log: ${log_path}"
echo "Output: ${out_path}"
