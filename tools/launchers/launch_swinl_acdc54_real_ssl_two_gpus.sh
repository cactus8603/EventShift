#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

launch_one() {
  local name="$1"
  local gpu="$2"
  local runner="$3"
  local out_dir="$4"

  mkdir -p "${out_dir}"
  if pgrep -af "${runner}" >/dev/null 2>&1; then
    echo "[skip] ${name} already appears to be running"
    return 0
  fi

  echo "[launch] ${name} on GPU ${gpu}"
  if command -v screen >/dev/null 2>&1; then
    local session_name
    session_name="$(echo "${name}" | tr -c '[:alnum:]_.-' '_')"
    screen -dmS "${session_name}" bash -lc "
      set -euo pipefail
      cd '$(pwd)'
      export PYTHONNOUSERSITE=1
      export CONDA_ENV='${CONDA_ENV:-mask2former}'
      export CUDA_VISIBLE_DEVICES='${gpu}'
      echo '[launch-log] name=${name} gpu=${gpu} runner=${runner} started='\"\$(date --iso-8601=seconds)\" > '${out_dir}/train.log'
      exec bash '${runner}' >> '${out_dir}/train.log' 2>&1
    "
    echo "${session_name}" > "${out_dir}/screen_session.txt"
    echo "[screen] ${name}: ${session_name}"
  else
    setsid bash -lc "
      set -euo pipefail
      cd '$(pwd)'
      export PYTHONNOUSERSITE=1
      export CONDA_ENV='${CONDA_ENV:-mask2former}'
      export CUDA_VISIBLE_DEVICES='${gpu}'
      echo '[launch-log] name=${name} gpu=${gpu} runner=${runner} started='\"\$(date --iso-8601=seconds)\" > '${out_dir}/train.log'
      exec bash '${runner}' >> '${out_dir}/train.log' 2>&1
    " >/dev/null 2>&1 < /dev/null &
    echo "$!" > "${out_dir}/pid.txt"
    echo "[pid] ${name}: $(cat "${out_dir}/pid.txt")"
  fi
}

launch_one \
  "real-ssl-acdc54-eventedge" \
  "${EVENTEDGE_GPU:-0}" \
  "tools/run_swinl_acdc54_real_ssl_eventedge.sh" \
  "work_dirs/real-ssl-acdc54_eventedge_headonly_lr2e-7_bs3"

launch_one \
  "real-ssl-acdc54-eventactive" \
  "${EVENTACTIVE_GPU:-1}" \
  "tools/run_swinl_acdc54_real_ssl_eventactive.sh" \
  "work_dirs/real-ssl-acdc54_eventactive_headonly_lr1e-7_bs3"
