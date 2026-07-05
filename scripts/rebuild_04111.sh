#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export EVENTSHIFT_ROOT="${EVENTSHIFT_ROOT:-${ROOT_DIR}}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

ORIGINAL_ARGS=("$@")
CONDA_BIN="${CONDA:-}"
RUNNER_ENV="${M2F_ENV:-mask2former}"
SHOW_HELP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda)
      CONDA_BIN="${2:-}"
      shift 2
      ;;
    --conda=*)
      CONDA_BIN="${1#--conda=}"
      shift
      ;;
    --m2f-env)
      RUNNER_ENV="${2:-}"
      shift 2
      ;;
    --m2f-env=*)
      RUNNER_ENV="${1#--m2f-env=}"
      shift
      ;;
    -h|--help)
      SHOW_HELP=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -n "${PYTHON:-}" ]]; then
  exec "${PYTHON}" -m tools.rebuild.rebuild_04111 "${ORIGINAL_ARGS[@]}"
fi

if [[ "${SHOW_HELP}" -eq 0 ]]; then
  if [[ -z "${CONDA_BIN}" ]]; then
    if command -v conda >/dev/null 2>&1; then
      CONDA_BIN="conda"
    elif [[ -x /root/miniconda3/bin/conda ]]; then
      CONDA_BIN="/root/miniconda3/bin/conda"
    elif [[ -x /home/u1621738/miniconda3/bin/conda ]]; then
      CONDA_BIN="/home/u1621738/miniconda3/bin/conda"
    fi
  fi
  if [[ -n "${CONDA_BIN}" ]]; then
    exec "${CONDA_BIN}" run --no-capture-output -n "${RUNNER_ENV}" python -m tools.rebuild.rebuild_04111 "${ORIGINAL_ARGS[@]}"
  fi
fi

if command -v python >/dev/null 2>&1; then
  exec python -m tools.rebuild.rebuild_04111 "${ORIGINAL_ARGS[@]}"
fi
exec python3 -m tools.rebuild.rebuild_04111 "${ORIGINAL_ARGS[@]}"
