#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export EVENTSHIFT_ROOT="${EVENTSHIFT_ROOT:-${ROOT_DIR}}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

ORIGINAL_ARGS=("$@")
CONDA_BIN=""
RUNNER_ENV="${CONDA_DEFAULT_ENV:-}"
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

if [[ -n "${CONDA_BIN}" && -z "${RUNNER_ENV}" ]]; then
  RUNNER_ENV="ebmv_seg"
fi

if [[ "${SHOW_HELP}" -eq 0 && -n "${CONDA_BIN}" && -n "${RUNNER_ENV}" ]]; then
  exec "${CONDA_BIN}" run --no-capture-output -n "${RUNNER_ENV}" python -m tools.rebuild.rebuild_04111 "${ORIGINAL_ARGS[@]}"
fi

if command -v python >/dev/null 2>&1; then
  exec python -m tools.rebuild.rebuild_04111 "${ORIGINAL_ARGS[@]}"
fi
exec python3 -m tools.rebuild.rebuild_04111 "${ORIGINAL_ARGS[@]}"
