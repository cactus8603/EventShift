#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export EVENTSHIFT_ROOT="${EVENTSHIFT_ROOT:-${ROOT_DIR}}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi
DEFAULT_CONFIG="configs/eventshift/cosec_eventshift.yaml"

has_selector=0
for arg in "$@"; do
  case "${arg}" in
    --config|--config=*|--model|--model=*|--variant|--variant=*)
      has_selector=1
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  args=(--config "${DEFAULT_CONFIG}")
elif [[ "${1}" != --* ]]; then
  config="${1}"
  shift
  args=(--config "${config}" "$@")
elif [[ "${has_selector}" -eq 0 ]]; then
  args=(--config "${DEFAULT_CONFIG}" "$@")
else
  args=("$@")
fi

"${PYTHON_BIN}" -m tools.train "${args[@]}"
