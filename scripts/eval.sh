#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export EVENTSHIFT_ROOT="${EVENTSHIFT_ROOT:-${ROOT_DIR}}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
python -m tools.evaluate "$@"
