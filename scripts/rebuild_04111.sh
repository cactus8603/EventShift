#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export EVENTSHIFT_ROOT="${EVENTSHIFT_ROOT:-${ROOT_DIR}}"
exec bash "${EVENTSHIFT_ROOT}/tools/rebuild/rebuild_04111_b75_from_bundle_checkpoints.sh" "$@"
