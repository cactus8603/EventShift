#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export EVENTSHIFT_ROOT="${EVENTSHIFT_ROOT:-${ROOT_DIR}}"
mkdir -p "${EVENTSHIFT_ROOT}/data" "${EVENTSHIFT_ROOT}/checkpoints" "${EVENTSHIFT_ROOT}/outputs"
cat <<MSG
EventShift workspace directories are ready.

Raw datasets are not stored in this repository. Prepare them externally and set:

  export BRENET_ROOT=/path/to/BRENet
  export COSEC_ROOT=/path/to/cosec
  export DSEC_ROOT=/path/to/dsec
  export ACDC_ROOT=/path/to/acdc
  export TEST_ROOT=/path/to/test

To rebuild checkpoints locally:

  bash scripts/train.sh configs/eventshift/cosec_eventshift.yaml --execute
MSG
