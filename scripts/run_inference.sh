#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_inference.sh [overall|day|night] [additional inference options]

Runs one of the best EventShift checkpoint presets on CoSEC test data.

Defaults:
  data:   /data/cosec/test
  output: /data/predictions/eventshift/<preset>

Examples:
  bash scripts/run_inference.sh
  bash scripts/run_inference.sh night --device cuda:0
  bash scripts/run_inference.sh day --test-root /other/cosec/test --out-dir /other/predictions/day

Use TEST_ROOT or EVENTSHIFT_OUTPUT_ROOT to change the default locations.
EOF
}

preset="overall"
if [[ $# -gt 0 ]]; then
  case "$1" in
    overall|day|night)
      preset="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
  esac
fi

case "${preset}" in
  overall) variant="best_overall_b1_lambda003" ;;
  day) variant="best_day_b0_zero_event" ;;
  night) variant="best_night_b4_sr025" ;;
esac

test_root="${TEST_ROOT:-/data/cosec/test}"
for ((index = 1; index <= $#; index++)); do
  argument="${!index}"
  case "${argument}" in
    --test-root)
      next_index=$((index + 1))
      if [[ ${next_index} -le $# ]]; then
        test_root="${!next_index}"
      fi
      ;;
    --test-root=*)
      test_root="${argument#--test-root=}"
      ;;
  esac
done

out_dir="${EVENTSHIFT_OUTPUT_ROOT:-/data/predictions/eventshift}/${preset}"
checkpoint="${ROOT_DIR}/checkpoints/${variant}.pth"

if [[ ! -d "${test_root}" ]]; then
  echo "CoSEC test data not found: ${test_root}" >&2
  echo "Place it at /data/cosec/test or pass --test-root /your/cosec/test." >&2
  exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  echo "Place the matching .pth file under checkpoints/ or use scripts/infer.sh with --weights." >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/infer.sh" \
  --model mask2former \
  --variant "${variant}" \
  --test-root "${test_root}" \
  --out-dir "${out_dir}" \
  --execute \
  "$@"
