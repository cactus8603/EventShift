#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export EVENTSHIFT_ROOT="${EVENTSHIFT_ROOT:-${ROOT_DIR}}"

COSEC_ROOT=""
TEST_ROOT=""
BRENET_ROOT=""
COSEC_MANIFEST=""
DSEC_ROOT=""
ACDC_ROOT=""
SPLIT_DIR="${EVENTSHIFT_ROOT}/data/splits/cosec"
ACDC_SPLIT_DIR="${EVENTSHIFT_ROOT}/work_dirs/mmseg/acdc_splits"
KFOLDS="3"
PREFIX="domaincover20"
VAL_FRACTION="0.20"
RUN_KFOLD=1
RUN_DOMAIN_COVER=1
RUN_ACDC_SPLITS=0
SHOW_HELP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cosec-root)
      COSEC_ROOT="${2:-}"; shift 2 ;;
    --cosec-root=*)
      COSEC_ROOT="${1#--cosec-root=}"; shift ;;
    --test-root)
      TEST_ROOT="${2:-}"; shift 2 ;;
    --test-root=*)
      TEST_ROOT="${1#--test-root=}"; shift ;;
    --brenet-root)
      BRENET_ROOT="${2:-}"; shift 2 ;;
    --brenet-root=*)
      BRENET_ROOT="${1#--brenet-root=}"; shift ;;
    --cosec-manifest)
      COSEC_MANIFEST="${2:-}"; shift 2 ;;
    --cosec-manifest=*)
      COSEC_MANIFEST="${1#--cosec-manifest=}"; shift ;;
    --dsec-root)
      DSEC_ROOT="${2:-}"; shift 2 ;;
    --dsec-root=*)
      DSEC_ROOT="${1#--dsec-root=}"; shift ;;
    --acdc-root)
      ACDC_ROOT="${2:-}"; shift 2 ;;
    --acdc-root=*)
      ACDC_ROOT="${1#--acdc-root=}"; shift ;;
    --split-dir)
      SPLIT_DIR="${2:-}"; shift 2 ;;
    --split-dir=*)
      SPLIT_DIR="${1#--split-dir=}"; shift ;;
    --acdc-split-dir)
      ACDC_SPLIT_DIR="${2:-}"; shift 2 ;;
    --acdc-split-dir=*)
      ACDC_SPLIT_DIR="${1#--acdc-split-dir=}"; shift ;;
    --kfolds)
      KFOLDS="${2:-}"; shift 2 ;;
    --kfolds=*)
      KFOLDS="${1#--kfolds=}"; shift ;;
    --prefix)
      PREFIX="${2:-}"; shift 2 ;;
    --prefix=*)
      PREFIX="${1#--prefix=}"; shift ;;
    --val-fraction)
      VAL_FRACTION="${2:-}"; shift 2 ;;
    --val-fraction=*)
      VAL_FRACTION="${1#--val-fraction=}"; shift ;;
    --skip-kfold)
      RUN_KFOLD=0; shift ;;
    --skip-domain-cover)
      RUN_DOMAIN_COVER=0; shift ;;
    --build-acdc-splits)
      RUN_ACDC_SPLITS=1; shift ;;
    -h|--help)
      SHOW_HELP=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      SHOW_HELP=1
      break ;;
  esac
done

mkdir -p \
  "${EVENTSHIFT_ROOT}/data" \
  "${EVENTSHIFT_ROOT}/data/splits" \
  "${EVENTSHIFT_ROOT}/checkpoints" \
  "${EVENTSHIFT_ROOT}/outputs" \
  "${EVENTSHIFT_ROOT}/work_dirs/cache" \
  "${EVENTSHIFT_ROOT}/work_dirs/diagnostics" \
  "${EVENTSHIFT_ROOT}/work_dirs/manifests" \
  "${EVENTSHIFT_ROOT}/work_dirs/mmseg"

if [[ -n "${TEST_ROOT}" ]]; then export TEST_ROOT; fi
if [[ -n "${COSEC_ROOT}" ]]; then export COSEC_ROOT; fi
if [[ -n "${BRENET_ROOT}" ]]; then export BRENET_ROOT; fi
if [[ -n "${COSEC_MANIFEST}" ]]; then export EVENTSHIFT_COSEC_MANIFEST="${COSEC_MANIFEST}"; fi
if [[ -n "${DSEC_ROOT}" ]]; then export DSEC_ROOT; fi
if [[ -n "${ACDC_ROOT}" ]]; then export ACDC_ROOT; fi
export EVENTSHIFT_COSEC_SPLIT_DIR="${SPLIT_DIR}"

if [[ "${SHOW_HELP}" -eq 1 || ( -z "${COSEC_ROOT}" && "${RUN_ACDC_SPLITS}" -eq 0 ) ]]; then
  cat <<MSG
EventShift workspace directories are ready.

Quick CoSEC split preparation:

  bash scripts/prepare_data.sh \\
    --cosec-root /path/to/cosec/train \\
    --split-dir data/splits/cosec

This writes sequence-level k-fold split files and a frame-list domain-cover split.
For 0.4111 rebuild or test inference, no preprocessing is required beyond passing:

  bash scripts/rebuild_04111.sh --test-root /path/to/cosec/test

Optional args:

  --kfolds N              Number of CoSEC sequence folds. Default: ${KFOLDS}
  --prefix NAME           Frame-list prefix split name. Default: ${PREFIX}
  --val-fraction FLOAT    Validation fraction for prefix split. Default: ${VAL_FRACTION}
  --skip-kfold            Do not build sequence-level k-fold splits.
  --skip-domain-cover     Do not build frame-list domain-cover split.
  --build-acdc-splits     Also write MMSeg ACDC split files when --acdc-root is set.

Dataset rationale and domain-gap notes:

  data/README.md
  docs/dataset_preparation.md
MSG
  if [[ "${SHOW_HELP}" -eq 1 ]]; then
    exit 0
  fi
  exit 0
fi

if [[ -n "${COSEC_ROOT}" ]]; then
  mkdir -p "${SPLIT_DIR}"

  if [[ "${RUN_KFOLD}" -eq 1 ]]; then
    python tools/data/build_cosec_kfold_splits.py \
      --root "${COSEC_ROOT}" \
      --folds "${KFOLDS}" \
      --write-splits \
      --split-dir "${SPLIT_DIR}"
  fi

  if [[ "${RUN_DOMAIN_COVER}" -eq 1 ]]; then
    python tools/data/build_cosec_domain_cover_frame_split.py \
      --root "${COSEC_ROOT}" \
      --prefix "${PREFIX}" \
      --val-fraction "${VAL_FRACTION}" \
      --write-splits \
      --split-dir "${SPLIT_DIR}"
  fi
fi

if [[ "${RUN_ACDC_SPLITS}" -eq 1 ]]; then
  if [[ -z "${ACDC_ROOT}" ]]; then
    echo "--build-acdc-splits requires --acdc-root" >&2
    exit 2
  fi
  python tools/data/build_mmseg_acdc_splits.py \
    --acdc-root "${ACDC_ROOT}" \
    --out-dir "${ACDC_SPLIT_DIR}"
fi

cat <<MSG

Dataset preparation completed.

CoSEC split dir:
  ${SPLIT_DIR}
ACDC split dir:
  ${ACDC_SPLIT_DIR}

Next references:
  data/README.md
  docs/dataset_preparation.md
MSG
