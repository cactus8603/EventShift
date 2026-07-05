#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
REGISTRY="${REGISTRY:-${ROOT}/unified_cosec_acdc/classcover_v1/checkpoints/full_desc_cosec_acdc}"
MANIFEST="${REGISTRY}/MANIFEST.tsv"

mkdir -p "${REGISTRY}"
tmp_manifest="${MANIFEST}.tmp"
printf 'model\tstage\tdomain\tstatus\tlink\tsource\n' > "${tmp_manifest}"

safe_link() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [[ -e "${dst}" && ! -L "${dst}" ]]; then
    return 2
  fi
  ln -sfn "${src}" "${dst}"
}

find_latest() {
  local dirname="$1"
  local pattern="$2"
  find "${dirname}" -maxdepth 1 -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

record() {
  local model="$1"
  local stage="$2"
  local domain="$3"
  local src="$4"
  local link_name="$5"
  local dst="${REGISTRY}/${model}/${stage}/${link_name}"
  local status

  if [[ -n "${src}" && -e "${src}" ]]; then
    if safe_link "${src}" "${dst}"; then
      status="symlink"
    else
      status="blocked_existing_non_symlink"
    fi
  else
    status="missing"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${model}" "${stage}" "${domain}" "${status}" "${dst}" "${src}" >> "${tmp_manifest}"
}

record_file() {
  local model="$1"
  local stage="$2"
  local domain="$3"
  local run_dir="$4"
  local filename="$5"
  record "${model}" "${stage}" "${domain}" "${run_dir}/${filename}" "${filename}"
}

record_latest() {
  local model="$1"
  local stage="$2"
  local domain="$3"
  local run_dir="$4"
  local pattern="$5"
  local link_name="$6"
  local src
  src="$(find_latest "${run_dir}" "${pattern}" || true)"
  record "${model}" "${stage}" "${domain}" "${src}" "${link_name}"
}

record_metadata() {
  local model="$1"
  local stage="$2"
  local run_dir="$3"
  for filename in config.yaml metrics.json log.txt last_checkpoint best_daynight_miou.json; do
    if [[ -e "${run_dir}/${filename}" ]]; then
      record "${model}" "${stage}" "metadata" "${run_dir}/${filename}" "${filename}"
    fi
  done
}

record_selected() {
  local model="$1"
  local domain="$2"
  local src="$3"
  local link_name="$4"
  record "${model}" selected "${domain}" "${src}" "${link_name}"
}

mask2former_step1="${ROOT}/swin_l/work_dirs/swinL_full_dsec_cosec_acdc_unified_bs1"
mask2former_step2="${ROOT}/swin_l/work_dirs/swinL_full_dsec_cosec_acdc_unified_2step_lr1e-6_bs1"
maskdino_step1="${ROOT}/maskdino_swinl/work_dirs/maskdino_swinl_full_dsec_cosec_acdc_unified_classcover_v1_bs1"
maskdino_step2="${ROOT}/maskdino_swinl/work_dirs/maskdino_swinl_full_dsec_cosec_acdc_unified_classcover_v1_2step_lr1e-6_bs1"
segformer_step1="${ROOT}/swin_l/work_dirs/mmseg/segformer_b5_full_dsec_cosec_acdc_unified"
segformer_step2="${ROOT}/swin_l/work_dirs/mmseg/segformer_b5_full_dsec_cosec_acdc_unified_2step_lr1e-6"
brenet_step1="${ROOT}/BRENet/work_dirs/brenet_b2_full_dsec_cosec_acdc_unified_bs4"
brenet_step2="${ROOT}/BRENet/work_dirs/brenet_b2_full_dsec_cosec_acdc_unified_acdc_2step_bs4"

for stage in step1 step2; do
  run_var="mask2former_${stage}"
  run_dir="${!run_var}"
  record_file mask2former "${stage}" day "${run_dir}" best_model_cosec_day.pth
  record_file mask2former "${stage}" night "${run_dir}" best_model_cosec_night.pth
  record_file mask2former "${stage}" acdc "${run_dir}" best_model_acdc_all.pth
  record_file mask2former "${stage}" acdc_night "${run_dir}" best_model_acdc_night.pth
  record_metadata mask2former "${stage}" "${run_dir}"
done

for stage in step1 step2; do
  run_var="maskdino_${stage}"
  run_dir="${!run_var}"
  record_file maskdino "${stage}" day "${run_dir}" best_model_cosec_day.pth
  record_file maskdino "${stage}" night "${run_dir}" best_model_cosec_night.pth
  record_file maskdino "${stage}" acdc "${run_dir}" best_model_acdc_all.pth
  record_file maskdino "${stage}" acdc_night "${run_dir}" best_model_acdc_night.pth
  record_metadata maskdino "${stage}" "${run_dir}"
done

for stage in step1 step2; do
  run_var="segformer_${stage}"
  run_dir="${!run_var}"
  record_latest segformer "${stage}" day "${run_dir}" 'best_day_mIoU*.pth' best_day_mIoU.pth
  record_latest segformer "${stage}" night "${run_dir}" 'best_night_mIoU*.pth' best_night_mIoU.pth
  record_latest segformer "${stage}" acdc "${run_dir}" 'best_acdc_mIoU*.pth' best_acdc_mIoU.pth
  record_metadata segformer "${stage}" "${run_dir}"
done

record segformer step1 metadata "${segformer_step1}/SegFormer_B5_FullDSEC_CoSEC_ACDC_Unified.py" config.py
record segformer step1 metadata "${segformer_step1}/20260628_133910/vis_data/scalars.json" scalars.json
record segformer step1 metadata "${segformer_step1}/20260628_133910/20260628_133910.log" log.txt
record segformer step1 metadata "${segformer_step1}/20260628_133910/vis_data/20260628_133910.json" run_meta.json

record segformer step2 metadata "${segformer_step2}/SegFormer_B5_FullDSEC_CoSEC_ACDC_Unified_2Step_LR1e-6.py" config.py
record segformer step2 metadata "${segformer_step2}/20260628_162037/vis_data/scalars.json" scalars_162037.json
record segformer step2 metadata "${segformer_step2}/20260628_172853/vis_data/scalars.json" scalars_172853.json
record segformer step2 metadata "${segformer_step2}/20260628_162037/20260628_162037.log" log_162037.txt
record segformer step2 metadata "${segformer_step2}/20260628_172853/20260628_172853.log" log_172853.txt
record segformer step2 metadata "${segformer_step2}/20260628_162037/vis_data/20260628_162037.json" run_meta_162037.json
record segformer step2 metadata "${segformer_step2}/20260628_172853/vis_data/20260628_172853.json" run_meta_172853.json

record_file brenet step1 day "${brenet_step1}" best_day_mIoU.pth
record_file brenet step1 night "${brenet_step1}" best_night_mIoU.pth
record_file brenet step1 acdc "${brenet_step1}" best_acdc_mIoU.pth
record_metadata brenet step1 "${brenet_step1}"

record_file brenet step2 acdc "${brenet_step2}" best_acdc_mIoU.pth
record_metadata brenet step2 "${brenet_step2}"

record_selected mask2former day "${REGISTRY}/mask2former/step2/best_model_cosec_day.pth" best_model_cosec_day.pth
record_selected mask2former night "${REGISTRY}/mask2former/step1/best_model_cosec_night.pth" best_model_cosec_night.pth
record_selected mask2former acdc "${REGISTRY}/mask2former/step2/best_model_acdc_all.pth" best_model_acdc_all.pth
record_selected mask2former acdc_night "${REGISTRY}/mask2former/step1/best_model_acdc_night.pth" best_model_acdc_night.pth

record_selected segformer day "${REGISTRY}/segformer/step1/best_day_mIoU.pth" best_day_mIoU.pth
record_selected segformer night "${REGISTRY}/segformer/step2/best_night_mIoU.pth" best_night_mIoU.pth
record_selected segformer acdc "${REGISTRY}/segformer/step2/best_acdc_mIoU.pth" best_acdc_mIoU.pth

record_selected maskdino day "${REGISTRY}/maskdino/step1/best_model_cosec_day.pth" best_model_cosec_day.pth
record_selected maskdino night "${REGISTRY}/maskdino/step1/best_model_cosec_night.pth" best_model_cosec_night.pth
record_selected maskdino acdc "${REGISTRY}/maskdino/step1/best_model_acdc_all.pth" best_model_acdc_all.pth
record_selected maskdino acdc_night "${REGISTRY}/maskdino/step1/best_model_acdc_night.pth" best_model_acdc_night.pth

record_selected brenet day "${REGISTRY}/brenet/step1/best_day_mIoU.pth" best_day_mIoU.pth
record_selected brenet night "${REGISTRY}/brenet/step1/best_night_mIoU.pth" best_night_mIoU.pth
record_selected brenet acdc "${REGISTRY}/brenet/step1/best_acdc_mIoU.pth" best_acdc_mIoU.pth

mv "${tmp_manifest}" "${MANIFEST}"
echo "Synced checkpoint registry: ${REGISTRY}"
echo "Manifest: ${MANIFEST}"
