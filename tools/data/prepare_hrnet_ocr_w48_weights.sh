#!/usr/bin/env bash
set -euo pipefail

ROOT="/work/u1621738/ebmv_eccv/eccv_segment/swin_l"
DEST_DIR="${ROOT}/work_dirs/pretrained"
URL="https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth"
DEST="${DEST_DIR}/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth"
TMP="${DEST}.tmp"

mkdir -p "${DEST_DIR}"

if [[ -s "${DEST}" ]]; then
    echo "HRNet-OCR-W48 checkpoint already exists: ${DEST}"
    ls -lh "${DEST}"
    exit 0
fi

rm -f "${TMP}"

if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 5 -o "${TMP}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -O "${TMP}" "${URL}"
else
    echo "Neither curl nor wget is available." >&2
    exit 1
fi

mv "${TMP}" "${DEST}"
printf "%s\n" "${URL}" > "${DEST}.url"
ls -lh "${DEST}"
