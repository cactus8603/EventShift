# Traincode 04111 Organization Notes

Extracted from:

```text
repro_bundle_20260629_portable_v2_traincode_04111.tar
```

This archive was extracted into `traincode_04111/` with the original top-level
`repro_bundle_20260629_portable_v2/` component stripped, so the bundle content is
directly under this directory.

## Main Layout

```text
artifacts/                         Fixed submission/report artifacts
checkpoints/                       Selected model checkpoints
code/                              Submission rebuild/export helpers
configs/                           Editable training/inference config copies
third_party/                       Mask2Former, detectron2, mmsegmentation
tools/                             Training dataset registration and launch tools
training/                          Training-side snapshot and wrapper scripts
training/README_TRAINING.md        Upstream training-pack notes
training/scripts/                  Bundle-local training launchers
```

## Training Entry Points

Mask2Former:

```bash
cd /path/to/traincode_04111
PYTHON_BIN=python \
bash training/scripts/train_mask2former_from_bundle.sh \
  configs/Mask2Former_SwinL_FullCoSEC_FromDayBest_Floor816070_LR5e-7.yaml
```

SegFormer / MMSeg:

```bash
cd /path/to/traincode_04111
PYTHON_BIN=python \
bash training/scripts/train_segformer_from_bundle.sh \
  configs/mmseg/SegFormer_B5_FullCoSEC_FromNightBest_Floor546453.py
```

MaskDINO:

```bash
cd /path/to/traincode_04111
PYTHON_BIN=python \
bash training/scripts/train_maskdino_from_bundle.sh
```

## Important Note

The training code/configs are now present, including dataset registration helpers
such as `tools/train_mask2former_cosec.py`, `tools/cosec_event_dataset.py`,
`tools/acdc_dataset.py`, and `tools/dsec19_filtered_dataset.py`.

The actual training datasets are not included in this tar. The upstream README
expects CoSEC/DSEC/ACDC images, labels, event files, and split roots to be placed
at the original paths or the config constants to be edited for local paths.
