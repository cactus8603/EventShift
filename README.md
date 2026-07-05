# EventShift

EventShift is a cleaned, submission-friendly restructuring of the 0411 CoSEC /
DSEC event-assisted semantic segmentation codebase.

This repository is not just a wrapper around the old bundle. The original 0411
scripts have been reorganized by purpose:

```text
configs/                  Training and inference configs
    eventshift/           Three readable entry configs: eventshift, RGB baseline, event-only
    mask2former/          0411 final/rebuild Mask2Former configs
    mask2former_experiments/
                           Historical Swin-L event/RGB experiment configs
    segformer/            0411 SegFormer configs
    mmseg/                MMSeg configs
    maskdino/             MaskDINO configs

eventshift/               Lightweight reusable Python package
scripts/                  User-facing shell entry points
tools/
    training/             Training backends and calibration trainers
    export/               Prediction export / TTA scripts
    rebuild/              0411 rebuild runners
    postprocess/          Submission composition, repair, routing, voting
    diagnostics/          Event and routing analysis tools
    data/                 Dataset split / conversion / cache builders
    analysis/             Summaries and run analysis
    launchers/            Historical experiment launch scripts
    cache/                Feature and prediction cache utilities
    misc/                 Less frequently used utilities
    *.py                  Shared compatibility modules used by older scripts
third_party/              Source-only copies of Mask2Former, detectron2, mmsegmentation
metadata/                 Original 0411 notes/manifests/provenance
checkpoints/              Local checkpoint storage, ignored by git
data/                     Local data placeholder, ignored by git
outputs/                  Local outputs, ignored by git
docs/                     Method/rebuild notes
```

## What Is Included

Included in git:

```text
source code
configs
scripts
docs
third_party source files
metadata/manifests
```

Not included in git:

```text
raw datasets
generated outputs
submission zips
compiled artifacts: *.so, *.o, build/, dist/, __pycache__/
```

`checkpoints/` is also ignored by git. For local reproduction, this workspace may
contain copied checkpoint files there, but they are not part of the source
submission.

## Environment

The rebuild work used one conda environment named `ebmv_seg` for both
Mask2Former/Swin-L and SegFormer/MMSeg inference.

```bash
cd /code/ebmv/EventShift
conda env create -f environment.yml
conda activate ebmv_seg
pip install --no-build-isolation -e third_party/detectron2
pip install -r third_party/Mask2Former/requirements.txt
```

The verified local stack was Python 3.10.20, PyTorch 2.6.0+cu124,
Torchvision 0.21.0+cu124, Detectron2 0.6, MMSegmentation 1.2.2,
MMEngine 0.10.7, and mmcv-lite 2.1.0. See
`docs/ebmv_seg_environment.md` for exact setup, validation commands, and 0411
rebuild notes.

If the target machine requires Mask2Former native ops, rebuild them locally:

```bash
cd third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd /code/ebmv/EventShift
```

## Data Paths

Raw datasets are expected outside the submitted repo. Point the code to them with
environment variables:

```bash
export BRENET_ROOT=/path/to/BRENet
export COSEC_ROOT=/path/to/cosec
export DSEC_ROOT=/path/to/dsec
export ACDC_ROOT=/path/to/acdc
export TEST_ROOT=/path/to/test
```

The CoSEC event manifest expected by the event dataset utilities is:

```text
$BRENET_ROOT/projects/brenet_cosec/manifests/cosec_train_bidir_50ms.json
```

## Common Commands

Dry-run an EventShift training command:

```bash
bash scripts/train.sh configs/eventshift/cosec_eventshift.yaml
```

Run training after the environment and data paths are ready:

```bash
bash scripts/train.sh configs/eventshift/cosec_eventshift.yaml --execute
```

Run inference dry-run:

```bash
bash scripts/infer.sh configs/eventshift/cosec_eventshift.yaml --weights checkpoints/model.pth --test-root /path/to/test
```

Rebuild the historical 0411 pipeline, if the required local checkpoints and
fixed artifacts are available:

```bash
TEST_ROOT=/path/to/test bash scripts/rebuild_04111.sh
```

See `docs/rebuild_04111_from_checkpoints.md` for the full 0411 rebuild notes.

## Checkpoints

A local copy of checkpoints can be placed under `checkpoints/` for reproduction.
This directory is ignored by git via `.gitignore` so weights do not enter the
source submission.

## Notes

- Event usage details are in `docs/event_usage_swinl_segformer.md`.
- Original file manifests and SHA notes are in `metadata/`.
- Historical launch scripts remain under `tools/launchers/`; they preserve the
  experiment history and may need environment-specific path adjustments.
