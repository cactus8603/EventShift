<div align="center">

# EventShift

### Semantic Segmentation Track @ ECCV 2026

**Event-Guided Illumination-Robust Semantic Segmentation for CoSEC**

[Overview](#overview) | [Installation](#installation) | [Rebuild 0.4111](#rebuild-04111) | [Dataset Preparation](#dataset-preparation)

</div>

## Overview

EventShift is our solution for the **CoSEC Semantic Segmentation Track** at **ECCV 2026**. The method targets semantic segmentation under challenging illumination conditions by using event streams as complementary cues to RGB images.

RGB images may suffer from severe illumination shift under low-light or exposure-changing scenes. Event signals provide additional motion and boundary information that is less dependent on absolute image brightness. EventShift uses these event cues to improve semantic segmentation robustness under difficult lighting conditions.

## News

- **2026-07-05:** Cleaned the public release with args-first scripts, composable configs, a recipe-driven 0.4111 rebuild path, dataset preparation notes, and third-party provenance.

## Highlights

- Event-guided semantic segmentation for CoSEC.
- Args-first inference, training, and 0.4111 rebuild entry points.
- Composable configs with base, dataset, model, variant, and recipe files.
- Supports RGB, event, and RGB-event fusion settings.
- Documents dataset preparation, domain-gap filtering, third-party source, and license notes.
- Checkpoints, datasets, generated outputs, and submission archives are not stored in Git.

## Installation

Create the verified conda environment explicitly:

```bash
conda create -n ebmv_seg python=3.10 pip setuptools wheel ninja -y
conda activate ebmv_seg
pip install -r requirements.txt
```

For a fully pinned environment, the equivalent conda file is also provided:

```bash
conda env create -f environment.yml
conda activate ebmv_seg
```

Install repository-local third-party dependencies if needed:

```bash
pip install --no-build-isolation -e third_party/detectron2
pip install -r third_party/Mask2Former/requirements.txt
```

The verified environment used for our experiments includes:

```text
Python 3.10
PyTorch 2.6.0 + CUDA 12.4
Torchvision 0.21.0 + CUDA 12.4
Detectron2 0.6
MMSegmentation 1.2.2
MMEngine 0.10.7
MMCV-lite 2.1.0
```

If `mmcv`/`mmcv-lite` installation fails, see the local [MMCV notes](docs/ebmv_seg_environment.md#mmcv-notes) and the official OpenMMLab [MMCV installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

See `docs/ebmv_seg_environment.md` for exact setup notes, validation commands, and troubleshooting.

If native Mask2Former operators are required, rebuild them locally:

```bash
cd third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd -
```

## Third-Party Code

Runtime-critical third-party source is kept under `third_party/` for reproducibility. See `third_party/README.md` for provenance and install notes.

## Dataset Preparation

Use the provided shell entry point to create the local workspace directories and build the recommended CoSEC split files:

```bash
bash scripts/prepare_data.sh \
  --cosec-root /path/to/cosec/train \
  --split-dir data/splits/cosec
```

This generates the sequence-level k-fold splits and the frame-list domain-cover prefix split used by the current configs. For the 0.4111 rebuild and normal test-set inference, dataset preprocessing is not required; pass the test root directly:

```bash
bash scripts/rebuild_04111.sh --test-root /path/to/cosec/test
```

Raw datasets, generated split files, caches, and manifests are kept outside committed source. See [`data/README.md`](data/README.md) for what each prepared file means, why the domain-aware splits are used, and how CoSEC, BRENet event assets, DSEC, ACDC, and REAL-style pools are handled. The command-level reference is in [`docs/dataset_preparation.md`](docs/dataset_preparation.md).

## Inference

Mask2Former example:

```bash
bash scripts/infer.sh \
  --model mask2former \
  --variant rgb_baseline \
  --weights /path/to/checkpoints/mask2former_rgb.pth \
  --test-root /path/to/test \
  --out-dir outputs/infer_rgb
```

SegFormer example:

```bash
bash scripts/infer.sh \
  --model segformer \
  --variant night_event_04111 \
  --weights /path/to/checkpoints/segformer_night.pth \
  --test-root /path/to/test \
  --out-dir outputs/infer_segformer_night
```

By default the command is printed. Add `--execute` to run it. Extra backend options can be passed after `--`.

## Rebuild 0.4111

The 0.4111 b75 submission is rebuilt through one public shell entry point. The command assumes it is run inside the activated `ebmv_seg` environment:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/test \
  --device cuda:0
```

The recipe regenerates two Mask2Former exports and one SegFormer export, then applies the bundled post-processing gates. Each model export uses a tqdm progress bar labeled with the model name.

For a quick argument and path check without running inference:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/test \
  --smoke-limit 0 \
  --skip-inference
```

The generated final zip is compared by content against `submit/sub_pipeline_b75.zip` when that local submitted reference is present. Zip container SHA values may differ because zip metadata is not stable, so the runner also compares the filenames and PNG bytes inside the archive.

## Training

EventShift training uses the CoSEC training set and the BRENet event manifest. To dry-run the training command:

```bash
bash scripts/train.sh \
  --model mask2former \
  --variant eventshift \
  --cosec-root /path/to/cosec/train \
  --brenet-root /path/to/BRENet \
  --cosec-manifest /path/to/cosec_train_bidir_50ms.json
```

To launch training, add `--execute`.

## Checkpoints

Model checkpoints are not included in this repository. Please place downloaded checkpoints under:

```text
checkpoints/
```

Example:

```text
checkpoints/
`-- eventshift_cosec.pth
```

## Repository Layout

| Path | What belongs there |
| --- | --- |
| `configs/eventshift/` | Composable base, dataset, model, variant, and rebuild recipe configs. |
| `configs/mask2former/`, `configs/segformer/`, `configs/mmseg/` | Backend configs used by the final Mask2Former and SegFormer/MMSeg runs. |
| `eventshift/` | Lightweight package code for config composition and backend selection. |
| `scripts/` | Thin user-facing entry points for training, inference, evaluation, data setup, and 0.4111 rebuild. |
| `tools/` | Training adapters, exporters, rebuild logic, dataset preparation, diagnostics, and post-processing utilities. |
| `third_party/` | Vendored source/subsets required by the legacy Mask2Former, Detectron2, and MMSegmentation execution paths. |
| `docs/` | Reproducibility notes, dataset preparation, environment setup, and historical experiment notes. |
| `data/`, `checkpoints/`, `outputs/`, `submit/`, `work_dirs/` | Local-only placeholders for datasets, checkpoints, predictions, submissions, and generated manifests/caches. |

Historical one-off shell scripts are preserved under `scripts/archive/` and `tools/rebuild/archive/`. The recommended public entry points are `scripts/infer.sh`, `scripts/train.sh`, `scripts/eval.sh`, `scripts/prepare_data.sh`, and `scripts/rebuild_04111.sh`.

## Results

The final challenge results and technical report details will be updated after the ECCV 2026 EBMV Workshop Challenge report is finalized.

## Notes

This repository contains cleaned and reorganized code for reproducibility. Some historical experiment scripts are preserved under `tools/` and `docs/` for reference, but the recommended entry points are the scripts under `scripts/`.

For the 0.4111 rebuild workflow, see `docs/rebuild_04111_from_checkpoints.md`.

## License

EventShift-specific code is released under the MIT License; see `LICENSE`. Third-party source, datasets, checkpoints, and pretrained weights follow their own upstream or challenge terms and are not redistributed in this repository.

## Acknowledgement

This repository builds on [Mask2Former](https://github.com/facebookresearch/Mask2Former), [Detectron2](https://github.com/facebookresearch/detectron2), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), SegFormer, DSEC/DSEC-Semantic, ACDC, and the CoSEC challenge resources. We thank the authors and maintainers for making their code and datasets available to the research community.
