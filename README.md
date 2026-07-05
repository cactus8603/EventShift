<div align="center">

# EventShift

**EventShift: Event-Guided Illumination-Robust Semantic Segmentation for CoSEC**

CoSEC Semantic Segmentation Track @ ECCV 2026 EBMV Workshop Challenge

[Overview](#overview) | [Results](#results) | [Installation](#installation) | [Dataset Preparation](#dataset-preparation) | [Inference](#inference) | [Reproducibility](#reproducing-the-final-submission)

</div>

## Why EventShift?

The name EventShift reflects the core idea of our method: RGB images under low-light or exposure-changing conditions often suffer from illumination-induced feature shifts, while event streams provide complementary motion and boundary cues that are less dependent on absolute brightness. EventShift uses event information to adapt RGB representations toward illumination-robust semantic features for CoSEC semantic segmentation.

## Overview

EventShift is our solution for the CoSEC Semantic Segmentation Track at the ECCV 2026 EBMV Workshop Challenge. The repository contains the model configs, inference scripts, dataset split utilities, and final-submission reproduction path used for RGB-event semantic segmentation under difficult illumination.

RGB frames can degrade under low light, glare, and exposure changes. Event signals provide motion and boundary cues that are less tied to absolute brightness, so EventShift uses RGB-event fusion and domain-aware post-processing to improve robustness on the challenge test set.

## Highlights

- Event-guided semantic segmentation for CoSEC.
- Supports RGB, event, and RGB-event fusion settings.
- Args-first scripts for inference, training, dataset preparation, and final-submission reproduction.
- Composable configs with shared base files, dataset files, model files, variants, and rebuild recipes.
- One public final-submission reproduction entry point with tqdm progress bars for each model export.
- Datasets, checkpoints, generated outputs, and submission archives are kept out of Git.

## Results

| Entry | Score | Reproducibility entry point | Notes |
| --- | ---: | --- | --- |
| Final b75 pipeline | 0.4111 | `bash scripts/rebuild_04111.sh` | Two Mask2Former exports, one SegFormer export, and bundled post-processing gates. |

The final challenge ranking and technical report details will be added after the ECCV 2026 EBMV Workshop Challenge report is finalized.

## Installation

Create the verified conda environment explicitly:

```bash
conda create -n ebmv_seg python=3.10 pip setuptools wheel ninja -y
conda activate ebmv_seg
pip install -r requirements.txt
```

For a pinned environment file, use:

```bash
conda env create -f environment.yml
conda activate ebmv_seg
```

Install repository-local third-party dependencies when the Mask2Former path is needed:

```bash
pip install --no-build-isolation -e third_party/detectron2
pip install -r third_party/Mask2Former/requirements.txt
```

The verified environment used for our experiments includes Python 3.10, PyTorch 2.6.0 + CUDA 12.4, Torchvision 0.21.0 + CUDA 12.4, Detectron2 0.6, MMSegmentation 1.2.2, MMEngine 0.10.7, and MMCV-lite 2.1.0.

If `mmcv` or `mmcv-lite` installation fails, see the local [MMCV notes](docs/ebmv_seg_environment.md#mmcv-notes) and the official OpenMMLab [MMCV installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

If native Mask2Former operators are required, rebuild them locally:

```bash
cd third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd -
```

See [docs/ebmv_seg_environment.md](docs/ebmv_seg_environment.md) for exact setup notes, validation commands, and troubleshooting.

## Dataset Preparation

Raw datasets should live outside the repository. Use the provided entry point to create local workspace folders and build the recommended CoSEC split files:

```bash
bash scripts/prepare_data.sh \
  --cosec-root /path/to/cosec/train \
  --split-dir data/splits/cosec
```

The command generates sequence-level k-fold splits and a frame-list domain-cover prefix split. These split files are local artifacts and are ignored by Git.

For challenge test-set inference and final-submission reproduction, preprocessing is not required. Pass the test root directly:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/cosec/test
```

Expected CoSEC roots follow the official sequence layout:

```text
/path/to/cosec/train/
|-- Day_*/
|   |-- img_co_left/
|   `-- segment_co/
`-- Night_*/
    |-- img_co_left/
    `-- segment_co/

/path/to/cosec/test/
|-- Day_*/
|   `-- img_co_left/
|-- Night_*/
|   `-- img_co_left/
`-- REAL_*/
    `-- img_co_left/
```

See [data/README.md](data/README.md) for the dataset provenance, split strategy, k-fold setup, frame-list prefix split, and domain-gap filtering notes. See [docs/dataset_preparation.md](docs/dataset_preparation.md) for the command reference.

## Checkpoints

Checkpoints are not stored in Git. Place them under `checkpoints/` or pass absolute paths through `--weights`.

Final-submission reproduction expects the checkpoint filenames referenced by the recipe variants:

| Checkpoint | Used by |
| --- | --- |
| `checkpoints/m2f_event_full_cosec_from_day_best_floor816070_lr5e-7.pth` | Mask2Former day event export |
| `checkpoints/m2f_full_desc_selected_cosec_night.pth` | Mask2Former night full-domain export |
| `checkpoints/segformer_b5_event_full_cosec_from_night_best_floor546453_lr1e-6_iter4500.pth` | SegFormer night event export |

Download links will be added when the public release package is finalized.

## Inference

Mask2Former example:

```bash
bash scripts/infer.sh \
  --model mask2former \
  --variant rgb_baseline \
  --weights /path/to/checkpoints/mask2former_rgb.pth \
  --test-root /path/to/cosec/test \
  --out-dir outputs/infer_rgb
```

SegFormer example:

```bash
bash scripts/infer.sh \
  --model segformer \
  --variant night_event_04111 \
  --weights /path/to/checkpoints/segformer_night.pth \
  --test-root /path/to/cosec/test \
  --out-dir outputs/infer_segformer_night
```

By default the backend command is printed for inspection. Add `--execute` to run it. Extra backend options can be passed after `--`.

Expected outputs are PNG masks under the selected `--out-dir`.

## Reproducing the Final Submission

The final b75 submission, corresponding to the 0.4111 local reference score, is rebuilt through one shell entry point. Run it inside the activated `ebmv_seg` environment:

```bash
conda activate ebmv_seg
bash scripts/rebuild_04111.sh \
  --test-root /path/to/cosec/test \
  --device cuda:0
```

The default output root is timestamped under `outputs/`. The runner exports masks, composes the final submission, validates the zip, and compares the archive contents against local 0.4111 reference zips when they are present.

For a quick argument and path check without running full inference:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/cosec/test \
  --smoke-limit 0 \
  --skip-inference
```

See [docs/rebuild_04111_from_checkpoints.md](docs/rebuild_04111_from_checkpoints.md) for the detailed 0.4111 recipe and comparison behavior.

## Training

EventShift training uses the CoSEC training set and a CoSEC event manifest. To dry-run the command:

```bash
bash scripts/train.sh \
  --model mask2former \
  --variant eventshift \
  --cosec-root /path/to/cosec/train \
  --cosec-manifest /path/to/cosec_train_bidir_50ms.json
```

Add `--execute` to launch training. `--brenet-root` is only needed for older manifests whose relative paths cannot be resolved from the manifest location.

## Documentation

| Topic | File |
| --- | --- |
| Environment setup | [docs/ebmv_seg_environment.md](docs/ebmv_seg_environment.md) |
| Dataset design and splits | [data/README.md](data/README.md) |
| Dataset command reference | [docs/dataset_preparation.md](docs/dataset_preparation.md) |
| Final-submission reproduction | [docs/rebuild_04111_from_checkpoints.md](docs/rebuild_04111_from_checkpoints.md) |
| Third-party source notes | [third_party/README.md](third_party/README.md) |

## Repository Layout

| Path | Contents |
| --- | --- |
| `configs/eventshift/` | Composable base, dataset, model, variant, and rebuild recipe configs. |
| `configs/mask2former/`, `configs/segformer/`, `configs/mmseg/` | Backend configs used by Mask2Former and SegFormer/MMSeg runs. |
| `eventshift/` | Lightweight package code for config composition and backend selection. |
| `scripts/` | User-facing entry points for training, inference, evaluation, data setup, and final-submission reproduction. |
| `tools/` | Training adapters, exporters, rebuild logic, dataset preparation, diagnostics, and post-processing utilities. |
| `third_party/` | Vendored source/subsets required by the legacy Mask2Former, Detectron2, and MMSegmentation paths. |
| `docs/` | Reproducibility notes, environment setup, dataset setup, and rebuild details. |
| `data/`, `checkpoints/`, `outputs/`, `submit/`, `work_dirs/` | Local-only placeholders for datasets, checkpoints, predictions, submissions, and generated caches. |

## License

EventShift-specific code is released under the MIT License; see [LICENSE](LICENSE). Third-party source, datasets, checkpoints, and pretrained weights follow their own upstream or challenge terms and are not redistributed in this repository.

## Acknowledgement

This repository builds on [Mask2Former](https://github.com/facebookresearch/Mask2Former), [Detectron2](https://github.com/facebookresearch/detectron2), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), SegFormer, DSEC/DSEC-Semantic, ACDC, and the CoSEC challenge resources. We thank the authors and maintainers for making their code and datasets available to the research community.
