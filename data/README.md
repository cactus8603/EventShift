# Data Preparation

This directory is a local workspace placeholder. Raw datasets are not committed to the repository; keep them on local storage and pass paths through args or environment variables. The public quick setup command is:

```bash
bash scripts/prepare_data.sh \
  --cosec-root /path/to/cosec/train \
  --split-dir data/splits/cosec
```

The script creates the local workspace directories and, when `--cosec-root` is provided, writes the active CoSEC split files used by EventShift:

```text
data/splits/cosec/
|-- train_kfold3_fold0.txt
|-- val_kfold3_fold0.txt
|-- train_kfold3_fold1.txt
|-- val_kfold3_fold1.txt
|-- train_kfold3_fold2.txt
|-- val_kfold3_fold2.txt
|-- train_domaincover20.txt
|-- val_domaincover20.txt
`-- domaincover20_summary.json
```

For the 0.4111 submission rebuild or test-set inference, no split generation is needed. Use `--test-root` directly with the inference or rebuild script.

## Dataset Roles

| Dataset | How it is collected | How EventShift uses it |
| --- | --- | --- |
| [CoSEC train/test](https://arxiv.org/abs/2408.08500) | Official challenge data with RGB driving frames and semantic labels for the training split. Sequences are organized by illumination domain such as `Day_*` and `Night_*`; the test split is unlabeled for submission. | Main target domain for training, validation, inference, and submission export. |
| [CoSEC events](https://arxiv.org/abs/2408.08500) / [BRENet-style assets](https://github.com/zyaocoder/BRENet) | Event streams aligned to CoSEC frames and stored through local BRENet-style event asset layouts and manifest files. | Event-based CoSEC training and RGB-event fusion variants. Not required for RGB-only inference or final-submission reproduction. |
| [DSEC](https://dsec.ifi.uzh.ch/) / [DSEC-Semantic](https://dsec.ifi.uzh.ch/dsec-semantic/) | Public driving data with synchronized RGB/event sensors and 19-class semantic labels. | Auxiliary RGB/event segmentation data for wider driving-domain coverage. |
| [ACDC](https://acdc.vision.ee.ethz.ch/) | Public adverse-condition driving data with fog, night, rain, and snow semantic labels. | Auxiliary adverse-condition data, especially night. We filter small CoSEC-like subsets when the full ACDC domain is too broad. |
| [REAL-style challenge/test pool](https://arxiv.org/abs/2408.08500) | Unlabeled challenge/test-style real-domain imagery from the CoSEC challenge package. Historical `REAL_dataset/*/gt` folders contain RGB images, not semantic labels. | Diagnostics, inference, and pseudo-label experiments only. It is not supervised ground truth. |

## What The Quick Script Builds

### Sequence-Level K-Fold Splits

The k-fold split assigns whole CoSEC sequences to train or validation. This avoids leakage from adjacent frames in the same sequence and gives a cleaner estimate of sequence-level generalization. The builder balances day/night domains and frame counts across folds.

The generated names are registered by the loaders as datasets such as:

```text
cosec_kfold3_fold0_train
cosec_kfold3_fold0_val
cosec_kfold3_fold0_day_val
cosec_kfold3_fold0_night_val
cosec_kfold3_fold0_train_event
cosec_kfold3_fold0_val_event
```

Use this split style when reporting validation behavior or comparing model variants.

### Frame-List Prefix Splits

The frame-list prefix split writes `train_<prefix>.txt` and `val_<prefix>.txt`, where each line is a `sequence/frame` id. The default quick setup writes `domaincover20`:

```text
train_domaincover20.txt
val_domaincover20.txt
domaincover20_summary.json
```

This split is intentionally frame-level. It can select validation frames that cover rare classes and both illumination domains, even when that means different frames from the same sequence appear on different sides. Use it for controlled ablations, class coverage checks, and domain/class coverage diagnostics. Use sequence-level k-fold for leakage-free validation.

## Why We Do Domain-Aware Preparation

EventShift targets illumination-robust segmentation, so the key failure mode is not only class imbalance but also day/night domain mismatch. A random frame split can hide this problem because nearby frames share lighting, scene layout, motion, and object composition. The current preparation separates two needs:

- Sequence-level k-fold checks whether the model transfers to held-out sequences without adjacent-frame leakage.
- Frame-list prefix splits make sure validation contains useful class and domain coverage when studying rare classes or event/RGB fusion behavior.
- ACDC domain-gap filtering selects auxiliary adverse-condition samples that look closer to CoSEC night instead of mixing the whole ACDC distribution blindly.

The ACDC filter compares semantic class distributions and coarse RGB/luminance/saturation statistics against the CoSEC reference domain. Greedy selection is preferred for the cleaned recipes because it keeps the selected subset globally closer to the target domain, rather than only choosing individually closest images.

## Common Commands

Build the default CoSEC split package:

```bash
bash scripts/prepare_data.sh \
  --cosec-root /path/to/cosec/train \
  --split-dir data/splits/cosec
```

Change the fold count or frame-list prefix:

```bash
bash scripts/prepare_data.sh \
  --cosec-root /path/to/cosec/train \
  --split-dir data/splits/cosec \
  --kfolds 3 \
  --prefix domaincover20 \
  --val-fraction 0.20
```

Build only the sequence-level k-fold splits:

```bash
bash scripts/prepare_data.sh \
  --cosec-root /path/to/cosec/train \
  --skip-domain-cover
```

Build only the frame-list domain-cover split:

```bash
bash scripts/prepare_data.sh \
  --cosec-root /path/to/cosec/train \
  --skip-kfold
```

Build MMSeg split files for ACDC when using ACDC configs:

```bash
bash scripts/prepare_data.sh \
  --acdc-root /path/to/acdc \
  --build-acdc-splits
```

For full command references, expected raw layouts, DSEC preparation, and ACDC domain-gap filtering commands, see [`../docs/dataset_preparation.md`](../docs/dataset_preparation.md).
