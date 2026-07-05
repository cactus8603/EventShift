# Dataset Preparation

This repository does not store raw datasets, checkpoints, generated masks, or submission archives. Keep large assets outside Git and pass their locations with args or environment variables.

The current public workflow uses an args-first interface. Lower-level dataset loaders still support environment variables, which is useful for training launchers and diagnostics.

## Quick Setup

Create the local workspace directories and generate the recommended CoSEC split files with the shell entry point:

```bash
bash scripts/prepare_data.sh \
  --cosec-root /path/to/cosec/train \
  --split-dir data/splits/cosec
```

The script builds the sequence-level k-fold files and the default frame-list domain-cover prefix split. Set dataset paths explicitly on training/inference commands when possible. Lower-level loaders also support these environment variables:

```bash
export COSEC_ROOT=/path/to/cosec/train
export TEST_ROOT=/path/to/cosec/test
export EVENTSHIFT_COSEC_MANIFEST=/path/to/cosec_train_bidir_50ms.json
export DSEC_ROOT=/path/to/dsec
export ACDC_ROOT=/path/to/acdc
export EVENTSHIFT_COSEC_SPLIT_DIR=/path/to/cosec/splits
export ACDC_SPLIT_DIR=/path/to/acdc/splits
export DSEC_FILTERED_630_MANIFEST=/path/to/dsec19_filtered_medium_more_630.json
```

For normal test inference or the 0.4111 rebuild, only `--test-root` is required:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/cosec/test
```

For training, pass the CoSEC root and event manifest explicitly:

```bash
bash scripts/train.sh \
  --model mask2former \
  --variant eventshift \
  --cosec-root /path/to/cosec/train \
  --cosec-manifest /path/to/cosec_train_bidir_50ms.json
```

`--brenet-root` is optional and only needed for older manifests whose relative paths cannot be resolved from the manifest location.

## Dataset Roles

| Dataset | Role in EventShift | Required for |
| --- | --- | --- |
| CoSEC train | Main supervised RGB semantic segmentation data with `Day_*` and `Night_*` domains. | Training, validation, split generation, domain-gap references. |
| CoSEC test | Challenge inference data. Includes day, night, and REAL-style sequences without labels. | Inference and submission rebuild. |
| CoSEC events / BRENet assets | Event streams and manifests aligned to CoSEC frames. | Event-based CoSEC training. Not needed for RGB-only inference or 0.4111 rebuild. |
| DSEC19 | Auxiliary 19-class driving data, with optional event windows around image timestamps. | Auxiliary RGB/event training and domain coverage experiments. |
| ACDC | Adverse-condition driving data, especially night. | Auxiliary night/adverse training and CoSEC-night domain patch filtering. |
| REAL pool | Unlabeled real-domain images. The `gt` directory in the historical REAL pool is RGB imagery, not semantic ground truth. | Inference, diagnostics, and pseudo-label experiments only. |

## Expected Layouts

CoSEC train root:

```text
${COSEC_ROOT}/
|-- Day_*/
|   |-- img_co_left/000000.png
|   `-- segment_co/000000.png
`-- Night_*/
    |-- img_co_left/000000.png
    `-- segment_co/000000.png
```

CoSEC test root:

```text
${TEST_ROOT}/
|-- Day_*/img_co_left/*.png
|-- Night_*/img_co_left/*.png
`-- REAL_*/img_co_left/*.png
```

DSEC root:

```text
${DSEC_ROOT}/
|-- train_image/<sequence>/images/left/rectified/*.png
|-- train_semantic_segmentation/<sequence>/19classes/*.png
`-- train_event/<sequence>/events/left/events.h5
```

ACDC root:

```text
${ACDC_ROOT}/
|-- rgb_anon/<condition>/<split>/<sequence>/*_rgb_anon.png
`-- gt/<condition>/<split>/<sequence>/*_gt_labelTrainIds.png
```

`condition` is one of `fog`, `night`, `rain`, or `snow`. `split` is usually `train` or `val`.

## CoSEC Split Preparation

We use two active split styles: sequence-level k-fold and frame-list prefix split.

### Sequence-Level K-Fold

Use this for leakage-free validation. Whole sequences are assigned to train or validation, so frames from the same sequence never appear on both sides. The builder balances day/night domains and frame counts across folds.

```bash
python tools/data/build_cosec_kfold_splits.py \
  --root /path/to/cosec/train \
  --folds 3 \
  --write-splits \
  --split-dir /path/to/cosec/splits
```

Useful registered dataset names:

```text
cosec_kfold3_fold0_train
cosec_kfold3_fold0_val
cosec_kfold3_fold0_day_train
cosec_kfold3_fold0_day_val
cosec_kfold3_fold0_night_train
cosec_kfold3_fold0_night_val
cosec_kfold3_fold0_train_event
cosec_kfold3_fold0_val_event
cosec_kfold3_fold0_day_val_event
cosec_kfold3_fold0_night_val_event
```

Replace `fold0` with `fold1` or `fold2` for the other folds. Event variants add `_event` to the dataset name.

Example config override:

```yaml
data:
  train_dataset: cosec_kfold3_fold0_train_event
  val_datasets:
    - cosec_kfold3_fold0_day_val_event
    - cosec_kfold3_fold0_night_val_event
```

### Frame-List Prefix Splits

Use this when you need a controlled frame list, for example class coverage, domain coverage, or ablations. A prefix split is stored as two text files:

```text
${EVENTSHIFT_COSEC_SPLIT_DIR}/train_<prefix>.txt
${EVENTSHIFT_COSEC_SPLIT_DIR}/val_<prefix>.txt
```

Each line is a frame id in `sequence/frame` format:

```text
Day_Campus_001/000000
Night_Campus_003/000142
```

The loader discovers these files automatically and registers dataset names from the prefix:

```text
cosec_<prefix>_train
cosec_<prefix>_val
cosec_<prefix>_day_train
cosec_<prefix>_day_val
cosec_<prefix>_night_train
cosec_<prefix>_night_val
cosec_<prefix>_train_event
cosec_<prefix>_val_event
cosec_<prefix>_day_train_event
cosec_<prefix>_day_val_event
cosec_<prefix>_night_train_event
cosec_<prefix>_night_val_event
```

Domain/class coverage split:

```bash
python tools/data/build_cosec_domain_cover_frame_split.py \
  --root /path/to/cosec/train \
  --prefix domaincover20 \
  --val-fraction 0.20 \
  --write-splits \
  --split-dir /path/to/cosec/splits
```

Frame-level stratified CV:

```bash
python tools/data/build_cosec_stratified_frame_splits.py \
  --root /path/to/cosec/train \
  --folds 5 \
  --prefix stratframe5 \
  --write-splits \
  --split-dir /path/to/cosec/splits
```

Frame-list prefix splits can put different frames from the same sequence into train and validation. That is useful for controlled coverage studies, but use sequence-level k-fold when measuring sequence generalization.

## DSEC Preparation

DSEC loaders use 19-class labels and optionally attach event windows from `events.h5`. The default DSEC validation sequences are:

```text
zurich_city_06_a
zurich_city_07_a
zurich_city_08_a
```

Common registered dataset names:

```text
dsec19_train_filtered630
dsec19_train_filtered630_event
dsec19_train_noval
dsec19_train_noval_event
dsec19_val
dsec19_val_event
dsec19_train_close180
dsec19_train_close240
```

If you already have the filtered manifest, point the loader to it:

```bash
export DSEC_FILTERED_630_MANIFEST=/path/to/dsec19_filtered_medium_more_630.json
```

To build an MMSeg-compatible symlink view for the full DSEC19 set:

```bash
python tools/data/build_mmseg_dsec19_full_flat.py \
  --dsec-root /path/to/dsec \
  --out-dir work_dirs/mmseg/dsec19_full_flat
```

## ACDC Preparation

ACDC is used as adverse-condition auxiliary data. The most important condition for this project is `night`, but loaders also support `fog`, `rain`, `snow`, and `all`.

Common registered dataset names:

```text
acdc_all_train
acdc_all_val
acdc_night_train
acdc_night_val
acdc_night_trainval
acdc_night_top50
acdc_night_top50_repeat4
acdc_night_top50_repeat8
acdc_night_kfold3_fold0_train
acdc_night_kfold3_fold0_val
acdc_all_kfold3_fold0_train
acdc_all_kfold3_fold0_val
```

Build MMSeg split text files when needed:

```bash
python tools/data/build_mmseg_acdc_splits.py \
  --acdc-root /path/to/acdc \
  --out-dir work_dirs/mmseg/acdc_splits
```

Audit sequence-level k-fold splits:

```bash
python tools/data/build_acdc_kfold_splits.py --condition night --folds 3
python tools/data/build_acdc_kfold_splits.py --condition all --folds 3
```

Build the CoSEC-night ACDC top50 domain patch used by the `acdc_night_top50` loader:

```bash
python tools/postprocess/filter_acdc_domain_patch.py \
  --conditions night \
  --splits train,val \
  --reference-domain night \
  --selection-mode greedy \
  --keep-count 50 \
  --output-name acdc_night_trainval_cosec_night_domain_patch_top50_greedy.json \
  --latest-name acdc_night_trainval_cosec_night_domain_patch_top50_filtered_greedy.json
```

The script writes both a JSON manifest and a Markdown summary under `work_dirs/manifests/`.

## Domain-Gap Filtering

The ACDC domain-gap filter selects auxiliary samples that look closer to CoSEC, especially CoSEC night. It combines two kinds of evidence:

- Semantic class distribution: per-image class histograms are compared with Jensen-Shannon distance. Global distribution terms keep the selected subset close to the CoSEC reference distribution.
- Image statistics: low-resolution RGB/luminance/saturation summaries estimate illumination and color-domain gap. Greedy selection can improve the global match of the selected subset.

The filtering script supports `top` and `greedy` selection. `top` keeps the individually closest samples. `greedy` starts from strong candidates and then optimizes the selected subset's global class and image statistics. For the cleaned training recipes, prefer `greedy` for domain-patch subsets.

Diagnostics:

```bash
python tools/diagnostics/diagnose_real_domain_gap.py \
  --test-root /path/to/cosec/test \
  --real-root /path/to/REAL_dataset \
  --output-dir work_dirs/diagnostics
```

REAL diagnostics are for understanding and pseudo-label experiments. Do not treat REAL pool images as supervised semantic labels.

## Generated Files

The following paths are generated locally and ignored by Git:

```text
work_dirs/cache/
work_dirs/diagnostics/
work_dirs/manifests/
work_dirs/mmseg/
outputs/
artifacts/
submit/*.zip
```

Keep split manifests, diagnostics, caches, predictions, and submission archives outside committed source unless a small metadata file is intentionally added for documentation.
