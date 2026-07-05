# EventShift

**EventShift: Event-Guided Illumination-Robust Semantic Segmentation for CoSEC**

EventShift is our solution for the **CoSEC Semantic Segmentation Track** of the **ECCV 2026 EBMV Workshop Challenge**. The method targets semantic segmentation under challenging illumination conditions by using event streams as complementary cues to RGB images.

RGB images may suffer from severe illumination shift under low-light or exposure-changing scenes. Event signals provide additional motion and boundary information that is less dependent on absolute image brightness. EventShift uses these event cues to improve semantic segmentation robustness under difficult lighting conditions.

## Highlights

- Event-guided semantic segmentation for CoSEC.
- Args-first inference and rebuild entry points.
- Composable configs with base, dataset, model, variant, and recipe files.
- Supports RGB, event, and RGB-event fusion settings.
- Checkpoints, datasets, generated outputs, and submission archives are not stored in Git.

## Repository Structure

```text
EventShift/
|-- configs/
|   |-- eventshift/       # Base configs, model variants, and rebuild recipes
|   |-- mask2former/      # Final Mask2Former backend configs
|   |-- segformer/        # Final SegFormer/MMSeg backend configs
|   |-- maskdino/         # MaskDINO backend configs
|   `-- mmseg/            # MMSeg training configs
|-- eventshift/           # Core Python package and config/backend helpers
|-- scripts/              # Thin user-facing shell entry points
|-- tools/                # Training, inference, export, rebuild, and post-processing tools
|-- third_party/          # Required third-party source code
|-- docs/                 # Notes and additional documentation
|-- metadata/             # Original notes, manifests, and provenance
|-- data/                 # Placeholder for local datasets
|-- checkpoints/          # Placeholder for local checkpoints
|-- outputs/              # Placeholder for generated predictions
|-- submit/               # Placeholder for local submission archives
|-- requirements.txt
|-- environment.yml
`-- README.md
```

Historical one-off shell scripts are preserved under `scripts/archive/` and `tools/rebuild/archive/`. The recommended public entry points are `scripts/infer.sh`, `scripts/train.sh`, `scripts/eval.sh`, `scripts/prepare_data.sh`, and `scripts/rebuild_04111.sh`.

## Environment

We recommend using the provided conda environment file:

```bash
conda env create -f environment.yml
conda activate ebmv_seg
```

Install additional third-party dependencies if needed:

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

See `docs/ebmv_seg_environment.md` for exact setup notes, validation commands, and troubleshooting.

If native Mask2Former operators are required, rebuild them locally:

```bash
cd third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd -
```

## Config Layout

EventShift configs are composed from a small set of reusable files:

```text
configs/eventshift/
|-- base.yaml
|-- datasets/
|-- models/
|-- variants/
|   |-- mask2former/
|   `-- segformer/
`-- recipes/
```

The public scripts accept either a legacy full config or a model/variant pair. The model chooses the backend exporter, and the variant supplies the backend config, checkpoint, output defaults, sequence list, and TTA options.

```bash
bash scripts/infer.sh \
  --model mask2former \
  --variant rgb_baseline \
  --test-root /path/to/test \
  --weights checkpoints/m2f_full_desc_selected_cosec_day.pth
```

Legacy config paths remain supported:

```bash
bash scripts/infer.sh \
  --config configs/eventshift/cosec_rgb_baseline.yaml \
  --test-root /path/to/test
```

## Dataset Preparation

Raw datasets are expected to be placed outside the repository. User-facing scripts accept dataset locations as command-line arguments:

```text
--test-root        CoSEC test root for inference and submission export
--cosec-root       CoSEC training root containing Day_* and Night_* sequences
--brenet-root      BRENet root containing CoSEC event assets for event-based training
--cosec-manifest   CoSEC event manifest JSON for event-based training
--dsec-root        DSEC dataset root for auxiliary training data
--acdc-root        ACDC dataset root for auxiliary training data
```

The 0.4111 rebuild and normal test-set inference only need `--test-root`. BRENet is only needed when training or preparing event-based CoSEC datasets.

### Dataset Collection

| Dataset | Source and collection | How it is used here |
| --- | --- | --- |
| CoSEC train/test | Official CoSEC challenge data. The challenge provides RGB driving frames and aligned event assets organized by illumination domain, including `Day_*`, `Night_*`, and unlabeled test/REAL-style sequences. | Primary target data for training, validation, inference, and submission export. |
| CoSEC events / BRENet assets | Local event assets and manifests derived from the challenge-provided CoSEC event streams, aligned to CoSEC frame ids. | Event-based CoSEC training and RGB-event fusion variants. |
| [DSEC](https://arxiv.org/abs/2103.06011) / [DSEC-Semantic](https://arxiv.org/abs/2203.10016) | Public driving dataset collected in Switzerland with synchronized RGB frame cameras, high-resolution event cameras, lidar, and RTK GPS across daytime, nighttime, and difficult illumination. DSEC-Semantic provides the 19-class semantic label layout used by this repo. | Auxiliary RGB/event segmentation data and DSEC-close subsets for domain coverage. |
| [ACDC](https://arxiv.org/abs/2104.13395) | Public adverse-condition driving dataset collected for robust semantic scene understanding, with fog, night, rain, and snow images plus pixel-level labels. | Auxiliary adverse-condition data. We mainly use night data and build an ACDC top50 CoSEC-night domain patch with the filtering tools. |
| REAL pool | Unlabeled real-domain sequences supplied with the challenge/test assets or local REAL pool. The historical `REAL_dataset/*/gt` directory contains RGB imagery, not semantic labels. | Inference, diagnostics, and pseudo-label experiments only. It is not used as supervised ground truth. |

See `docs/dataset_preparation.md` for dataset layouts, CoSEC sequence-level k-fold splits, frame-list prefix splits, ACDC domain-gap filtering, and generated manifest locations. New experiments should use k-fold splits for sequence-level validation and frame-list prefix splits for controlled domain/class coverage studies.

## Inference

Mask2Former example:

```bash
bash scripts/infer.sh \
  --model mask2former \
  --variant rgb_baseline \
  --test-root /path/to/test \
  --out-dir outputs/infer_rgb
```

SegFormer example:

```bash
bash scripts/infer.sh \
  --model segformer \
  --variant night_event_04111 \
  --test-root /path/to/test \
  --out-dir outputs/infer_segformer_night
```

By default the command is printed. Add `--execute` to run it. Extra backend options can be passed after `--`.

## Rebuild 0.4111

The 0.4111 b75 submission is rebuilt through one public shell entry point and one recipe:

```bash
bash scripts/rebuild_04111.sh \
  --recipe configs/eventshift/recipes/rebuild_04111_b75.yaml \
  --test-root /path/to/test \
  --conda /root/miniconda3/bin/conda \
  --m2f-env ebmv_seg \
  --mmseg-env ebmv_seg \
  --device cuda:0
```

The recipe regenerates two Mask2Former exports and one SegFormer export, then applies the bundled post-processing gates. Each model export uses a tqdm progress bar labeled with the model name.

For a quick argument and path check without running inference:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/test \
  --conda /root/miniconda3/bin/conda \
  --m2f-env ebmv_seg \
  --mmseg-env ebmv_seg \
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

## Results

The final challenge results and technical report details will be updated after the ECCV 2026 EBMV Workshop Challenge report is finalized.

## Notes

This repository contains cleaned and reorganized code for reproducibility. Some historical experiment scripts are preserved under `tools/` and `docs/` for reference, but the recommended entry points are the scripts under `scripts/`.

For the 0.4111 rebuild workflow, see `docs/rebuild_04111_from_checkpoints.md`.

## License

This project is released under the MIT License.

## Acknowledgements

This repository builds upon several open-source semantic segmentation and event-based vision codebases. We thank the authors of the related projects for making their code publicly available.
