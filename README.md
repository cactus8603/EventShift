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

## Third-Party Code

EventShift keeps the runtime-critical third-party source under `third_party/` so the challenge pipeline can be rebuilt without relying on moving external checkouts. Only the pieces used by the current training/inference paths are documented here.

| Vendored path | Upstream project | Used for | License / citation |
| --- | --- | --- | --- |
| `third_party/Mask2Former/` | [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former) | Mask2Former Swin-L configs, trainer base class, semantic dataset mapper, TTA wrapper, and pixel decoder ops. | Mostly MIT; upstream notes also mention Swin Transformer MIT and Deformable DETR Apache-2.0 portions. Cite Mask2Former and MaskFormer. |
| `third_party/detectron2/` | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) | Detectron2 config, data catalog, training loop, checkpointing, evaluation, and project utilities used by the Mask2Former path. | Apache-2.0. Cite Detectron2. |
| `third_party/mmsegmentation/` | [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation) | Source subset for `mmseg`, `tools/train.py`, and `tools/test.py` used by SegFormer/MMSeg inference and training utilities. | Apache-2.0 upstream. Cite MMSegmentation and SegFormer. |

The vendored directories are source-only; compiled artifacts are intentionally omitted. Rebuild native extensions in the target environment when needed. See `third_party/README.md` for the shorter provenance table and install commands.

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

## Citation

The EventShift technical report citation will be added after the ECCV 2026 EBMV Workshop Challenge report is finalized. If you use this repository before then, please also cite the upstream projects used by the implementation.

<details>
<summary>Third-party BibTeX</summary>

```bibtex
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}

@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}

@misc{wu2019detectron2,
  author={Yuxin Wu and Alexander Kirillov and Francisco Massa and Wan-Yen Lo and Ross Girshick},
  title={Detectron2},
  howpublished={\url{https://github.com/facebookresearch/detectron2}},
  year={2019}
}

@misc{mmseg2020,
  title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
  author={MMSegmentation Contributors},
  howpublished={\url{https://github.com/open-mmlab/mmsegmentation}},
  year={2020}
}

@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M. and Luo, Ping},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

</details>

## License

EventShift-specific code is released under the MIT License; see `LICENSE`. Third-party source under `third_party/` keeps its upstream license terms:

| Component | License note |
| --- | --- |
| Mask2Former | Mostly MIT, with separate MIT/Apache-2.0 portions noted by upstream. |
| Detectron2 | Apache-2.0 upstream. |
| MMSegmentation | Apache-2.0 upstream. |

Datasets, checkpoints, and pretrained weights are governed by their own upstream or challenge terms and are not redistributed in this repository.

## Acknowledgements

This repository builds on Mask2Former, Detectron2, MMSegmentation, SegFormer, DSEC/DSEC-Semantic, ACDC, and the CoSEC challenge resources. We thank the authors and maintainers for making their code and datasets available to the research community.
