# EventShift

**EventShift: Event-Guided Illumination-Robust Semantic Segmentation for CoSEC**

EventShift is our solution for the **CoSEC Semantic Segmentation Track** of the **ECCV 2026 EBMV Workshop Challenge**.
The method targets semantic segmentation under challenging illumination conditions by using event streams as complementary cues to RGB images.

RGB images may suffer from severe illumination shift under low-light or exposure-changing scenes. Event signals provide additional motion and boundary information that is less dependent on absolute image brightness. EventShift uses these event cues to improve semantic segmentation robustness under difficult lighting conditions.

## Highlights

- Event-guided semantic segmentation for CoSEC.
- Designed for illumination-robust RGB-event perception.
- Supports RGB, event, and RGB-event fusion settings.
- Includes cleaned configuration files, inference scripts, and submission utilities.
- Checkpoints and large datasets are not stored in this repository.

## Repository Structure

```text
EventShift/
|-- configs/              # Training and inference configs
|-- eventshift/           # Core Python package
|-- scripts/              # User-facing shell scripts
|-- tools/                # Training, inference, export, and post-processing tools
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

Large files such as datasets, checkpoints, generated outputs, and submission archives are ignored by Git.

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

## Dataset Preparation

Raw datasets are expected to be placed outside the repository. User-facing scripts accept dataset locations as command-line arguments:

```text
--cosec-root       CoSEC training root containing Day_* and Night_* sequences
--brenet-root      BRENet root containing CoSEC event assets
--cosec-manifest   CoSEC event manifest JSON
--dsec-root        DSEC dataset root
--acdc-root        ACDC dataset root
--test-root        CoSEC test root for inference and submission export
```

For CoSEC event-based training or inference, prepare the event manifest before running the scripts. The default manifest location is:

```text
/path/to/BRENet/projects/brenet_cosec/manifests/cosec_train_bidir_50ms.json
```

The same paths can also be provided through environment variables (`COSEC_ROOT`, `BRENET_ROOT`, `EVENTSHIFT_COSEC_MANIFEST`, `DSEC_ROOT`, `ACDC_ROOT`, and `TEST_ROOT`) when that is more convenient.

## Inference

Print the inference command with:

```bash
bash scripts/infer.sh configs/eventshift/cosec_eventshift.yaml \
  --weights checkpoints/model.pth \
  --test-root /path/to/test \
  --cosec-root /path/to/cosec/train \
  --brenet-root /path/to/BRENet \
  --cosec-manifest /path/to/cosec_train_bidir_50ms.json
```

Run inference by adding `--execute`:

```bash
bash scripts/infer.sh configs/eventshift/cosec_eventshift.yaml \
  --weights checkpoints/model.pth \
  --test-root /path/to/test \
  --cosec-root /path/to/cosec/train \
  --brenet-root /path/to/BRENet \
  --cosec-manifest /path/to/cosec_train_bidir_50ms.json \
  --execute
```

Generated predictions will be saved under the output directory specified in the config or script.

## Rebuild 0.4111

The 0.4111 b75 submission can be rebuilt through one user-facing shell entry point:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/test \
  --conda /root/miniconda3/bin/conda \
  --m2f-env ebmv_seg \
  --mmseg-env ebmv_seg \
  --device cuda:0
```

The runner regenerates two Mask2Former exports and one SegFormer export, then applies the bundled post-processing steps. Each model export uses a tqdm progress bar labeled with the model name.

For a quick argument and path check without running inference, use:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/test \
  --conda /root/miniconda3/bin/conda \
  --m2f-env ebmv_seg \
  --mmseg-env ebmv_seg \
  --smoke-limit 0 \
  --skip-inference
```

## Training

To dry-run the training command:

```bash
bash scripts/train.sh configs/eventshift/cosec_eventshift.yaml \
  --cosec-root /path/to/cosec/train \
  --brenet-root /path/to/BRENet \
  --cosec-manifest /path/to/cosec_train_bidir_50ms.json
```

To launch training:

```bash
bash scripts/train.sh configs/eventshift/cosec_eventshift.yaml \
  --cosec-root /path/to/cosec/train \
  --brenet-root /path/to/BRENet \
  --cosec-manifest /path/to/cosec_train_bidir_50ms.json \
  --execute
```

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

For the historical 0411 rebuild workflow, see `docs/rebuild_04111_from_checkpoints.md`.

## License

This project is released under the MIT License.

## Acknowledgements

This repository builds upon several open-source semantic segmentation and event-based vision codebases. We thank the authors of the related projects for making their code publicly available.
