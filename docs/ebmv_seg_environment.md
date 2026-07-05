# ebmv_seg Environment

This project was reproduced with one conda environment named `ebmv_seg` for both
Mask2Former/Swin-L and SegFormer/MMSeg inference.

The successful commands used the environment like this:

```bash
bash scripts/rebuild_04111.sh \
  --recipe configs/eventshift/recipes/rebuild_04111_b75.yaml \
  --test-root /code/ebmv/portable_submission_bundle_v3_20260629/test \
  --conda /root/miniconda3/bin/conda \
  --m2f-env ebmv_seg \
  --mmseg-env ebmv_seg \
  --device cuda:0
```

## Verified Local Environment

The local environment used during cleanup/rebuild was:

```text
conda env: ebmv_seg
python: 3.10.20
torch: 2.6.0+cu124
torchvision: 0.21.0+cu124
torch CUDA runtime: 12.4
GPU used: NVIDIA GeForce RTX 4090
opencv-python: 4.13.0.92
numpy: 2.2.6
PyYAML: 6.0.3
timm: 1.0.27
detectron2: 0.6
mmsegmentation: 1.2.2
mmengine: 0.10.7
mmcv-lite: 2.1.0
```

Important compatibility notes:

- PyTorch 2.6 changed `torch.load` defaults to `weights_only=True`; the export
  scripts include compatibility handling for trusted local 0411 checkpoints.
- This environment uses `mmcv-lite`, not full `mmcv` with compiled ops. The
  bundled mmsegmentation source has optional-import guards for modules that are
  not used by the 0411 SegFormer inference path.
- Detectron2 should be installed from this repository's
  `third_party/detectron2`, not from the old source bundle path.
- `third_party/Mask2Former` and `third_party/mmsegmentation` are used through
  `PYTHONPATH` by the rebuild scripts.

## Install From Scratch

From the EventShift repository root:

```bash
cd /code/ebmv/EventShift
conda env create -f environment.yml
conda activate ebmv_seg
```

If `environment.yml` is not used, the equivalent manual setup is:

```bash
conda create -n ebmv_seg python=3.10 pip setuptools wheel ninja -y
conda activate ebmv_seg

pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124

pip install -r requirements.txt
```

Install the repository-local Detectron2 source after the base packages:

```bash
cd /code/ebmv/EventShift
pip install --no-build-isolation -e third_party/detectron2
```

Install Mask2Former Python dependencies if they are missing:

```bash
cd /code/ebmv/EventShift
pip install -r third_party/Mask2Former/requirements.txt
```

Mask2Former's pixel decoder ops are source-only in this submission. Rebuild them
on the target machine only if the local import path requires the native
extension:

```bash
cd /code/ebmv/EventShift/third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
cd /code/ebmv/EventShift
```

## Quick Validation

Run these probes after installation:

```bash
conda run -n ebmv_seg python -c "import torch, cv2, yaml, timm, detectron2, mmseg, mmengine, mmcv; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); print('cv2', cv2.__version__); print('detectron2', detectron2.__version__); print('mmseg', mmseg.__version__); print('mmengine', mmengine.__version__); print('mmcv', mmcv.__version__)"
```

Expected output should include CUDA availability on a GPU machine and versions
matching the list above.

## Rebuild 0.4111 Submission

Required local inputs:

```text
checkpoints/*.pth
artifacts/submission_zips/*.zip
TEST_ROOT containing the CoSEC/DSEC test folders
```

Run:

```bash
cd /code/ebmv/EventShift
bash scripts/rebuild_04111.sh \
  --recipe configs/eventshift/recipes/rebuild_04111_b75.yaml \
  --test-root /path/to/test \
  --conda /root/miniconda3/bin/conda \
  --m2f-env ebmv_seg \
  --mmseg-env ebmv_seg \
  --device cuda:0
```

The final generated zip is written under:

```text
outputs/rebuild_04111_b75_from_checkpoints_<timestamp>/submission_zips/
```

The local submitted reference is:

```text
submit/sub_pipeline_b75.zip
```

The runner prints the generated zip SHA and then compares archive contents. Zip container bytes can differ because metadata is not stable; the reliable check is that the entry list and PNG bytes match the submitted reference.

## Reproducibility Knobs

The rebuild script defaults to deterministic mode:

```text
EVENTSHIFT_DETERMINISTIC=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

This reduces CUDA/PyTorch boundary-pixel drift but may be slower. To disable it:

```bash
bash scripts/rebuild_04111.sh --test-root /path/to/test --non-deterministic
```
