# Third-Party Code

This directory contains source-only copies or trimmed source subsets used by the EventShift runtime. Generated binaries, build outputs, model weights, and datasets are intentionally excluded.

| Path | Upstream | Used by EventShift |
| --- | --- | --- |
| `Mask2Former/` | <https://github.com/facebookresearch/Mask2Former> | Mask2Former Swin-L trainer, configs, semantic mapper, TTA wrapper, and pixel decoder ops. |
| `detectron2/` | <https://github.com/facebookresearch/detectron2> | Detectron2 config/data/training/checkpoint/evaluation stack for the Mask2Former path. |
| `mmsegmentation/` | <https://github.com/open-mmlab/mmsegmentation> | Trimmed MMSegmentation source subset for SegFormer/MMSeg configs, train/test scripts, and runtime imports. |

Install or rebuild the pieces needed by your environment:

```bash
pip install --no-build-isolation -e third_party/detectron2
pip install -r third_party/Mask2Former/requirements.txt
cd third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

For upstream licenses and model zoo terms, consult the linked upstream repositories. EventShift-specific code is covered by the repository-level `LICENSE`.
