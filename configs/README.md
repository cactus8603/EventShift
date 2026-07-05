# Configs

```text
eventshift/               User-facing EventShift configs, variants, and rebuild recipes
  base.yaml               Shared defaults
  datasets/               Dataset path groups
  models/                 Backend/exporter model groups
  variants/               Model-specific variants selected by --model/--variant
  recipes/                Multi-stage rebuild recipes
mask2former/              Final/rebuild Mask2Former backend configs
mask2former_experiments/  Historical Swin-L event/RGB experiment configs
segformer/                Final/rebuild SegFormer backend configs
mmseg/                    MMSeg training configs
maskdino/                 MaskDINO configs
```

Use the public scripts with model/variant selection when possible:

```bash
bash scripts/infer.sh --model mask2former --variant rgb_baseline --weights /path/to/checkpoints/mask2former_rgb.pth --test-root /path/to/test
bash scripts/infer.sh --model segformer --variant night_event_04111 --weights /path/to/checkpoints/segformer_night.pth --test-root /path/to/test
```

Legacy full config paths under `configs/eventshift/*.yaml` remain supported and now compose the same base/model/variant files internally.

The `mask2former_experiments/` files were moved one directory deeper from the original 0411 bundle, and their `_BASE_` references were adjusted accordingly.
