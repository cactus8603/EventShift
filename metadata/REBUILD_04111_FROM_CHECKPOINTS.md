# Rebuild 0.4111 From Bundle Checkpoints

This note is the bundle-local path for regenerating:

```text
sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip
```

Expected hidden score reported by the user:

```text
mIoU 0.4111250649
mAcc 0.5528343575
aAcc 0.8973025919
```

Expected SHA256:

```text
4c369c3d3ce554618366a0db66189f5b92cf7ffe64ebc28ac251374d56bda46b
```

## One-Command Runner

From this bundle directory:

```bash
TEST_ROOT=/path/to/test \
bash code/rebuild_04111_b75_from_bundle_checkpoints.sh
```

Example on the original machine:

```bash
TEST_ROOT=./data/test \
bash code/rebuild_04111_b75_from_bundle_checkpoints.sh
```

All pipeline inputs except `TEST_ROOT`, conda environments, and installed Python packages are inside this bundle:

```text
checkpoints/
configs/
code/
tools/
third_party/
artifacts/
```

The output defaults to:

```text
outputs/rebuild_04111_b75_from_checkpoints_<timestamp>/
```

The final zip is written to:

```text
outputs/.../submission_zips/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip
```

## What Is Regenerated From Checkpoints

The runner regenerates these prediction masks from bundle checkpoints:

```text
1. Mask2Former Swin-L event-trained Day
   config: configs/mask2former/Mask2Former_SwinL_FullCoSEC_FromDayBest_Floor816070_LR5e-7.yaml
   checkpoint: checkpoints/m2f_event_full_cosec_from_day_best_floor816070_lr5e-7.pth

2. Mask2Former Swin-L full-desc Night
   config: configs/mask2former/Mask2Former_SwinL_FullDSEC_CoSEC_ACDC_UnifiedClassCover.yaml
   checkpoint: checkpoints/m2f_full_desc_selected_cosec_night.pth

3. SegFormer B5 event-trained Night
   config: configs/segformer/SegFormer_B5_FullCoSEC_FromNightBest_Floor546453.py
   checkpoint: checkpoints/segformer_b5_event_full_cosec_from_night_best_floor546453_lr1e-6_iter4500.pth
```

The runner then applies the same score-free post-processing gates:

```text
EventSeg Night p70 gate:
  component boundary5 >= 0.60 OR component area <= 5000

Pipeline b75 gate:
  component boundary5 >= 0.75 OR component area <= 2000
```

Finally it composes:

```text
Day/Night = generated b75 keep-real output
REAL      = bundled realgate60a5000 artifact
```

## Bundle Artifacts Used As Fixed Anchors

The original pipeline was not a pure single-model prediction. It used fixed anchor/repaired submissions as inputs to the later gates. These are physically included in this bundle under `artifacts/submission_zips/`:

```text
sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628.zip
sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629.zip
sub_04085realvote_eventday_valpairnight_p50n100c500_realgate60a5000_20260629.zip
```

The final authoritative expected zip is also included for verification:

```text
artifacts/submission_zips/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip
```

## Expected Verification

With a matching software stack and deterministic inference, the final SHA should be:

```text
4c369c3d3ce554618366a0db66189f5b92cf7ffe64ebc28ac251374d56bda46b
```

The runner compares the rebuilt final zip against:

```text
artifacts/submission_zips/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip
```

If the inference stack differs, the generated masks may differ. In that case the script still leaves all intermediate outputs and manifests in `outputs/` for debugging.

## External Dependencies

This bundle now includes local copies of:

```text
third_party/Mask2Former
third_party/detectron2
tools/mmseg_unified_metrics.py
tools/mmseg_best_score_floor.py
code/export_mask2former_submission.py
code/export_mmseg_submission.py
```

The conda environments still need the compiled/runtime packages already used by the project:

```text
mask2former env: torch, opencv, detectron2-compatible dependencies
mmseg env: torch, mmseg, mmengine
```

The runner defaults are:

```text
CONDA=conda
M2F_ENV=mask2former
MMSEG_ENV=mmseg
DEVICE=cuda:0
```

Override them if needed:

```bash
CONDA=/path/to/conda M2F_ENV=mask2former MMSEG_ENV=mmseg DEVICE=cuda:0 TEST_ROOT=/path/to/test \
bash code/rebuild_04111_b75_from_bundle_checkpoints.sh
```
