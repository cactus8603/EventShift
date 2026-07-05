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

Use the `ebmv_seg` conda environment for both Mask2Former and SegFormer paths.
See `docs/ebmv_seg_environment.md` for installation details.

From this bundle directory:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/test \
  --conda /root/miniconda3/bin/conda \
  --m2f-env ebmv_seg \
  --mmseg-env ebmv_seg
```

Example on the original machine:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /work/u1621738/ebmv_eccv/eccv_segment/swin_l/data/test
```

All pipeline inputs except `TEST_ROOT`, conda environments, and installed Python packages are inside this bundle:

```text
checkpoints/
configs/
tools/export or tools/postprocess/
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
tools/export or tools/postprocess/export_mask2former_submission.py
tools/export or tools/postprocess/export_mmseg_submission.py
```

The verified local runtime environment is `ebmv_seg`:

```text
python 3.10.20
torch 2.6.0+cu124
torchvision 0.21.0+cu124
detectron2 0.6
mmsegmentation 1.2.2
mmengine 0.10.7
mmcv-lite 2.1.0
```

The runner defaults are historical; override them on this cleaned repository:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/test \
  --conda /root/miniconda3/bin/conda \
  --m2f-env ebmv_seg \
  --mmseg-env ebmv_seg \
  --device cuda:0
```

Detailed installation and validation commands are in
`docs/ebmv_seg_environment.md`.
