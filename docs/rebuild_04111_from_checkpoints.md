# Rebuild 0.4111 From Bundle Checkpoints

This note documents the recipe-driven path for regenerating:

```text
sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip
```

Expected hidden score reported by the user:

```text
mIoU 0.4111250649
mAcc 0.5528343575
aAcc 0.8973025919
```

## One-Command Runner

Run from the activated `ebmv_seg` environment. See `docs/ebmv_seg_environment.md` for installation details.

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/test \
  --device cuda:0
```

The output defaults to:

```text
outputs/rebuild_04111_b75_from_checkpoints_<timestamp>/
```

The final zip is written to:

```text
outputs/.../submission_zips/sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip
```

For a quick path and argument check without model inference:

```bash
bash scripts/rebuild_04111.sh \
  --test-root /path/to/test \
  --smoke-limit 0 \
  --skip-inference
```

## Recipe

The rebuild is controlled by:

```text
configs/eventshift/recipes/rebuild_04111_b75.yaml
```

That recipe points to three model variants:

```text
configs/eventshift/variants/mask2former/day_event_04111.yaml
configs/eventshift/variants/mask2former/night_full_desc_04111.yaml
configs/eventshift/variants/segformer/night_event_04111.yaml
```

The variants select the backend config, checkpoint, sequence group, TTA options, and tqdm progress label. The runner loads the correct exporter through the backend registry instead of relying on separate shell scripts per model.

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

## Fixed Anchor Artifacts

The original pipeline was not a pure single-model prediction. It used fixed anchor/repaired submissions as inputs to the later gates. These are included under `artifacts/submission_zips/`:

```text
sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628.zip
sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629.zip
sub_04085realvote_eventday_valpairnight_p50n100c500_realgate60a5000_20260629.zip
```

The local submitted reference is:

```text
submit/sub_pipeline_b75.zip
```

## Verification

The runner prints the generated zip SHA. Zip container bytes may differ because archive metadata is not stable, so the pass/fail check compares archive contents: entry names and PNG bytes.

On this workspace, a no-inference rebuild from existing raw masks produced a final archive with 982 entries whose contents matched `submit/sub_pipeline_b75.zip` exactly.

If the inference stack differs, regenerated masks may differ. The script leaves all intermediate outputs and reports under the selected output root for debugging.
