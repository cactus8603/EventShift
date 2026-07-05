# SwinL / SegFormer Event Usage Notes

This note summarizes how event information is used in the 04111 training and
rebuild bundle. The important distinction is:

- Mask2Former Swin-L has code paths that directly load event H5 files and feed
  event tensors / event-edge tensors into the model.
- The bundled SegFormer B5 inference config is RGB-only. Its `event` naming in
  the 04111 pipeline refers to an event-related training/selection branch and to
  the candidate submission used by later gates, not to raw event tensors being
  read by SegFormer at inference time.

## 1. Event Data Source

The CoSEC event dataset wrapper is:

```text
tools/cosec_event_dataset.py
```

It reads:

```text
BRENet/projects/brenet_cosec/manifests/cosec_train_bidir_50ms.json
```

Each event record contains:

```text
file_name
sem_seg_file_name
image_id
event_h5
event_old
event_new
```

`event_old` and `event_new` are the two time windows around the RGB frame. The
H5 slicer also handles `t_offset` and uses `ms_to_idx` when available, so the
event read is not a naive full-file scan.

## 2. Event Representations

### Voxel Event Tensor

`load_event_representation()` builds the tensor used by `MODEL.EVENT_FUSION`.

- `INPUT.EVENT.NUM_BINS = 5`
- old window produces 5 voxel bins
- new window produces 5 voxel bins
- final event tensor is 10 channels: `old_voxel[5] + new_voxel[5]`
- nonzero voxel values are normalized by mean/std

It also returns 4 auxiliary channels:

```text
old_density
new_density
old_pos + new_pos
old_neg + new_neg
```

The dataset mapper converts those aux channels into `event_stats`:

```text
density_total = log1p(old_density + new_density)
temporal_balance = 1 - abs(old_density - new_density) / total
polarity_balance = 1 - abs(pos - neg) / total_polarity
support = density above percentile AND temporal_balance above threshold
```

Typical support settings:

```text
SUPPORT_PERCENTILE: 50 or 60
TEMPORAL_THRESHOLD: 0.05 or 0.10
SUPPORT_DILATION: 1 or 2
```

### Event-Edge Tensor

`load_event_edge_representation()` builds the tensor used by event-edge heads,
early adapters, and boundary refiners.

For each configured radius it slices:

```text
[center_time - radius_ms, center_time + radius_ms]
```

Then it produces 3 channels per radius:

```text
normalized density_log
normalized edge_score
polarity_balance
```

So:

```text
EDGE_WINDOW_RADII_MS: [25, 50, 100] -> 9 channels
EDGE_WINDOW_RADII_MS: [50]          -> 3 channels
```

Important naming detail: the config value is a half-window radius. For example,
`50` means `[t-50ms, t+50ms]`, a 100ms centered window.

## 3. Event Signal Filtering / Usefulness Checks

There are several layers that answer "is this event signal useful enough to
trust?"

### Mapper-Level Support Mask

The Mask2Former mapper always loads the event fields for `*_event` datasets,
applies the same geometric transforms as RGB/label, and writes:

```text
dataset_dict["event"]
dataset_dict["event_edge"]
dataset_dict["event_stats"]
dataset_dict["boundary_r3"], ["boundary_r5"], ...
```

`event_stats[-1]` is the event support mask. It is created by density percentile
plus temporal-balance threshold, then optionally dilated.

The mapper also supports ablations:

```text
INPUT.EVENT.ZERO_EVENT
INPUT.EVENT.DROPOUT_PROB
INPUT.EVENT.BIN_DROPOUT_PROB
INPUT.EVENT.EDGE_DROPOUT_PROB
```

These are used to test whether event actually helps or whether the branch is
overfitting/noisy.

### Temporal / Spatial Alignment Scan

Script:

```text
tools/diagnose_event_temporal_spatial_alignment.py
```

Default grid:

```text
time offsets:     -50, -25, -10, 0, 10, 25, 50 ms
window radii:     25, 50, 100 ms
spatial shifts:   -6, 0, 6 px
edge percentiles: 70, 80, 90
GT boundary rad:  3 px
```

It compares event-edge maps against semantic GT boundaries. It also computes RGB
edge scores and combined event/RGB variants. This is the main "alignment" tool:
it checks whether event edges land on real segmentation boundaries after time
offset and small x/y shifts.

The `10ms` value appears here as a temporal offset candidate. I did not find a
main training config that uses `10ms` as an event-edge window; the common train
window radii are `[25, 50, 100]` or `[25, 50, 200]`, plus a `[50]` ablation.

### Event-Active Repair Support

Script:

```text
tools/diagnose_event_active_support.py
```

Defaults:

```text
event radii:  [50]
percentiles:  50, 70, 80, 90
dilate:       2 px
```

It thresholds event-edge score into an event-active mask and measures overlap
with:

```text
base segmentation errors
new segmentation errors
repaired pixels
damaged pixels
changed pixels
```

This is the direct "does event activity cover real repairs, or is it just noise?"
diagnostic.

### Event-Edge Cache

Script:

```text
tools/build_cosec_event_edge_cache.py
```

Defaults:

```text
datasets:          cosec_train_event,cosec_day_val_event,cosec_night_val_event
window radii:      [25, 50]
time offset:       0 ms
spatial shift:     0, 0
percentile:        80
```

It stores per-image `.npz` files with:

```text
score
mask
threshold
```

The score is the max over edge-score channels, normalized to `[0, 1]`.

## 4. SwinL / Mask2Former Event Use

Event datasets are registered in:

```text
tools/train_mask2former_cosec.py
```

Examples:

```text
cosec_train_event
cosec_day_train_event
cosec_night_train_event
cosec_day_val_event
cosec_night_val_event
dsec19_train_noval_event
dsec19_val_event
```

### A. EventFusion Branch

Config family:

```text
configs/Mask2Former_SwinL_CoSEC_EventRes34_Epoch.yaml
configs/Mask2Former_SwinL_CoSEC_EventReliability_Exp2.yaml
```

Model settings:

```text
MODEL.EVENT_FUSION.ENABLED: True
IN_CHANNELS: 10
STAT_CHANNELS: 4
STAGES: ["res3", "res4"]
```

This path consumes the 10-channel voxel event tensor plus the 4-channel
`event_stats`. `EventReliability_Exp2` adds a reliability prior using density,
temporal balance, and polarity balance:

```text
RELIABILITY_DENSITY_POWER: 0.5
RELIABILITY_TEMPORAL_POWER: 1.0
RELIABILITY_POLARITY_POWER: 0.25
RELIABILITY_FLOOR: 0.25
```

### B. Event Edge Head

Config family:

```text
configs/Mask2Former_SwinL_CoSEC_EventEdgeSemantic_Exp3a.yaml
```

Key settings:

```text
MODEL.EVENT_EDGE.ENABLED: True
MODEL.EVENT_EDGE.IN_CHANNELS: 9
INPUT.EVENT.EDGE_WINDOW_RADII_MS: [25, 50, 200]
INPUT.EVENT.BOUNDARY_RADII: [3, 5]
```

This trains an event-edge head against semantic boundary targets. It is a
pretraining / auxiliary route for learning where event edges are meaningful.

### C. Early Event Edge Adapter

Config family:

```text
configs/Mask2Former_SwinL_DSEC19_EventAdapter_Finetune.yaml
configs/Mask2Former_SwinL_DSEC19_EventAdapter_50msOnly_Finetune.yaml
```

Main version:

```text
EARLY_EVENT_EDGE_ADAPTER.ENABLED: True
IN_CHANNELS: 9
EDGE_WINDOW_RADII_MS: [25, 50, 100]
```

50ms-only ablation:

```text
IN_CHANNELS: 3
EDGE_WINDOW_RADII_MS: [50]
```

The adapter gates event correction using event reliability plus RGB confidence /
margin / entropy and boundary constraints:

```text
USE_EVENT_RELIABILITY: True
EVENT_EDGE_THRESHOLD: 0.20
CONFIDENCE_THRESHOLD: 0.65
MARGIN_THRESHOLD: 0.25
ENTROPY_THRESHOLD: 0.55
REQUIRE_BOUNDARY: True
BOUNDARY_RADIUS: 3
```

### D. Day Event Boundary Refiner

Config family:

```text
configs/Mask2Former_SwinL_CoSEC_FullCoSEC_Exp11A_Day65EventActiveMinFilter.yaml
configs/Mask2Former_SwinL_CoSEC_FullCoSEC_Exp18A_EventBoundaryMinDynamic.yaml
```

This is a conservative logit-level correction branch. It keeps the RGB path
mostly fixed and trains only:

```text
TRAINABLE_PREFIXES:
  - "day_event_boundary_refiner."
```

Exp18A is the strict version. Event can edit only when these conditions are all
true:

```text
event-active edge
RGB low-margin / high-entropy uncertainty
RGB semantic boundary
allowed class group
```

Important gates:

```text
REQUIRE_EVENT_ACTIVE: True
EVENT_ACTIVE_SOURCE: "density_or_support"
EDGE_THRESHOLD: 0.10
USE_RGB_UNCERTAINTY: True
USE_SEMANTIC_BOUNDARY: True
REQUIRE_UNCERTAIN_BOUNDARY_INTERSECTION: True
EDGE_ONLY_CORRECTION: True
SKIP_MASK2FORMER_LOSS: True
```

This matches the idea that event should only be allowed to repair pixels in
small, boundary-like, event-active regions.

## 5. SegFormer Event Use In This Bundle

The relevant config is:

```text
configs/segformer/SegFormer_B5_FullCoSEC_FromNightBest_Floor546453.py
```

Confirmed from the config:

```text
backbone.in_channels = 3
pipeline = LoadImageFromFile / LoadAnnotations / PackSegInputs
```

There is no event H5 loader, no event tensor, and no event-edge channel in the
bundled SegFormer inference path. So the safest interpretation is:

- `segformer_b5_event_full_cosec_from_night...pth` is an event-related
  experiment/checkpoint name.
- At inference/rebuild time it outputs normal RGB semantic masks.
- Those masks are used as an "EventSeg Night" candidate in the final submission
  pipeline.

## 6. Final 0.4111 Rebuild Role

Document:

```text
docs/rebuild_04111_from_checkpoints.md
```

Runner:

```text
scripts/rebuild_04111.sh --recipe configs/eventshift/recipes/rebuild_04111_b75.yaml
```

The runner regenerates:

```text
1. Mask2Former Swin-L event-trained Day
2. Mask2Former Swin-L full-desc Night
3. SegFormer B5 event-trained Night
```

Then it applies score-free submission-delta filters.

First, SegFormer Night is filtered against Mask2Former full-desc Night:

```text
base:      raw/mask2former_night_full_desc
candidate: raw/segformer_night_event
domain:    Night
gate:      component boundary5 >= 0.60 OR component area <= 5000
```

The allowed class transitions are explicitly listed in the rebuild script, for
example:

```text
building->road
wall->road
fence->road
vegetation->road
vegetation->sidewalk
sky->building
sky->vegetation
```

Then the pipeline compares the main keep-real output against the eventseg-night
candidate and applies a stricter gate:

```text
component boundary5 >= 0.75 OR component area <= 2000
```

Finally:

```text
Day/Night = generated b75 keep-real output
REAL      = bundled realgate60a5000 artifact
```

The key takeaway is that the final 04111 zip is not a single direct event model
output. It is an ensemble/rebuild pipeline where SwinL event-trained branches,
SegFormer event-named Night candidates, transition whitelists, component
boundary gates, and fixed REAL anchors are composed conservatively.

## 7. Quick Mental Model

```text
event H5
  -> time slice with t_offset/ms_to_idx
  -> voxel event tensor: old/new 5 bins = 10 channels
  -> aux stats: density/temporal/polarity/support
  -> event-edge tensor: 3 channels per radius
  -> diagnostics: active support + temporal/spatial alignment
  -> SwinL event fusion / edge head / early adapter / boundary refiner
  -> submission candidates
  -> transition + boundary5 + area gates
  -> final 04111 zip
```

