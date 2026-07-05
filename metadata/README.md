# Repro Bundle 20260629 Portable v2

這個目錄是可搬移的 non-symlink package。`checkpoints/`、`artifacts/`、`code/`、`configs/` 都是實體檔案，不是捷徑；來源 absolute path 仍保留在 `checkpoints_manifest.tsv` 和 `artifacts_manifest.tsv`，方便回查 provenance。

本 v2 bundle 的工作範圍只限：

```text
./work_dirs/submissions/repro_bundle_20260629_portable_v2/
```

checkpoint 已實體複製到 `checkpoints/`，總量約 13G。submission zip/report/allow-pairs 也已實體複製到 `artifacts/`。v2 在原 portable bundle 基礎上補入主線 event-trained Day/Night replacement 會用到的兩個模型、config、以及 export script。

同目錄上層另有已打包的 tar：

```text
./work_dirs/submissions/repro_bundle_20260629_portable.tar
size: 11G
SHA256: 3f0df11758ed8a835541b1a5037d8666db2f242f44582e2eccbab22e427738fb
tar symlink entries: 0
```

v2 tar：

```text
./work_dirs/submissions/repro_bundle_20260629_portable_v2.tar
size: 13G
SHA256: recorded in sibling file `repro_bundle_20260629_portable_v2.tar.sha256`
tar symlink entries: 0
```

2026-06-30 refreshed tar with the 0.4111 checkpoint rebuild runner and bundled third-party code:

```text
./work_dirs/submissions/repro_bundle_20260629_portable_v2_checkpoint_rebuild_04111.tar
size: 13G
SHA256: recorded in sibling file `repro_bundle_20260629_portable_v2_checkpoint_rebuild_04111.tar.sha256`
tar symlink entries: 0
```

2026-06-30 also added the training-side source/config pack:

```text
training/README_TRAINING.md
training/scripts/train_mask2former_from_bundle.sh
training/scripts/train_segformer_from_bundle.sh
training/scripts/train_maskdino_from_bundle.sh
training/selected_run_configs/
```

This training pack collects the code/config snapshots needed to audit or relaunch
the model families behind the 0.4111 candidate. It does not duplicate the
training datasets.

Training-code tar:

```text
./work_dirs/submissions/repro_bundle_20260629_portable_v2_traincode_04111.tar
size: 13G
SHA256: recorded in sibling file `repro_bundle_20260629_portable_v2_traincode_04111.tar.sha256`
tar symlink entries: 0
```

## 目前主線 Candidate

目前主線候選是：

```text
0.4085 anchor
+ Day event branch
+ Night val-positive repair-pair gate
+ REAL keep 0.4085
```

主線輸出：

```text
./work_dirs/submissions/submission_zips/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629.zip
SHA256: 58f94a40977ad90092fbd2a7789ce984b4fd9abc50da509e1bc55e55e5556dc7
```

對 0.4085 anchor 的 footprint report：

```text
./work_dirs/submissions/reports/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_explainable_delta_20260629.json
```

主線 domain 組成：

```text
Day   = work_dirs/submissions/prediction_dirs/event_replacements_tta4flip_20260628_raw/mask2former_day_event
Night = work_dirs/submissions/composed/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629
REAL  = work_dirs/submissions/composed/rerun_04085_realvote_verify_20260629_latest
```

## Event-Trained Models Added In v2

v2 補入主線會用到的 event-trained replacement model files；以下都是 bundle 內實體檔案，不是 symlink。

| model | source | bundle path | purpose | TTA |
| --- | --- | --- | --- | --- |
| Event/full-CoSEC Mask2Former Swin-L Day | `./work_dirs/swinL_full_cosec_from_day_best_floor816070_lr5e-7_savefix_20260628_225858/` | config: `configs/mask2former/Mask2Former_SwinL_FullCoSEC_FromDayBest_Floor816070_LR5e-7.yaml`; checkpoint: `checkpoints/m2f_event_full_cosec_from_day_best_floor816070_lr5e-7.pth` | exports Day event replacement predictions used by the main candidate Day branch | min sizes `[512,624,768,1024]`, max 1600, flip True |
| Event-trained/full-CoSEC SegFormer B5 Night | `./work_dirs/mmseg/segformer_b5_full_cosec_from_night_best_floor546453_lr1e-6/` | config: `configs/segformer/SegFormer_B5_FullCoSEC_FromNightBest_Floor546453.py`; checkpoint: `checkpoints/segformer_b5_event_full_cosec_from_night_best_floor546453_lr1e-6_iter4500.pth` | exports Night event-trained replacement/reference predictions for the event replacement pipeline | scales `s512+s624+s768+s1024`, flip |

Shared export script:

```text
code/export_event_replacement_tta_predictions.sh
source: ./tools/export_event_replacement_tta_predictions.sh
SHA256: 59d4fc33b191286df5ad628617cfb4c3f9bd87f4435397f25945a9fe766ba039
```

## Val-Positive Day Alternative

另一條較乾淨但尚未 hidden-proven 的候選是：

```text
full-desc M2F Day
+ SegFormer Day val-positive transition repair
+ Night val-positive repair-pair gate
+ REAL keep 0.4085
```

輸出：

```text
./work_dirs/submissions/submission_zips/sub_fulldescday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629.zip
SHA256: 2604c547a6092b425ec00ea522ccdb568db3fd0aba751a3844e7d7a1b67c6330
```

這條路線把 Day 也改成 unified val 上可解釋的 local repair，不再依賴 eventday branch。它在 Day val 上的 gate evidence 是 `81.5921 -> 81.8267`，但 hidden test 不一定會比 eventday 主線好。

## 重現流程

以下命令以 `.` 為工作目錄。這些命令是 CPU/IO pipeline 記錄，不在本 bundle 內重新跑 GPU cache。

1. 從 val cache 產生 Night allow-pairs：

```bash
env PYTHONNOUSERSITE=1 conda run --no-capture-output -n mask2former \
  python tools/score_free_repair_gate_from_npz.py \
  --dataset cosec_unified_classcover_v1_night_val \
  --base-map-dir work_dirs/ensemble_feature_cache/full_desc_m2f_night_unified_val_tta/maps/cosec_unified_classcover_v1_night_val \
  --candidate-map-dir work_dirs/ensemble_feature_cache/full_desc_segformer_night_unified_val_tta/maps/cosec_unified_classcover_v1_night_val \
  --out work_dirs/diagnostics/score_free_repair_gate/full_desc_night_segformer_vs_m2f_pairs_b5comp60a5000_p50_n100_c500_20260629.json \
  --allow-pairs-out work_dirs/diagnostics/score_free_repair_gate/full_desc_night_segformer_vs_m2f_pairs_b5comp60a5000_p50_n100_c500_20260629.txt \
  --min-net 100 \
  --min-precision 0.50 \
  --min-changed 500 \
  --component-min-boundary5-rate 0.60 \
  --component-max-area 5000
```

1b. 可選：從 val cache 產生 Day allow-pairs：

```bash
env PYTHONNOUSERSITE=1 conda run --no-capture-output -n mask2former \
  python tools/score_free_repair_gate_from_npz.py \
  --dataset cosec_unified_classcover_v1_day_val \
  --base-map-dir work_dirs/ensemble_feature_cache/full_desc_m2f_day_unified_val_tta/maps/cosec_unified_classcover_v1_day_val \
  --candidate-map-dir work_dirs/ensemble_feature_cache/full_desc_segformer_day_unified_val_tta/maps/cosec_unified_classcover_v1_day_val \
  --out work_dirs/diagnostics/score_free_repair_gate/full_desc_day_segformer_vs_m2f_pairs_b5comp60a5000_p50_n100_c500_20260629.json \
  --allow-pairs-out work_dirs/diagnostics/score_free_repair_gate/full_desc_day_segformer_vs_m2f_pairs_b5comp60a5000_p50_n100_c500_20260629.txt \
  --min-net 100 \
  --min-precision 0.50 \
  --min-changed 500 \
  --component-min-boundary5-rate 0.60 \
  --component-max-area 5000
```

2. 將 allow-pairs 套到 Night test candidate，產生 Night repair intermediate：

```bash
ALLOW_PAIRS=$(cat work_dirs/diagnostics/score_free_repair_gate/full_desc_night_segformer_vs_m2f_pairs_b5comp60a5000_p50_n100_c500_20260629.txt)

env PYTHONNOUSERSITE=1 conda run --no-capture-output -n mask2former \
  python tools/filter_submission_delta_by_transition.py \
  --base work_dirs/submissions/submission_zips/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628.zip \
  --candidate work_dirs/submissions/submission_zips/sub_04085realvote_eventday_nightrswbveg_b5comp60a5000_keepreal_classpatch_20260629.zip \
  --out-dir work_dirs/submissions/composed/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629 \
  --zip work_dirs/submissions/submission_zips/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629.zip \
  --summary work_dirs/submissions/reports/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_summary_20260629.json \
  --domains Night \
  --allow-pairs "$ALLOW_PAIRS" \
  --component-min-boundary5-rate 0.60 \
  --component-max-area 5000 \
  --overwrite
```

2b. 可選：將 Day allow-pairs 套到 full-desc M2F Day / SegFormer Day test predictions：

```bash
ALLOW_PAIRS_DAY=$(cat work_dirs/diagnostics/score_free_repair_gate/full_desc_day_segformer_vs_m2f_pairs_b5comp60a5000_p50_n100_c500_20260629.txt)

env PYTHONNOUSERSITE=1 conda run --no-capture-output -n mask2former \
  python tools/filter_submission_delta_by_transition.py \
  --base work_dirs/submissions/prediction_dirs/full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628_raw/mask2former_day \
  --candidate work_dirs/submissions/prediction_dirs/full_desc_cosec_acdc_m2f_segformer_maskdino_tta_domainweights_gpu0_now_20260628_raw/segformer_day \
  --out-dir work_dirs/submissions/composed/full_desc_m2fday_segformer_valrepairpairs_p50n100c500_b5comp60a5000_20260629 \
  --zip work_dirs/submissions/submission_zips/full_desc_m2fday_segformer_valrepairpairs_p50n100c500_b5comp60a5000_20260629.zip \
  --summary work_dirs/submissions/reports/full_desc_m2fday_segformer_valrepairpairs_p50n100c500_b5comp60a5000_summary_20260629.json \
  --domains Day \
  --allow-pairs "$ALLOW_PAIRS_DAY" \
  --component-min-boundary5-rate 0.60 \
  --component-max-area 5000 \
  --overwrite
```

3. Compose main eventday submission：

```bash
env PYTHONNOUSERSITE=1 conda run --no-capture-output -n mask2former \
  python tools/compose_domain_submission.py \
  --day-dir work_dirs/submissions/prediction_dirs/event_replacements_tta4flip_20260628_raw/mask2former_day_event \
  --night-dir work_dirs/submissions/composed/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629 \
  --real-dir work_dirs/submissions/composed/rerun_04085_realvote_verify_20260629_latest \
  --test-root data/test \
  --out-dir work_dirs/submissions/composed/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629 \
  --zip work_dirs/submissions/submission_zips/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629.zip \
  --overwrite
```

3b. 可選：Compose full-desc Day val-positive alternative：

```bash
env PYTHONNOUSERSITE=1 conda run --no-capture-output -n mask2former \
  python tools/compose_domain_submission.py \
  --day-dir work_dirs/submissions/composed/full_desc_m2fday_segformer_valrepairpairs_p50n100c500_b5comp60a5000_20260629 \
  --night-dir work_dirs/submissions/composed/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629 \
  --real-dir work_dirs/submissions/composed/rerun_04085_realvote_verify_20260629_latest \
  --test-root data/test \
  --out-dir work_dirs/submissions/composed/sub_fulldescday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629 \
  --zip work_dirs/submissions/submission_zips/sub_fulldescday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629.zip \
  --overwrite
```

4. 跑 footprint report：

```bash
env PYTHONNOUSERSITE=1 conda run --no-capture-output -n mask2former \
  python tools/explain_submission_delta.py \
  --base work_dirs/submissions/submission_zips/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628.zip \
  --candidate work_dirs/submissions/submission_zips/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629.zip \
  --out work_dirs/submissions/reports/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_explainable_delta_20260629.json \
  --boundary-source either \
  --top-k 30 \
  --component-top-k 40
```

## Score-Free / Val-Positive 與 Hidden Reference

Score-free / val-positive 的部分：

- Night allow-pairs 是用 CoSEC unified Night val 的 M2F base maps 和 SegFormer candidate maps 產生。
- 採用條件是 `min_net=100`、`min_precision=0.50`、`min_changed=500`、`component_min_boundary5_rate=0.60`、`component_max_area=5000`。
- Night pair gate 在 val 上由 M2F Night anchor `56.9114` 提升到 SegFormer repair-pairs p50 `57.7056`。
- Day allow-pairs 是同一套規則套在 CoSEC unified Day val；full-desc M2F Day `81.5921`，SegFormer Day 整份 `68.4472`，但 local repair-pairs p50 合併後到 `81.8267`。
- Day per-class route 沒有任何 class 贏過 M2F，因此只保留 transition repair，不做 class 整份替換。
- 主線 REAL 是 keep 0.4085，不把提交紀錄當成 REAL 替換規則來源。

Hidden submission history reference 只作為風險排序和歷史比較：

- `sub_04085realvote_eventday_nightrswb_transfer4050_negdeny_top2_keepreal_20260629.zip` 回報 `0.4106405259`，但包含 submission-feedback-derived transfer/deny 類規則。
- `no_tsign_no_road_keepreal.zip` 回報 `0.4105348925`，同樣屬於提交紀錄觀察候選。
- 這些 hidden reference 不應回灌成 score-free rule；README 只把它們列作外部比較。

## 候選 Zip / Report / SHA256

| item | path | SHA256 / note |
| --- | --- | --- |
| 0.4085 anchor | `./work_dirs/submissions/submission_zips/sub_reqswin_04075base_seg_swin_m2f_min3_realvote_20260628.zip` | `84fdf54d98d1f4a0350436c644141bbe88c2033745720714b239e1eeead575b2` |
| Night repair intermediate | `./work_dirs/submissions/submission_zips/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_20260629.zip` | `7241c1da7d3f56b14cd5f7a1084ee48212a3de4c648e18ec847b8713bc649ad4` |
| Primary keepreal candidate | `./work_dirs/submissions/submission_zips/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629.zip` | `58f94a40977ad90092fbd2a7789ce984b4fd9abc50da509e1bc55e55e5556dc7` |
| Day full-desc repair intermediate | `./work_dirs/submissions/submission_zips/full_desc_m2fday_segformer_valrepairpairs_p50n100c500_b5comp60a5000_20260629.zip` | `e86e8aac353afc67e1d0ed8c1e2f5c7b5894ba81db55f9f1bc497bef43a39cc6` |
| Full-desc Day val-positive candidate | `./work_dirs/submissions/submission_zips/sub_fulldescday_valpairnight_p50n100c500_b5comp60a5000_keepreal_20260629.zip` | `2604c547a6092b425ec00ea522ccdb568db3fd0aba751a3844e7d7a1b67c6330` |
| REAL gate probe | `./work_dirs/submissions/submission_zips/sub_04085realvote_eventday_valpairnight_p50n100c500_realgate60a5000_20260629.zip` | `c1d67a9b71fc00bc1b30ff0e03a8a9a3e3181b0b6a90752ac8243154895cdeba` |
| REAL fence probe | `./work_dirs/submissions/submission_zips/sub_04085realvote_eventday_valpairnight_p50n100c500_realfence_20260629.zip` | `6cd40a076f400cec04ce7c17929702423e92c5b7bc9f83e0fcd1f5b8830ccb9e` |
| Primary footprint report | `./work_dirs/submissions/reports/sub_04085realvote_eventday_valpairnight_p50n100c500_b5comp60a5000_keepreal_explainable_delta_20260629.json` | Day 2.4384%, Night 1.0288%, REAL 0.0000%, total 1.7958% changed vs anchor |
| Night repair summary | `./work_dirs/submissions/reports/sub_04085realvote_eventday_night_valrepairpairs_p50n100c500_b5comp60a5000_keepreal_summary_20260629.json` | applied pixels 2,357,391 |
| Day repair summary | `./work_dirs/submissions/reports/full_desc_m2fday_segformer_valrepairpairs_p50n100c500_b5comp60a5000_summary_20260629.json` | applied pixels 509,239 |

## Checkpoint 路徑

Checkpoint 皆已複製為 portable bundle 內的 `.pth` 實體檔；原始來源 absolute path 與用途見 `checkpoints_manifest.tsv`。

主要會檢查的 checkpoint family：

- full-desc M2F day/night/acdc: `unified_cosec_acdc/classcover_v1/checkpoints/full_desc_cosec_acdc/mask2former/selected/*.pth`
- event/full-CoSEC M2F Day: `swin_l/work_dirs/swinL_full_cosec_from_day_best_floor816070_lr5e-7_savefix_20260628_225858/best_model_cosec_day.pth`
- full-desc SegFormer day/night/acdc: `unified_cosec_acdc/classcover_v1/checkpoints/full_desc_cosec_acdc/segformer/selected/*.pth`
- event-trained/full-CoSEC SegFormer B5 Night: `swin_l/work_dirs/mmseg/segformer_b5_full_cosec_from_night_best_floor546453_lr1e-6/best_night_mIoU_iter_4500.pth`
- full-desc MaskDINO day/night/acdc: `unified_cosec_acdc/classcover_v1/checkpoints/full_desc_cosec_acdc/maskdino/step1/*.pth`
- MaskDINO kfold3 fold0/1/2 day/night: `unified_cosec_acdc/classcover_v1/checkpoints/maskdino/maskdino_swinl_cosec_dsec_acdc_safe_kfold3_fold{0,1,2}_bs1/best_model_cosec_{day,night}.pth`

## 比較差模型也可能有比較好的 Class

整體 mIoU 較差的模型不代表每個 class 都比較差。SegFormer Night 整體低於 M2F Night，但部分 transition 在 val 上確實修對多於修錯；MaskDINO Day/Night 整體 sanity check 不支持直接替換，也仍可能作為局部 supporter 或 veto。

後續應使用 per-class route、merged val mIoU 或 transition repair gate 驗證單一 class/transition，而不是把所有模型整份平均或投票。這也是目前 Night 使用 repair-pair gate、而非整份 SegFormer Night 的原因。

## Bundle 內容

- `code/`: pipeline 會用到的小型工具實體檔。
- `configs/`: full-desc Mask2Former、SegFormer、MaskDINO 與 MaskDINO kfold3 config 實體檔。
- `artifacts/`: submission zip、report、allow-pairs 實體檔。
- `checkpoints/`: `.pth` checkpoint 實體檔。
- `third_party/`: bundled Mask2Former, detectron2, and the MMSegmentation `mmseg/` package used by the export scripts.
- `tools/`: local helper modules imported by the bundled MMSeg config.
- `artifacts_manifest.tsv`: submission/report/allow-pairs/source/config manifest。
- `checkpoints_manifest.tsv`: checkpoint manifest。
- `file_manifest_sha256.tsv`: all regular files in this portable v2 bundle with SHA256.

目前未發現缺失的已列 artifact；所有 manifest 中 `exists=yes` 的目標在整理時皆存在。整理後 portable bundle 內 symlink 總數為 `0`。

## 0.4111 Checkpoint Rebuild

2026-06-30 補入了 bundle-local 的 0.4111 rebuild path。目標是從 bundle 內 checkpoint/config/code 重新 export masks，再生成：

```text
sub_pipeline_b75_eventseg_plus_realgate60a5000_20260629.zip
```

主要入口：

```text
REBUILD_04111_FROM_CHECKPOINTS.md
code/rebuild_04111_b75_from_bundle_checkpoints.sh
EXPECTED_04111_SHA256.txt
```

新增的 inference/export code：

```text
code/export_mask2former_submission.py
code/export_mmseg_submission.py
```

新增的 local runtime code copies：

```text
third_party/Mask2Former
third_party/detectron2
third_party/mmsegmentation/mmseg
third_party/mmsegmentation/configs
tools/mmseg_unified_metrics.py
tools/mmseg_best_score_floor.py
```

重建命令：

```bash
TEST_ROOT=/path/to/test \
bash code/rebuild_04111_b75_from_bundle_checkpoints.sh
```

除了 `TEST_ROOT`、conda env、以及已安裝 Python runtime packages 之外，checkpoint、config、export script、postprocess script、expected artifacts 都在本 bundle 內。

0.4111 runner 的 CPU smoke test 已在沒有可用 CUDA 的環境下通過：

```text
TEST_ROOT=./data/test
SMOKE_LIMIT=1
DEVICE=cpu
OUT_ROOT=/tmp/repro_bundle_v2_04111_smoke_cpu

Mask2Former event Day checkpoint export: pass, 1 image
Mask2Former full-desc Night checkpoint export: pass, 1 image
SegFormer event Night checkpoint export: pass, 1 image
```

CUDA smoke 第一次因目前 session 無可用 GPU 停在：

```text
RuntimeError: No CUDA GPUs are available
```

這不是路徑或 checkpoint 缺失；CPU smoke 已驗證 bundle-local config/checkpoint/import path 可實際跑 inference。

## Smoke Test

測試 script：

```text
swin_l/tools/smoke_test_repro_bundle.py
code/smoke_test_repro_bundle.py
```

已完成的檢查：

| check | env | result |
| --- | --- | --- |
| all 15 checkpoints `torch.load(map_location="cpu")` | `mmseg` | pass |
| Mask2Former config + model build + copied checkpoint load | `mask2former` | pass |
| SegFormer config + model build + copied checkpoint load | `mmseg` | pass |
| MaskDINO config + model build + copied checkpoint load | `mask2former` | pass |
| v2 event-trained checkpoints `torch.load(map_location="cpu")` | `mmseg` | pass |

重跑命令：

```bash
env PYTHONNOUSERSITE=1 conda run --no-capture-output -n mmseg \
  python swin_l/tools/smoke_test_repro_bundle.py \
  --bundle work_dirs/submissions/repro_bundle_20260629_portable \
  --checks torch-load

env PYTHONNOUSERSITE=1 SKIP_CODE_BACKUP=1 conda run --no-capture-output -n mask2former \
  python swin_l/tools/smoke_test_repro_bundle.py \
  --bundle work_dirs/submissions/repro_bundle_20260629_portable \
  --checks m2f

env PYTHONNOUSERSITE=1 conda run --no-capture-output -n mmseg \
  python swin_l/tools/smoke_test_repro_bundle.py \
  --bundle work_dirs/submissions/repro_bundle_20260629_portable \
  --checks segformer

env PYTHONNOUSERSITE=1 SKIP_CODE_BACKUP=1 conda run --no-capture-output -n mask2former \
  python swin_l/tools/smoke_test_repro_bundle.py \
  --bundle work_dirs/submissions/repro_bundle_20260629_portable \
  --checks maskdino
```

這些 smoke tests 不跑完整 validation；它們只驗證 package 內 copied checkpoint 可讀、且目前 repo/config 能 build model 並載入代表權重。
