import os

MMSEGMENTATION_ROOT = os.getenv("MMSEGMENTATION_ROOT", "third_party/mmsegmentation")
_base_ = f"{MMSEGMENTATION_ROOT}/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py"

custom_imports = dict(imports=["tools.mmseg_unified_metrics"], allow_failed_imports=False)

crop_size = (512, 1024)
dataset_type = "CityscapesDataset"
cosec_root = os.getenv("COSEC_MMSEG_ROOT", "data/cosec_mmseg")
dsec_root = os.getenv("DSEC_MMSEG_ROOT", "work_dirs/mmseg/dsec19_full_flat")
acdc_root = os.getenv("ACDC_ROOT", "data/acdc")
unified_root = os.getenv("EVENTSHIFT_UNIFIED_ROOT", "work_dirs/unified_cosec_acdc/classcover_v1")

model = dict(
    backbone=dict(init_cfg=None),
    data_preprocessor=dict(size=crop_size),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(384, 768)),
)

load_from = None
work_dir = (
    os.getenv("WORK_DIR", "work_dirs/mmseg/segformer_b5_full_dsec_cosec_acdc_unified")
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(1200, 624), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1200, 624), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

cosec_train = dict(
    type=dataset_type,
    data_root=cosec_root,
    data_prefix=dict(img_path="images", seg_map_path="labels"),
    ann_file=f"{unified_root}/splits/cosec/train_unified_classcover_v1.txt",
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=train_pipeline,
)
dsec_train = dict(
    type=dataset_type,
    data_root=dsec_root,
    data_prefix=dict(img_path="images", seg_map_path="labels"),
    ann_file=f"{dsec_root}/splits/train_full.txt",
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=train_pipeline,
)
acdc_train = dict(
    type=dataset_type,
    data_root=acdc_root,
    data_prefix=dict(img_path="rgb_anon", seg_map_path="gt"),
    ann_file=f"{unified_root}/splits/acdc/train_unified_classcover_v1.txt",
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=train_pipeline,
)

cosec_val = dict(
    type=dataset_type,
    data_root=cosec_root,
    data_prefix=dict(img_path="images", seg_map_path="labels"),
    ann_file=f"{unified_root}/splits/cosec/val_unified_classcover_v1.txt",
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=test_pipeline,
)
acdc_val = dict(
    type=dataset_type,
    data_root=acdc_root,
    data_prefix=dict(img_path="rgb_anon", seg_map_path="gt"),
    ann_file=f"{unified_root}/splits/acdc/val_unified_classcover_v1.txt",
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=test_pipeline,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(_delete_=True, type="ConcatDataset", datasets=[cosec_train, dsec_train, acdc_train]),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(_delete_=True, type="ConcatDataset", datasets=[cosec_val, acdc_val]),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="DomainSplitIoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

train_cfg = dict(type="IterBasedTrainLoop", max_iters=24000, val_interval=1000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=5e-6, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=500, end=24000, by_epoch=False),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=2,
        save_best=["day_mIoU", "night_mIoU", "acdc_mIoU"],
        rule=["greater", "greater", "greater"],
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)
