_base_ = "/work/u1621738/ebmv_eccv/mmsegmentation/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py"

custom_imports = dict(
    imports=["tools.mmseg_unified_metrics", "tools.mmseg_best_score_floor"],
    allow_failed_imports=False,
)

crop_size = (512, 1024)
dataset_type = "CityscapesDataset"
cosec_root = "/work/u1621738/ebmv_eccv/MambaSeg/data/cosec_mmseg"
unified_root = "/work/u1621738/ebmv_eccv/eccv_segment/unified_cosec_acdc/classcover_v1"

model = dict(
    backbone=dict(init_cfg=None),
    data_preprocessor=dict(size=crop_size),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(384, 768)),
)

load_from = (
    "/work/u1621738/ebmv_eccv/eccv_segment/unified_cosec_acdc/classcover_v1/"
    "checkpoints/full_desc_cosec_acdc/segformer/selected/best_night_mIoU.pth"
)
work_dir = (
    "/work/u1621738/ebmv_eccv/eccv_segment/swin_l/work_dirs/mmseg/"
    "segformer_b5_full_cosec_from_night_best_floor546453_lr1e-6"
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

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=cosec_root,
        data_prefix=dict(img_path="images", seg_map_path="labels"),
        ann_file=f"{unified_root}/splits/cosec/train_unified_classcover_v1.txt",
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=cosec_root,
        data_prefix=dict(img_path="images", seg_map_path="labels"),
        ann_file=f"{unified_root}/splits/cosec/val_unified_classcover_v1.txt",
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="DomainSplitIoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

train_cfg = dict(type="IterBasedTrainLoop", max_iters=6000, val_interval=500)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=300),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=300, end=6000, by_epoch=False),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=100000000,
        save_last=False,
        max_keep_ckpts=1,
        save_best="night_mIoU",
        rule="greater",
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)
custom_hooks = [
    dict(
        type="BestScoreFloorHook",
        floors=dict(night_mIoU=54.6453),
        primary_metric="night_mIoU",
    )
]
