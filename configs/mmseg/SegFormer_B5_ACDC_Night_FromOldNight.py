_base_ = "/work/u1621738/ebmv_eccv/mmsegmentation/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py"

crop_size = (512, 1024)
data_root = "/work/u1621738/ebmv_eccv/MambaSeg/data/acdc"
dataset_type = "CityscapesDataset"
split_root = "/work/u1621738/ebmv_eccv/eccv_segment/swin_l/work_dirs/mmseg/acdc_splits"

model = dict(
    backbone=dict(init_cfg=None),
    data_preprocessor=dict(size=crop_size),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(384, 768)),
)

load_from = (
    "/work/u1621738/ebmv_eccv/MambaSeg/log/mmseg/"
    "segformer_b5_cosec_daynight_finetune/best_night_mIoU_iter_8000.pth"
)
work_dir = (
    "/work/u1621738/ebmv_eccv/eccv_segment/swin_l/work_dirs/mmseg/"
    "segformer_b5_acdc_night_from_old_night_lr1e-5"
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
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="rgb_anon", seg_map_path="gt"),
        ann_file=f"{split_root}/night_train.txt",
        img_suffix="_rgb_anon.png",
        seg_map_suffix="_gt_labelTrainIds.png",
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="rgb_anon", seg_map_path="gt"),
        ann_file=f"{split_root}/night_val.txt",
        img_suffix="_rgb_anon.png",
        seg_map_suffix="_gt_labelTrainIds.png",
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

train_cfg = dict(type="IterBasedTrainLoop", max_iters=6000, val_interval=500)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01),
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
        interval=500,
        max_keep_ckpts=2,
        save_best="mIoU",
        rule="greater",
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)
