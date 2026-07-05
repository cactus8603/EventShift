_base_ = "/work/u1621738/ebmv_eccv/mmsegmentation/configs/ocrnet/ocrnet_hr48_4xb2-160k_cityscapes-512x1024.py"

custom_imports = dict(imports=["tools.mmseg_cosec_metrics"], allow_failed_imports=False)

crop_size = (512, 1024)
data_root = "/work/u1621738/ebmv_eccv/MambaSeg/data/cosec_mmseg"
dataset_type = "CityscapesDataset"

model = dict(
    pretrained=None,
    data_preprocessor=dict(size=crop_size),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(384, 768)),
)

load_from = (
    "/work/u1621738/ebmv_eccv/eccv_segment/swin_l/work_dirs/pretrained/"
    "ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth"
)
work_dir = (
    "/work/u1621738/ebmv_eccv/eccv_segment/swin_l/work_dirs/mmseg/"
    "hrnet_ocr_w48_cosec_from_cityscapes_lr5e-4"
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
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images", seg_map_path="labels"),
        ann_file="splits/train.txt",
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
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images", seg_map_path="labels"),
        ann_file="splits/val.txt",
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="CoSECDayNightIoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

train_cfg = dict(type="IterBasedTrainLoop", max_iters=8000, val_interval=500)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    optimizer=dict(type="SGD", lr=5e-4, momentum=0.9, weight_decay=0.0005),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=300),
    dict(type="PolyLR", eta_min=0.0, power=0.9, begin=300, end=8000, by_epoch=False),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=500,
        max_keep_ckpts=2,
        save_best=["day_mIoU", "night_mIoU"],
        rule=["greater", "greater"],
    ),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)
