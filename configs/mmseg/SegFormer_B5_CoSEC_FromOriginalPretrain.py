_base_ = "./configs/SegFormer_B5_CoSEC_DayNight_Finetune.py"

# Fresh SegFormer-B5 CoSEC run from the originally downloaded Cityscapes
# checkpoint. This intentionally does not load any previous CoSEC best_day /
# best_night checkpoint.
data_root = "./data/cosec_mmseg"
load_from = None
work_dir = (
    "./work_dirs/mmseg/"
    "segformer_b5_cosec_from_original_pretrain_lr2e-5"
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(data_root=data_root),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(data_root=data_root),
)
test_dataloader = val_dataloader

train_cfg = dict(type="IterBasedTrainLoop", max_iters=12000, val_interval=500)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=500, end=12000, by_epoch=False),
]

default_hooks = dict(
    checkpoint=dict(
        interval=500,
        max_keep_ckpts=2,
        save_best=["day_mIoU", "night_mIoU"],
        rule=["greater", "greater"],
    ),
    logger=dict(interval=50),
)
