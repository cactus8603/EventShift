_base_ = "/work/u1621738/ebmv_eccv/MambaSeg/configs/SegFormer_B5_CoSEC_DayNight_Finetune.py"

data_root = "/work/u1621738/ebmv_eccv/MambaSeg/data/cosec_mmseg"

load_from = (
    "/work/u1621738/ebmv_eccv/MambaSeg/log/mmseg/"
    "segformer_b5_cosec_daynight_finetune/best_day_mIoU_iter_3000.pth"
)
work_dir = (
    "/work/u1621738/ebmv_eccv/eccv_segment/swin_l/work_dirs/mmseg/"
    "segformer_b5_cosec_continue_from_old_day_lr1e-5"
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

train_cfg = dict(type="IterBasedTrainLoop", max_iters=8000, val_interval=500)

optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=300),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=300, end=8000, by_epoch=False),
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
