_base_ = "./SegFormer_B5_FullDSEC_CoSEC_ACDC_Unified.py"

load_from = None
work_dir = (
    "/work/u1621738/ebmv_eccv/eccv_segment/swin_l/work_dirs/mmseg/"
    "segformer_b5_full_dsec_cosec_acdc_unified_2step_lr1e-6"
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=8000, val_interval=500)
optim_wrapper = dict(optimizer=dict(lr=1e-6))
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=300),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=300, end=8000, by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(interval=500),
    logger=dict(interval=50),
)
