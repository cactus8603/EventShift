_base_ = "./HRNet_OCR_W48_ACDC_Night_Finetune.py"

split_root = "./work_dirs/mmseg/acdc_splits"

work_dir = (
    "./work_dirs/mmseg/"
    "hrnet_ocr_w48_acdc_all_from_cityscapes_lr5e-4"
)

train_dataloader = dict(
    dataset=dict(
        ann_file=f"{split_root}/all_train.txt",
    ),
)
val_dataloader = dict(
    dataset=dict(
        ann_file=f"{split_root}/all_val.txt",
    ),
)
test_dataloader = val_dataloader
