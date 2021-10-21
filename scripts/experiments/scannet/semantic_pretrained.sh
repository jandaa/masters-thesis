python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/pretrained \
    tasks=["train","eval"] \
    preload_data=False \
    gpus=1 \
    model.train.train_workers=10 \
    pretrain_checkpoint=\"epoch=119-step=2159-val_loss=8.57.ckpt\"