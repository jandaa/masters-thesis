python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/pretrained\
    tasks=["train","eval"] \
    gpus=1 \
    max_epochs=120 \
    check_val_every_n_epoch=10 \
    pretrain_checkpoint=\"epoch=119-step=2159-val_loss=8.57.ckpt\"