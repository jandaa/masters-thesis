python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=s3dis \
    model=minkowski \
    tasks=["train","eval"] \
    hydra.run.dir=outputs/s3dis/minkowski-5cm-pretrained-supervised \
    gpus=1 \
    dataset.scale=20 \
    max_epochs=180 \
    check_val_every_n_epoch=10 \
    model.train.train_workers=10 \
    supervised_pretrain_checkpoint=last.ckpt