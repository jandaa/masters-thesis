python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=s3dis \
    model=minkowski \
    tasks=["train","eval"] \
    hydra.run.dir=outputs/s3dis/minkowski-2cm-pretrained-verify \
    gpus=1 \
    dataset.scale=50 \
    max_epochs=400 \
    check_val_every_n_epoch=10 \
    model.train.train_workers=10 \
    pretrain_checkpoint="partition8_4096_100k.pth"