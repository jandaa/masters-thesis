python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=s3dis \
    model=minkowski \
    hydra.run.dir=outputs/s3dis/minkowski-2cm \
    tasks=["train","eval"] \
    gpus=1 \
    max_epochs=400 \
    check_val_every_n_epoch=10 \
    model.train.train_workers=10 