python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=s3dis \
    model=minkowski \
    tasks=["train","eval"] \
    hydra.run.dir=outputs/pretrain/sampling-other-scenes \
    gpus=1 \
    max_time="00:12:00:00" \
    max_epochs=400 \
    check_val_every_n_epoch=10 \
    model.train.train_workers=10 \
    pretrain_checkpoint=\"last.ckpt\"