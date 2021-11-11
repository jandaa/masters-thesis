python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=s3dis \
    model=minkowski \
    tasks=["train","eval"] \
    hydra.run.dir=outputs/s3dis/minkowski-5cm-pretrained \
    gpus=1 \
    dataset.scale=20 \
    max_epochs=180 \
    check_val_every_n_epoch=10 \
    model.train.train_workers=10 \
    pretrain_checkpoint=\"epoch=80-step=6074-val_loss=210682.38.ckpt\"