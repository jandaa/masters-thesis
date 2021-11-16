python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    model=minkowski \
    hydra.run.dir=outputs/scannetv2/minkowski-2cm \
    tasks=["train","eval"] \
    gpus=1 \
    max_epochs=180 \
    dataset.scale=20 \
    check_val_every_n_epoch=10 \