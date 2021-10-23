python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/baseline \
    tasks=["train","eval"] \
    preload_data=False \
    gpus=1 \
    max_epochs=120 \
    check_val_every_n_epoch=10