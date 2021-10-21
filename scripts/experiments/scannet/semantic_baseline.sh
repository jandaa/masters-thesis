python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/baseline \
    tasks=["train","eval"] \
    preload_data=False \
    gpus=1 \
    model.train.train_workers=10