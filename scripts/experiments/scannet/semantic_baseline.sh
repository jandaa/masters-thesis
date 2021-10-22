python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/baseline-deterministic \
    tasks=["train","eval"] \
    preload_data=False \
    gpus=1 \
    max_epochs=120 \
    model.train.train_workers=10