python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/semantic_pointgroup \
    tasks=["train","eval"] \
    gpus=1 