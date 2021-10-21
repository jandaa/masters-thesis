python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/debug \
    gpus=[0] \
    preload_data=False \
    model.train.train_workers=10