source .venv/bin/activate
python src/train.py \
    dataset_dir=/media/starslab/datasets/scannet \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/preload \
    gpus=[0] \
    tasks=['pretrain'] \
    preload_data=False \
    dataset.batch_size=2 \
    model.train.train_workers=4 \