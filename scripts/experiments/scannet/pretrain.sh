source .venv/bin/activate
python src/train.py \
    dataset_dir=/media/starslab/datasets/scannet \
    dataset=scannet \
    tasks=["train","eval"] \
    hydra.run.dir=outputs/scannetv2/pretrained \
    gpus=[0] \
    preload_data=False \
    dataset.pretrain.batch_size=2 \
    model.train.train_workers=8 \
    pretrain_checkpoint=last.ckpt
    # limit_train_batches=0.01 