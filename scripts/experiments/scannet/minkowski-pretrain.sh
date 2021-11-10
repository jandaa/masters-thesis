python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    model=minkowski \
    tasks=["pretrain"] \
    hydra.run.dir=outputs/scannetv2/minkowski-pretrained-2cm \
    gpus=[1] \
    dataset.pretrain.batch_size=1 \
    dataset.pretrain.accumulate_grad_batches=16 \
    model.train.train_workers=8 \
    dataset.scale=20