python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    model=minkowski \
    tasks=["pretrain"] \
    hydra.run.dir=outputs/scannetv2/minkowski-pretrained-2cm \
    gpus=2 \
    dataset.pretrain.batch_size=16 \
    dataset.pretrain.accumulate_grad_batches=1 \
    model.train.train_workers=1 \
    check_val_every_n_epoch=20