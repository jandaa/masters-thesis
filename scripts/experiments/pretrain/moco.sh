python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    dataset.name=scannetv2_pretrain \
    model=minkowski \
    model.name=minkowski_moco \
    tasks=["pretrain"] \
    hydra.run.dir=outputs/pretrain/moco \
    gpus=[1] \
    max_time="01:00:00:00" \
    dataset.pretrain.batch_size=4 \
    dataset.pretrain.accumulate_grad_batches=4 \
    model.train.train_workers=8 \
    check_val_every_n_epoch=10 \
    model.optimizer.type=SGD \
    model.optimizer.lr=0.1