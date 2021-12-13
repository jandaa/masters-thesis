python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    model=minkowski \
    tasks=["pretrain"] \
    hydra.run.dir=outputs/pretrain/sampling-other-scenes \
    gpus=1 \
    max_time="00:12:00:00" \
    dataset.pretrain.batch_size=8 \
    dataset.pretrain.accumulate_grad_batches=2 \
    model.train.train_workers=8 \
    check_val_every_n_epoch=10 \
    model.optimizer.type=SGD \
    model.optimizer.lr=0.1