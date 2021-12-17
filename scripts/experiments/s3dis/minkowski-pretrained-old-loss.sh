python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=s3dis \
    model=minkowski \
    tasks=["train","eval"] \
    hydra.run.dir=outputs/pretrain/5cm \
    gpus=1 \
    dataset.batch_size=6 \
    dataset.voxel_size=0.05 \
    model.optimizer.type=SGD \
    model.optimizer.lr=0.1 \
    model.scheduler.type=PolyLR \
    model.scheduler.poly_power=0.9 \
    model.scheduler.max_iter=20000 \
    model.scheduler.interval=step \
    model.scheduler.frequency=100 \
    max_time="00:12:00:00" \
    max_epochs=400 \
    check_val_every_n_epoch=10 \
    model.train.train_workers=10 \
    pretrain_checkpoint=\"epoch=270-step=19999-val_loss=6.47.ckpt\"