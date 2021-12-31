python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=s3dis \
    model=minkowski \
    tasks=["train","eval"] \
    hydra.run.dir=outputs/pretrain/5cm-extra-slow \
    gpus=1 \
    dataset.batch_size=16 \
    dataset.accumulate_grad_batches=3 \
    dataset.voxel_size=0.05 \
    model.optimizer.type=SGD \
    model.optimizer.lr=0.2 \
    model.scheduler.type=PolyLR \
    model.scheduler.poly_power=0.9 \
    model.scheduler.max_iter=20000 \
    model.scheduler.interval=step \
    model.scheduler.frequency=10 \
    max_time="02:00:00:00" \
    max_epochs=2000 \
    check_val_every_n_epoch=10 \
    model.train.train_workers=10 \
    pretrain_checkpoint=\"epoch=270-step=19999-val_loss=6.47.ckpt\"