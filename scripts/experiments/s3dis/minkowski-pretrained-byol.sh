python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=s3dis \
    model=minkowski \
    model.name=minkowski_byol \
    tasks=["train","eval"] \
    hydra.run.dir=$output_dir/pretrain/byol \
    gpus=[0] \
    dataset.batch_size=6 \
    dataset.accumulate_grad_batches=8 \
    dataset.voxel_size=0.05 \
    model.optimizer.type=SGD \
    model.optimizer.lr=0.2 \
    model.scheduler.type=PolyLR \
    model.scheduler.poly_power=0.9 \
    model.scheduler.max_iter=10000 \
    model.scheduler.interval=step \
    model.scheduler.frequency=10 \
    max_time="01:00:00:00" \
    max_epochs=2000 \
    max_steps=4000 \
    check_val_every_n_epoch=20 \
    model.train.train_workers=6 \
    pretrain_checkpoint=\"last.ckpt\"