python src/main.py \
    model=minkowski \
    hydra.run.dir=$output_dir/scannetv2/minkowski \
    model.name=minkowski_cme \
    tasks=["train","eval"] \
    gpus=[0] \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    dataset.batch_size=32 \
    dataset.accumulate_grad_batches=2 \
    dataset.voxel_size=0.05 \
    max_time="01:00:00:00" \
    max_epochs=2000 \
    check_val_every_n_epoch=20 \
    model.train.train_workers=10