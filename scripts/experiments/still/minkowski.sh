python src/main.py \
    model=minkowski \
    hydra.run.dir=$output_dir/still/minkowski \
    model.name=minkowski_cme \
    tasks=["eval", "visualize"] \
    gpus=[0] \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    dataset.batch_size=1 \
    dataset.accumulate_grad_batches=2 \
    dataset.voxel_size=0.05 \
    max_time="01:00:00:00" \
    max_epochs=2000 \
    check_val_every_n_epoch=20 \
    model.train.train_workers=10