python src/main.py \
    model=minkowski \
    model.name=minkowski_cme \
    hydra.run.dir=$output_dir/scannetv2/frames \
    tasks=["train","eval"] \
    gpus=[0] \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    dataset.name=scannetv2_pretrain_new \
    dataset.batch_size=32 \
    dataset.accumulate_grad_batches=2 \
    dataset.voxel_size=0.05 \
    limit_val_batches=0.05 \
    limit_test_batches=0.2 \
    max_time="01:00:00:00" \
    max_epochs=2000 \
    check_val_every_n_epoch=2 \
    model.train.train_workers=10