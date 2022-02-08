python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=s3dis \
    model=minkowski \
    tasks=["train","eval"] \
    hydra.run.dir=$output_dir/pretrain/cme \
    gpus=[0] \
    dataset.batch_size=6 \
    dataset.accumulate_grad_batches=8 \
    dataset.voxel_size=0.05 \
    max_time="02:00:00:00" \
    max_epochs=2000 \
    check_val_every_n_epoch=20 \
    model.train.train_workers=10 \
    pretrain_checkpoint=\"epoch=4-step=5723-val_loss=0.67.ckpt\"
    # model.net.freeze_backbone=True \
