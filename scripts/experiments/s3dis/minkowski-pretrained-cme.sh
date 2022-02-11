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
    pretrain_checkpoint=\"epoch=33-step=39465-val_loss=0.63.ckpt\"
