python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    model=minkowski \
    model.name=pointcontrast \
    dataset.name=scannetv2_pretrain_new \
    tasks=["pretrain"] \
    hydra.run.dir=$output_dir/pretrain/old-loss-scan-only \
    gpus=[0] \
    max_time="01:00:00:00" \
    dataset.pretrain.batch_size=16 \
    dataset.pretrain.accumulate_grad_batches=4 \
    model.train.train_workers=8 \
    check_val_every_n_epoch=-1 \
    model.optimizer.type=SGD \
    model.optimizer.lr=0.1 \
    val_check_interval=1000 
    # pretrain_checkpoint=\"epoch=5-step=6364-val_loss=6.14.ckpt\"