python src/main.py \
    tasks=["pretrain"] \
    hydra.run.dir=$output_dir/pretrain/images \
    gpus=[0] \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    dataset.name=scannetv2_pretrain_new \
    dataset.pretrain.batch_size=64 \
    model=minkowski \
    model.name=image_pretrain \
    model.train.train_workers=12 \
    model.pretrain.optimizer.lr=0.01 \
    check_val_every_n_epoch=2 \
    model.net.warmup_steps=50 \
    pretrain_checkpoint=\"epoch=11-step=7079-val_loss=6.42.ckpt\"