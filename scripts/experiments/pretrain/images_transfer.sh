python src/main.py \
    tasks=["pretrain"] \
    hydra.run.dir=$output_dir/pretrain/images \
    gpus=[0] \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    dataset.name=scannetv2_pretrain_new \
    dataset.pretrain.batch_size=8 \
    dataset.voxel_size=0.05 \
    model=minkowski \
    model.name=minkowski_cme \
    model.train.train_workers=12 \
    check_val_every_n_epoch=1 \
    dataset.classes=16 \
    pretrain_checkpoint_2d=\"epoch=31-step=18879-val_loss=6.38.ckpt\"