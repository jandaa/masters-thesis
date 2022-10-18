python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    dataset.name=scannetv2_pretrain_new \
    model=minkowski \
    model.name=minkowski_cme \
    tasks=["pretrain"] \
    hydra.run.dir=$output_dir/pretrain/cme \
    gpus=[0] \
    dataset.pretrain.batch_size=32 \
    check_val_every_n_epoch=-1 \
    val_check_interval=1000