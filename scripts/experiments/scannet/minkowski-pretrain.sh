python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    model=minkowski \
    tasks=["pretrain"] \
    hydra.run.dir=outputs/scannetv2/minkowski-pretrained-2cm \
    gpus=[0] \
    dataset.pretrain.batch_size=4 \
    dataset.pretrain.accumulate_grad_batches=1 \
    model.train.train_workers=8 \
    dataset.scale=50 \
    check_val_every_n_epoch=10 
    # pretrain_checkpoint=last.ckpt