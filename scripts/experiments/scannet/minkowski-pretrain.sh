python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    model=minkowski \
    tasks=["pretrain"] \
    hydra.run.dir=outputs/scannetv2/minkowski-pretrained-2cm \
    gpus=[0,1] \
    dataset.pretrain.batch_size=16 \
    dataset.pretrain.accumulate_grad_batches=2 \
    model.train.train_workers=20 \
    dataset.scale=50 \
    check_val_every_n_epoch=10 \
    dataset.pretrain.max_epochs=500
    # pretrain_checkpoint=last.ckpt