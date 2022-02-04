python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    model=minkowski \
    tasks=["train","eval"] \
    hydra.run.dir=$output_dir/scannetv2/minkowski-2cm-pretrained \
    gpus=1 \
    max_epochs=180 \
    check_val_every_n_epoch=10 \
    pretrain_checkpoint=\"epoch=16-step=322-val_loss=8.61.ckpt\"