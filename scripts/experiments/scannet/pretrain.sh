python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    tasks=["pretrain"] \
    hydra.run.dir=outputs/scannetv2/pretrained-2cm \
    gpus=[0] \
    dataset.pretrain.batch_size=16 \
    model.structure.m=32 \
    preload_data=False