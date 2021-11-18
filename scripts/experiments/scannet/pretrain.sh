dataset_dir=/media/starslab/datasets/scannet_preprocessed_2mm
python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    tasks=["pretrain"] \
    hydra.run.dir=outputs/scannetv2/pretrained-2cm \
    gpus=[0] \
    dataset.pretrain.batch_size=32 \
    model.structure.m=32 \
    preload_data=False