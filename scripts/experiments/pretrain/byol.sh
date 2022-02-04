python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    dataset.name=scannetv2_pretrain \
    model=minkowski \
    model.name=minkowski_byol \
    tasks=["pretrain"] \
    hydra.run.dir=$output_dir/pretrain/byol \
    gpus=[0]