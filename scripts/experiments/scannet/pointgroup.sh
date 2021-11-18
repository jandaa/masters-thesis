source .venv/bin/activate
python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/pointgroup \
    gpus=[0]