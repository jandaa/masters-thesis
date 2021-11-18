source .venv/bin/activate
python src/main.py \
    dataset_dir=$dataset_dir \
    dataset=s3dis \
    hydra.run.dir=outputs/s3dis/pointgroup \
    gpus=[0]