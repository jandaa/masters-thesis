source .venv/bin/activate
python src/train.py \
    dataset_dir=/home/andrej/datasets/S3DIS \
    dataset=s3dis \
    hydra.run.dir=outputs/s3dis/single-gpu-v1 \