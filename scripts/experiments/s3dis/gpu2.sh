source .venv/bin/activate
python src/train.py \
    dataset_dir=/home/andrej/datasets/S3DIS \
    dataset=s3dis \
    dataset.batch_size=2
    hydra.run.dir=outputs/s3dis/gpu-2-v1 \
    preload_data=True \