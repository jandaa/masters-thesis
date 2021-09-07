source .venv/bin/activate
python src/train.py \
    dataset_dir=/home/andrej/datasets/S3DIS \
    dataset=s3dis \
    hydra.run.dir=outputs/s3dis/debug \
    gpus=[0] \
    preload_data=False \
    dataset.train_split=[1] \
    dataset.val_split=[1] \
    dataset.test_split=[1] \