source .venv/bin/activate
python src/train.py \
    dataset_dir=/home/andrej/datasets/scannet \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/debug \
    gpus=[0] \
    preload_data=False \
    dataset.train_split_file=scannetv2-val-debug.txt \
    dataset.val_split_file=scannetv2-val-debug.txt \
    dataset.test_split_file=scannetv2-val-debug.txt \