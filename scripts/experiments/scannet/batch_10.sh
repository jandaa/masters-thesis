source .venv/bin/activate
python src/train.py \
    dataset_dir=/home/andrej/datasets/scannet \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/single-gpu-batch-10 \
    gpus=[1] \
    dataset.batch_size=10 \
    checkpoint=last.ckpt