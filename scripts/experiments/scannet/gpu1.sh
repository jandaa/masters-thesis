source .venv/bin/activate
python src/train.py \
    dataset_dir=/home/andrej/datasets/scannet \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/single-gpu-v2 \
    gpus=[0] \
    preload_data=False \
    checkpoint=last.ckpt \