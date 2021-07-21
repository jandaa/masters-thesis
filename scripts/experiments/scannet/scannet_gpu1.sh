source .venv/bin/activate
python src/train.py \
    dataset_dir=/home/andrej/datasets/scannet \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/single-gpu-v1 \