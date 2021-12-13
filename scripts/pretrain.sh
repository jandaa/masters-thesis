# Initalize environment
dataset_dir=/media/starslab/datasets/scannet_preprocessed_2cm
source .venv/bin/activate

# Run training
tensorboard --logdir=outputs & \
source $1