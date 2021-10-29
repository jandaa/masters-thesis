
# Initalize environment
dataset_dir=~/datasets/scannet/
source .venv/bin/activate

# Run experiment
tensorboard --logdir=outputs & \
source $1