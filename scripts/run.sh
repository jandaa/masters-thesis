
# Initalize environment
dataset_dir=~/datasets/S3DIS_preprocessed
source .venv/bin/activate

# Run experiment
tensorboard --logdir=outputs & \
source $1