
# Initalize environment
dataset_dir=~/datasets/S3DIS_preprocessed
source .venv/bin/activate

# Set output folder
output_dir=/media/starslab/users/andrej-janda/outputs/

# Run experiment
tensorboard --logdir=outputs & \
source $1