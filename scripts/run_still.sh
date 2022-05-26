
# Initalize environment
dataset_dir=/home/andrej/datasets/still
source .venv/bin/activate

# Set output folder
output_dir=/media/starslab/users/andrej-janda/outputs/

# Run experiment
tensorboard --logdir=outputs & \
source $1