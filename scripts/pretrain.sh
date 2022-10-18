# Initalize environment
dataset_dir=/media/starslab/datasets/scannet_preprocessed_2cm_frames_and_scans
source .venv/bin/activate

# Set output folder
output_dir=/media/starslab/users/andrej-janda/outputs/

# Run training
tensorboard --logdir=outputs & \
source $1