# Initalize environment
dataset_dir=/media/starslab/datasets/scannet_preprocessed_2cm_fpfh_cluster_no_images
source .venv/bin/activate

# Set output folder
output_dir=/media/starslab/users/andrej-janda/outputs/

# Run training
tensorboard --logdir=outputs & \
source $1