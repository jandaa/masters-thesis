
# Initalize environment
dataset_dir=/media/starslab/datasets/scannet_material_properties
source .venv/bin/activate

# Set output folder
output_dir=/media/starslab/users/andrej-janda/outputs/

# Run experiment
tensorboard --logdir=outputs & \
source $1