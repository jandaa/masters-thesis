#!/bin/bash
#SBATCH --gres=gpu:p100:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=10  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=20000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=1-00:00:00     # DD-HH:MM:SS

# Setup default environment
source scripts/slurm/base.sh

# Untar data
dataset=~/scratch/datasets/scannet_preprocessed_2cm_fpfh_cluster_no_images.tar
tar -xf $dataset -C $SLURM_TMPDIR

base_dir=$PWD
source $SLURM_TMPDIR/env/bin/activate

# Set output dir
output_dir=base_dir/outputs

# Run training
cd $base_dir
dataset_dir=$SLURM_TMPDIR/scannet_preprocessed_2cm_fpfh_cluster_no_images
tensorboard --logdir=outputs --host 0.0.0.0 --load_fast false & \
source $1