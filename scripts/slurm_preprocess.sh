#!/bin/bash
#SBATCH --cpus-per-task=40  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=47000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-12:00:00     # DD-HH:MM:SS

# Setup default environment
source scripts/slurm_base.sh

export SCANNET_DIR=~/scratch/datasets/scannet & \
export OUTPUT_DIR=~/scratch/datasets/scannet_preprocessed_2mm & \
source scripts/experiments/scannet/preprocess.sh