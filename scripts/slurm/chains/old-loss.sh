#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=10  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=47000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=1-00:03:00     # DD-HH:MM:SS

bash scripts/slurm/pretrain.sh scripts/experiments/pretrain/old_loss.sh
bash scripts/slurm/run.sh scripts/experiments/s3dis/minkowski-pretrained-old-loss.sh