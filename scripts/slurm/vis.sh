#!/bin/bash
#SBATCH --cpus-per-task=10  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=47000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-01:00:00     # DD-HH:MM:SS

module load python/3.8

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --no-index tensorboard

tensorboard --logdir=outputs --host 0.0.0.0 --load_fast false

# to start an interactive session
# salloc --account=def-jskelly --gres=gpu:1 --cpus-per-task=10 --mem=46000M --time=1:00:00
# salloc --account=def-jskelly --gres=gpu:4 --cpus-per-task=40 --mem=186000M --time=1:00:00

# to connect to tensorboard
# ssh -N -L localhost:6006:$1:6006 ajanda@beluga.computecanada.ca

# to tar all event files
# find dir/ -name "events.out*" | tar cvf outputs.tar -T -