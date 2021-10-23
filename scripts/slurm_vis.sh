#!/bin/bash
#SBATCH --cpus-per-task=10  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=47000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-01:00:00     # DD-HH:MM:SS

module load python/3.8

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --no-index tensorboard

# # install each requirement individually incase some are unavailable
# cat setup/requirements.dev | xargs -n 1 pip3 install --no-index 
# cat setup/requirements.prod | xargs -n 1 pip3 install --no-index

tensorboard --logdir=outputs --host 0.0.0.0 --load_fast false

# to connect to tensorboard
# ssh -N -L localhost:6006:$1:$2 ajanda@beluga.computecanada.ca