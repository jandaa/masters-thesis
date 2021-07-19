#!/bin/bash
#SBATCH --gres=gpu:2       # Request GPU "generic resources"
#SBATCH --cpus-per-task=20  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=1-00:00:00     # DD-HH:MM:SS

module load python/3.8
module load sparsehash
module load boost/1.72.0
module load cuda/11.1

module load StdEnv/2020  
module load cudacore/.11.1.1
module load cudnn/8.2.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install torch --no-index
pip3 install  dist/*.tar.gz --no-index
pip3 install  dist/*.whl --no-index

# install each requirement individually incase some are unavailable
cat requirements.dev | xargs -n 1 pip3 install --no-index 
cat requirements.prod | xargs -n 1 pip3 install --no-index

# Run from the root of repository (e.g. sbatch scripts/slurm.sh)
base_dir=$PWD

cd $base_dir/src/packages/spconv
python setup.py bdist_wheel
cd $base_dir/src/packages/spconv/dist
pip3 install *.whl

cd $base_dir/src/packages/pointgroup_ops
python setup.py develop

# Untar data
dataset=~/projects/def-jskelly/ajanda/datasets/scannetv2.tar
tar -xf $dataset -C $SLURM_TMPDIR

# Train model
cd $base_dir
dataset_dir=$SLURM_TMPDIR/scannet
python src/train.py \
    dataset_dir=$dataset_dir \
    dataset=scannet \
    hydra.run.dir=outputs/scannetv2/single-gpu-v1 \
   
   