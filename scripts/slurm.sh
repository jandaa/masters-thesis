#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=2-00:00:00     # DD-HH:MM:SS

module load python/3.7
module load sparsehash
module load boost/1.72.0
module load cuda/11.1

module load StdEnv/2020  
module load cudacore/.11.1.1
module load cudnn/8.2.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --no-index -r requirements.dev
pip3 install --no-index -r requirements.prod
pip3 install --no-index torch

# Run from the root of repository (e.g. sbatch scripts/slurm.sh)
base_dir=$PWD

cd $base_dir/src/lib/spconv
python setup.py bdist_wheel
cd $base_dir/src/lib/spconv/dist
pip3 install *.whl

cd $base_dir/lib/pointgroup_ops
python setup.py develop

# copy over files
cp -r $base_dir/dataset/ $SLURM_TMPDIR

scannet_dir=~/projects/def-jskelly/ajanda/scannet/
tar -xf $scannet_dir/train.tar -C $SLURM_TMPDIR/dataset/scannetv2/
tar -xf $scannet_dir/val.tar -C $SLURM_TMPDIR/dataset/scannetv2/
cp $scannet_dir/scannetv2-labels.combined.tsv $SLURM_TMPDIR/dataset/scannetv2/

cd $SLURM_TMPDIR/dataset/scannetv2/
python $SLURM_TMPDIR/dataset/scannetv2/prepare_data_inst.py --data_split train
python $SLURM_TMPDIR/dataset/scannetv2/prepare_data_inst.py --data_split val
python $SLURM_TMPDIR/dataset/scannetv2/prepare_data_inst_gt.py --data_split val

# Train model
cd $base_dir
python src/train.py dataset_dir=~/projects/def-jskelly/ajanda/scannet/ batch_size=4