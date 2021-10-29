#!/bin/bash

module load python/3.8
module load sparsehash
module load boost/1.72.0
module load cuda/11.1

module load StdEnv/2020  
module load cudacore/.11.1.1
module load cudnn/8.2.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install torch==1.9.1 --no-index
pip3 install  dist/*.tar.gz --no-index
pip3 install  dist/*.whl --no-index

# install each requirement individually incase some are unavailable
cat setup/requirements.dev | xargs -n 1 pip3 install --no-index 
cat setup/requirements.prod | xargs -n 1 pip3 install --no-index

# Run from the root of repository (e.g. sbatch scripts/slurm.sh)
base_dir=$PWD

cd $base_dir/src/packages/spconv
python setup.py bdist_wheel
cd $base_dir/src/packages/spconv/dist
pip3 install *.whl

cd $base_dir/src/packages/pointgroup_ops
python setup.py develop

cd $base_dir/src/packages/SensReader
make

cd $base_dir