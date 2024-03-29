#!/bin/bash

module load python/3.8
module load sparsehash
module load openblas/0.3.17
module load boost/1.72.0
module load cuda/11.4

module load StdEnv/2020
module load cudnn/8.2.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install torch==1.10 torchvision==0.11.1 --no-index
pip install  dist/*.tar.gz --no-index
pip install  dist/*.whl --no-index

# install each requirement individually incase some are unavailable
cat setup/requirements.slurm | xargs -n 1 pip install --no-index 

# Run from the root of repository (e.g. sbatch scripts/slurm.sh)
base_dir=$PWD

# Install minkowki engine
minkowski_dir=$base_dir/src/packages/MinkowskiEngine 
is_wheel=$(find $minkowski_dir -name "*.whl" | wc -l)
if [ $is_wheel -eq 0 ]
then
    cd $minkowski_dir
    export CXX=c++;
    python setup.py bdist_wheel --blas=openblas --force_cuda
fi

cd $base_dir/src/packages/MinkowskiEngine/dist
pip3 install *.whl

cd $base_dir/src/packages/SensReader
make

cd $base_dir