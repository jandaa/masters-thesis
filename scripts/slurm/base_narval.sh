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
pip3 install torch==1.10 --no-index
pip3 install  dist/*.tar.gz --no-index
pip3 install  dist/*.whl --no-index

# install each requirement individually incase some are unavailable
cat setup/requirements.dev | xargs -n 1 pip3 install --no-index 
cat setup/requirements.prod | xargs -n 1 pip3 install --no-index

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

# Install spconv
spconv_dir=$base_dir/src/packages/spconv
is_wheel=$(find $spconv_dir -name "*.whl" | wc -l)
if [ $is_wheel -eq 0 ]
then
    cd $spconv_dir
    python setup.py bdist_wheel
fi

cd $base_dir/src/packages/spconv/dist
pip3 install *.whl

# Install pointgroup ops
pointgroup_ops_dir=$base_dir/src/packages/pointgroup_ops
is_wheel=$(find $pointgroup_ops_dir -name "*.whl" | wc -l)
if [ $is_wheel -eq 0 ]
then
    cd $pointgroup_ops_dir
    python setup.py bdist_wheel
fi

cd $pointgroup_ops_dir/dist
pip3 install *.whl

cd $base_dir/src/packages/SensReader
make

cd $base_dir