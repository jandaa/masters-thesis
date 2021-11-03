# Install dependencies
sudo apt-get install \
    libsparsehash-dev \
    libboost-all-dev \
    build-essential \
    python3-dev \
    libopenblas-dev

# Create virtual environment
python -m venv .venv
source ~/.bashrc
source .venv/bin/activate

# Install pytorch with CUDA version 11.1
pip3 install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install all other dependencies
pip3 install -r setup/requirements.dev
pip3 install -r setup/requirements.prod

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

cd $minkowski_dir/dist
pip3 install *.whl

# Install spconv
spconv_dir=$base_dir/src/packages/spconv
is_wheel=$(find $spconv_dir -name "*.whl" | wc -l)
if [ $is_wheel -eq 0 ]
then
    cd $spconv_dir
    export CUDACXX=/usr/local/cuda/bin/nvcc
    python setup.py bdist_wheel
fi

cd $spconv_dir/dist
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