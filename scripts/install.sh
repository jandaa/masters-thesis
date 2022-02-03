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
# pip3 install torch==1.10.+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

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

cd $base_dir/src/packages/SensReader
make