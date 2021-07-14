python -m venv .venv
source .venv/bin/activate

# Install pytorch with CUDA version 11.1
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install all other dependencies
pip3 install -r requirements.dev
pip3 install -r requirements.prod

base_dir=$PWD

# Install dependencies
sudo apt-get install libsparsehash-dev libboost-all-dev

# Load in third part source code
/bin/bash src/thrid_party.sh

cd $base_dir/src/packages/spconv
python setup.py bdist_wheel
cd dist
pip3 install *.whl

cd $base_dir/src/packages/pointgroup_ops
python setup.py develop