python -m venv .venv
source .venv/bin/activate

pip install -r requirements.dev
pip install -r requirements.prod

base_dir=$PWD

# Install dependencies
sudo apt-get install libsparsehash-dev libboost-all-dev

# Clone spconv
git clone git@github.com:traveller59/spconv.git src/lib/spconv --recursive
cd src/lib/spconv
git checkout fad3000249d27ca918f2655ff73c41f39b0f3127

python setup.py bdist_wheel
cd dist
pip install *.whl

cd $base_dir/src/lib/pointgroup_ops
python setup.py develop