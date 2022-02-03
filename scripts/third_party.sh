# Clone spconv
# git clone git@github.com:jandaa/spconv.git src/packages/spconv --recursive
# git clone https://github.com/outsidercsy/spconv.git src/packages/spconv --recursive

# Clone Minkowski Engine for SLURM
base_dir=$PWD
git clone  --depth 1 https://github.com/NVIDIA/MinkowskiEngine.git src/packages/MinkowskiEngine
cd src/packages/MinkowskiEngine
git checkout 2f31bc51e0abdf89ed20730e531480df1b2cc64a
# git clone  --depth 1 --branch v0.5.4 https://github.com/NVIDIA/MinkowskiEngine.git src/packages/MinkowskiEngine