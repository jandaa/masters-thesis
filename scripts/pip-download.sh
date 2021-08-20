mkdir dist
pip3 download antlr4-python3-runtime==4.8 -d dist/
pip3 download --no-deps hydra-core omegaconf -d dist/
pip3 download --no-deps torch-metrics pytorch-lightning==1.4.2 -d dist/

