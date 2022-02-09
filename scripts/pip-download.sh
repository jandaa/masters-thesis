module load python/3.8

mkdir dist
pip3 download antlr4-python3-runtime==4.8 -d dist/
pip3 download --no-deps hydra-core==1.1.1 omegaconf==2.1.1 -d dist/
pip3 download --no-deps torchmetrics==0.5.1 pytorch-lightning==1.4.9 -d dist/
pip3 download --no-deps pyDeprecate==0.3.1 -d dist/