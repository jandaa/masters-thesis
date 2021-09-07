## Getting Started

Follow these steps chronologically to get started using this project. For those running on SLURM clusters, skip ahead to the SLURM section.

### Python

Make sure that you have a compatable python version. Install using

```shell
sudo apt-get install python3
```
OR 
```shell
sudo apt-get install python-is-python3
```

if you want the python command to automatically use python3. Note that the python command should point to python3. If it does not use the command ``update-alternatives`` to set python to python3.

Then install pip and venv to install packages in a seperate virtual environment.

```shell
sudo apt-get install python3-pip python3-venv
```

### Nvidia Drivers

This project requires both nvidia [cuDNN](https://developer.nvidia.com/cudnn) and the [cuda toolkit](https://developer.nvidia.com/cuda-downloads) to be installed.

When installing the cuDNN library, download both the runtime and developer library and install each using the following commands.

```shell
sudo dpkg -i libcudnn8.deb
sudo dpkg -i libcudnn8-dev.deb
```

remember to restart your computer after completing this installation. After installing, set the LD_LIBRARY_PATH to point to the cuda library by adding this line to your .bashrc

```bash
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

and reloading with

```shell
source ~/.bashrc
```

### Installation

Run the following commands to install a virtual entironment with all the requirements. Note this is not necessary if running on a SLURM cluster (refer to the next section).

```shell
bash scripts/third_party.sh
sudo bash scripts/install.sh
```

The virtual environment can be activated using:

```shell
source .venv/bin/activate
```

### SLURM Compute Clusters

This step is meant onlt for running on SLURM clusters. Most of the installation is already in the slurm script, all that is required is to pre-download any packages that may not be available on the system. This can be down using the following commands:

```shell
bash scripts/third_party.sh
bash scripts/pip-download.sh
```

and then submitting the desired job with


```shell
sbatch scripts/slurm/slurm_*.sh
```