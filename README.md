# aribic-semantics
Instance segmentation algorithms built for the ARIBIC project

### Requirements

- Ubuntu 18.04+
- Python 3.6+
- g++ 9.3.0

## Getting Started

Follow these steps chronologically to start using this project. For those running on SLURM clusters, skip ahead to the SLURM section.

### Python

Make sure that you have a compatable python version. Install using:

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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib
```

and reloading with:

```shell
source ~/.bashrc
```

### Installation

Run the following commands to install a virtual entironment with all the requirements. Note this is not necessary if running on a SLURM cluster (refer to the next section).

```shell
bash scripts/third_party.sh
sudo bash scripts/install.sh
```

### SLURM Compute Clusters

This step is meant onlt for running on SLURM clusters.