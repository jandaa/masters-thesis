# Self-Supervised Pre-training of 3D Point Cloud Networks with Image Data

This repository implements a framework for pre-training neural networks that operate on 3D point clouds using images corresponding to the same scene. The work is part of my master's [thesis](https://starslab.ca/wp-content/papercite-data/pdf/2022_janda_contrastive.pdf).

## Getting Started

Please refer to the instructions under the ``setup`` folder.

## Downloading Datasets
Please refer to the instructions under the ``datasets`` folder.

## Preprocessing Datasets
Scripts for preprocessing the datasets into a format usable by this project are provided under the ``scripts/preprocess`` directory.
## Training

To train a model, use one of the appropriate scripts located under the ``scripts/experiments/`` directory. Update the path to the dataset root directory and run the desired script from the command line. Training can be performed in multiple stages, first training a 2D model with the script ``scripts/experiments/pretrain/images.sh``, then transferring the learned features to the 3D model with ``scripts/experiments/pretrain/images_transfer.sh`` and finally fine-tuning with either ``scripts/experiments/scannet/minkovski.sh`` for semantic segmentation on ScanNet or ``scripts/experiments/s3dis/minkovski.sh`` for semantic segmenation on S3DIS.

## Evaluation

Evaluating a trained model can be done from the command line, with a few command-line arguments that specify the dataset directory, the run directory and the checkpoint to use. This could look like the following:

```shell
python main.py \
    tasks=['eval','visualize'] \
    dataset=scannet \
    dataset_dir=/path/to/dataset/ \
    hydra.run.dir=outputs/scannet/run_dir/ \
    checkpoint=last.ckpt
```