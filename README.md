# aribic-semantics
Instance segmentation algorithms built for the ARIBIC project

### Requirements

- Ubuntu 18.04+
- Python 3.6+
- gcc 9.3.0+

## Getting Started

Please refer to instructions under the ``setup`` folder.

## Downloading Datasets
Please refer to instructions under the ``datasets`` folder.

## Training

To train a model, use one of the appropriate scripts located under the ``scripts/experiments/`` directory. Update the path to the dataset root directory and run the stript from the command line. 

## Evaluation

Evaluating a trained model can be done from the command line, with a few command line arguments that specify the dataset directory, the run directory and the checkpoint to use. This could look like the following:

```shell
python train.py \
    tasks=['eval','visualize'] \
    dataset=scannet \
    dataset_dir=/path/to/dataset/ \
    hydra.run.dir=outputs/scannet/run_dir/ \
    checkpoint=last.ckpt
```

## Contributing

### Formatting

For formatting python code, we use ``black``. The best practice is to setup black to run every time you save. This can be done through the settings of your specific editor or can be run from the command line using:

```shell
black src/
```

### Pre-Commit Hooks

Before any commit is pushed through, it should first be run using the pre-commit package. This is automatically install in the virtual environment but can also be installed separately using:

```shell
pip install pre-commit
pre-commit install
```
