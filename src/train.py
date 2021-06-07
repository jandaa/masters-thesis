import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

from dataloaders.scannetv2 import ScannetDataModule
from model.pointgroup import PointGroup

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def semantics(cfg: DictConfig) -> None:
    # Print output config for debug purposes
    print(OmegaConf.to_yaml(cfg))

    # Try instanciating and loading Scannet
    scannet = ScannetDataModule(cfg)
    scannet.setup()
    scannet.train_dataloader()
    scannet.val_dataloader()

    model = PointGroup(cfg)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, scannet.train_dataloader(), scannet.val_dataloader())


if __name__ == "__main__":
    semantics()
