import os
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from util import utils

import pytorch_lightning as pl

from dataloaders.scannetv2 import ScannetDataModule
from model.pointgroup import PointGroupWrapper


log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def semantics(cfg: DictConfig) -> None:

    log.info("Loading data module")
    scannet = ScannetDataModule(cfg)
    scannet.setup()

    log.info("Loading model")
    model = PointGroupWrapper(cfg)

    checkpoint_path = None
    log.info("Building trainer")
    trainer = pl.Trainer(
        gpus=1,
        accelerator="ddp",
        resume_from_checkpoint=checkpoint_path,
        max_epochs=1,
        check_val_every_n_epoch=10,
    )

    # Train model
    log.info("starting training")
    trainer.fit(model, scannet.train_dataloader(), scannet.val_dataloader())

    # run to test the output
    trainer.test(model, scannet.test_dataloader())

    # Visualize results
    Visualizer = utils.Visualizer(Path.cwd() / "predictions")
    Visualizer.visualize_results()


if __name__ == "__main__":
    semantics()
