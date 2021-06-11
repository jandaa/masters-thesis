import os
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from util import utils

import pytorch_lightning as pl

from dataloaders.scannetv2 import ScannetDataModule
from model.pointgroup import PointGroup


log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def semantics(cfg: DictConfig) -> None:

    # Print output config for debug purposes
    print(OmegaConf.to_yaml(cfg))

    # # Load from checkpoint if availble
    # if cfg.continue_from:
    #     continue_from_dir = hydra.utils.get_original_cwd() + "/" + cfg.continue_from
    #     utils.load_previous_training(Path(continue_from_dir), Path.cwd())

    log.info("Loading data module")
    # Try instanciating and loading Scannet
    scannet = ScannetDataModule(cfg)
    scannet.setup()
    
    log.info("Loading model")
    model = PointGroup(cfg)
    
    checkpoint_path = None
    if cfg.continue_from:
        # TODO: Write a function to select the latest checkpoint
        checkpoint_path = os.path.join(
            hydra.utils.get_original_cwd(), "checkpoints/epoch=15-step=31.ckpt"
        )

    log.info("Building trainer")
    trainer = pl.Trainer(
        gpus=1, 
        resume_from_checkpoint=checkpoint_path, 
        max_epochs=364,
        check_val_every_n_epoch=10
    )

    # run to test the output
    # trainer.test(model, scannet.test_dataloader())

    # Train model
    log.info("starting training")
    trainer.fit(model, scannet.train_dataloader(), scannet.val_dataloader())


if __name__ == "__main__":
    semantics()
