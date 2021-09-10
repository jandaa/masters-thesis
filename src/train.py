import logging
from pathlib import Path
from dateutil.parser import parse

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import random
import torch

from util import utils
from dataloaders.dataloader import DataModule
from dataloaders.data_interface import DataInterfaceFactory
from model.pointgroup import PointGroupWrapper, PointGroupBackboneWrapper


log = logging.getLogger("train")


def get_checkpoint_callback():
    return ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
    )


def set_random_seeds(cfg):
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)


@hydra.main(config_path="config", config_name="config")
def semantics(cfg: DictConfig) -> None:

    # Set random seeds for reproductability
    # set_random_seeds(cfg)

    # Load a checkpoint if given
    checkpoint_path = None
    if cfg.checkpoint:
        checkpoint_path = str(Path.cwd() / "checkpoints" / cfg.checkpoint)

    log.info("Loading data module")
    data_interface_factory = DataInterfaceFactory(cfg.dataset_dir, cfg.dataset)
    data_interface = data_interface_factory.get_interface()
    data_loader = DataModule(data_interface, cfg)

    log.info("Creating model")
    model = PointGroupWrapper(cfg, data_interface=data_interface)

    log.info("Building trainer")
    checkpoint_callback = get_checkpoint_callback()
    trainer = pl.Trainer(
        gpus=cfg.gpus,
        accelerator=cfg.accelerator,
        resume_from_checkpoint=checkpoint_path,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=int(cfg.check_val_every_n_epoch),
        callbacks=[checkpoint_callback],
    )

    if "pretrain" in cfg.tasks:
        model = PointGroupBackboneWrapper(cfg)
        log.info("starting pre-training")
        trainer.fit(
            model, data_loader.pretrain_dataloader(), data_loader.pretrain_dataloader()
        )

    # Train model
    if "train" in cfg.tasks:

        log.info("starting training")
        trainer.fit(model, data_loader.train_dataloader(), data_loader.val_dataloader())

        # Load the best checkpoint so far if desired
        if cfg.eval_on_best:
            checkpoint_path = checkpoint_callback.best_model_path

    # Run to test the output
    if "eval" in cfg.tasks:
        log.info(f"Running evaluation on model {cfg.checkpoint}")

        if checkpoint_path:

            # Set the epoch to that loaded in the module
            loaded_checkpoint = torch.load(checkpoint_path)
            do_instance_segmentation = False
            if loaded_checkpoint["epoch"] >= cfg.model.train.prepare_epochs:
                do_instance_segmentation = True

            model = PointGroupWrapper.load_from_checkpoint(
                cfg=cfg,
                data_interface=data_interface,
                checkpoint_path=checkpoint_path,
                do_instance_segmentation=do_instance_segmentation,
            )

        log.info("Running on test set")
        trainer.test(model, data_loader.test_dataloader())

    # Visualize results
    if "visualize" in cfg.tasks:
        log.info("Visualizing output")
        predictions_dir = Path.cwd() / "predictions"

        Visualizer = utils.Visualizer(predictions_dir)
        Visualizer.visualize_results()


if __name__ == "__main__":
    semantics()
