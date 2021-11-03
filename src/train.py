import logging
from pathlib import Path
from dateutil.parser import parse

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np
import random
import torch

from model_factory import ModelFactory
from util import utils
from dataloaders.dataloader import DataModule
from dataloaders.data_interface import DataInterfaceFactory


log = logging.getLogger("train")


def get_pretrain_checkpoint_callback():
    return ModelCheckpoint(
        dirpath="pretrain_checkpoints",
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
    )


def get_checkpoint_callback():
    return ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
    )


@hydra.main(config_path="config", config_name="config")
def semantics(cfg: DictConfig) -> None:

    # Set random seeds for reproductability
    pl.seed_everything(42, workers=True)

    # Load a checkpoint if given
    checkpoint_path = None
    if cfg.checkpoint:
        checkpoint_path = str(Path.cwd() / "checkpoints" / cfg.checkpoint)

    log.info("Loading data module")
    data_interface_factory = DataInterfaceFactory(cfg)
    data_interface = data_interface_factory.get_interface()
    data_loader = DataModule(data_interface, cfg)

    # load pretrained backbone if desired
    backbone = None
    pretrain_checkpoint = None
    if cfg.pretrain_checkpoint:
        pretrain_checkpoint = str(
            Path.cwd() / "pretrain_checkpoints" / cfg.pretrain_checkpoint
        )
        backbone = PointGroupBackboneWrapper.load_from_checkpoint(
            cfg=cfg,
            checkpoint_path=pretrain_checkpoint,
        ).model
        log.info(f"Loaded pretrained checkpoint: {cfg.pretrain_checkpoint}")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    if "pretrain" in cfg.tasks:

        log.info("Creating backbone model")
        backbonewraper = PointGroupBackboneWrapper(cfg)

        log.info("Building trainer")
        checkpoint_callback = get_pretrain_checkpoint_callback()

        trainer = pl.Trainer(
            gpus=cfg.gpus,
            accelerator=cfg.accelerator,
            resume_from_checkpoint=pretrain_checkpoint,
            max_epochs=cfg.dataset.pretrain.max_epochs,
            check_val_every_n_epoch=int(5),
            callbacks=[checkpoint_callback, lr_monitor],
            limit_train_batches=cfg.limit_train_batches,
            deterministic=True,
        )

        log.info("starting pre-training")
        trainer.fit(
            backbonewraper,
            data_loader.pretrain_dataloader(),
            data_loader.pretrain_dataloader(),
        )
        log.info("finished pretraining")

        backbone = backbonewraper.model

    log.info("Building trainer")
    checkpoint_callback = get_checkpoint_callback()
    trainer = pl.Trainer(
        gpus=cfg.gpus,
        accelerator=cfg.accelerator,
        resume_from_checkpoint=checkpoint_path,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=int(cfg.check_val_every_n_epoch),
        callbacks=[checkpoint_callback, lr_monitor],
        limit_train_batches=cfg.limit_train_batches,
        deterministic=True,
        precision=cfg.precision
        # profiler="simple",
    )

    log.info("Creating model")
    model_factory = ModelFactory(cfg, data_interface, backbone=backbone)
    model = model_factory.get_model()

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
            model = model_factory.load_from_checkpoint(checkpoint_path)

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
