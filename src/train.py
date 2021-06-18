import logging
from pathlib import Path
from dateutil.parser import parse

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from util import utils
from dataloaders.scannetv2 import ScannetDataModule
from model.pointgroup import PointGroupWrapper


log = logging.getLogger(__name__)


def get_checkpoint_callback():
    return ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )


@hydra.main(config_path="config", config_name="config")
def semantics(cfg: DictConfig) -> None:

    log.info(f"Current working directory: {Path.cwd()}")

    # Load a checkpoint if given
    checkpoint_path = None
    if cfg.checkpoint:
        checkpoint_path = str(Path.cwd() / "checkpoints" / cfg.checkpoint)

    log.info(f"Checkpoint path: {checkpoint_path}")

    log.info("Loading data module")
    scannet = ScannetDataModule(cfg)
    scannet.setup()

    log.info("Creating model")
    model = PointGroupWrapper(cfg)

    log.info("Building trainer")
    checkpoint_callback = get_checkpoint_callback()
    trainer = pl.Trainer(
        gpus=cfg.gpus,
        accelerator=cfg.accelerator,
        resume_from_checkpoint=checkpoint_path,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
    )

    # Train model
    if "train" in cfg.tasks:

        log.info("starting training")
        trainer.fit(model, scannet.train_dataloader(), scannet.val_dataloader())

        # Load the best checkpoint so far if desired
        if cfg.eval_on_best:
            checkpoint_path = checkpoint_callback.best_model_path

    # Run to test the output
    if "eval" in cfg.tasks:
        log.info(f"Running evaluation on model {cfg.checkpoint}")

        if checkpoint_path:
            model = PointGroupWrapper.load_from_checkpoint(
                cfg=cfg, checkpoint_path=checkpoint_path
            )

        log.info("Running on test set")
        trainer.test(model, scannet.test_dataloader())

    # Visualize results
    if "visualize" in cfg.tasks:
        log.info("Visualizing output")
        predictions_dir = Path.cwd() / "predictions"

        Visualizer = utils.Visualizer(predictions_dir)
        Visualizer.visualize_results()


if __name__ == "__main__":
    semantics()
