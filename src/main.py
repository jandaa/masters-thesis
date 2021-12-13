import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pytorch_lightning.trainer import data_loading

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from util import utils
from models.factory import ModelFactory
from dataloaders.datamodule import DataModule
from datasets.interface import DataInterfaceFactory


log = logging.getLogger("main")


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


lr_monitor = LearningRateMonitor(logging_interval="step")


class Trainer:
    """High level training class."""

    def __init__(self, cfg: DictConfig):

        # Save configuration
        self.cfg = cfg

        # Set absolute checkpoint paths
        self.checkpoint_path = None
        if cfg.checkpoint:
            self.checkpoint_path = str(Path.cwd() / "checkpoints" / cfg.checkpoint)
            log.info(f"Resuming checkpoint: {Path(self.checkpoint_path).name}")

        self.pretrain_checkpoint = None
        if cfg.pretrain_checkpoint:
            self.pretrain_checkpoint = str(
                Path.cwd() / "pretrain_checkpoints" / cfg.pretrain_checkpoint
            )
            log.info(
                f"Resuming pretraining checkpoint: {Path(self.pretrain_checkpoint).name}"
            )

        self.supervised_pretrain_checkpoint = None
        if cfg.supervised_pretrain_checkpoint:
            self.supervised_pretrain_checkpoint = str(
                Path.cwd() / "pretrain_checkpoints" / cfg.supervised_pretrain_checkpoint
            )
            log.info(
                f"Resuming supervied pretraining checkpoint: {Path(self.supervised_pretrain_checkpoint).name}"
            )

        log.info("Loading data module")
        self.data_interface = DataInterfaceFactory(cfg).get_interface()
        self.model_factory = ModelFactory(cfg, self.data_interface)

        # Create model
        log.info("Creating Model")
        self.model = self.model_factory.get_model()

        # Init variables
        self.trainer = self.get_trainer()
        self.pretrainer = self.get_pretrainer()
        self.data_loader = None

        # Load supervised pretrain checkpoint if available
        if self.supervised_pretrain_checkpoint:
            self.load_supervised_checkpoint()

    def load_supervised_checkpoint(self):
        state_dict = torch.load(self.supervised_pretrain_checkpoint)["state_dict"]
        for weight in state_dict.keys():
            if "model.backbone" not in weight:
                del state_dict[weight]

        self.model.load_state_dict(state_dict, strict=False)

        log.info(
            f"Loaded supervised pretrained checkpoint: {Path(self.supervised_pretrain_checkpoint).name}"
        )

    def get_trainer(self):
        """Build a trainer for regular training"""
        log.info("Building Trainer")
        return pl.Trainer(
            gpus=self.cfg.gpus,
            accelerator=self.cfg.accelerator,
            resume_from_checkpoint=self.checkpoint_path,
            max_epochs=self.cfg.max_epochs,
            check_val_every_n_epoch=int(self.cfg.check_val_every_n_epoch),
            callbacks=[get_checkpoint_callback(), lr_monitor],
            limit_train_batches=self.cfg.limit_train_batches,
            deterministic=True,
            precision=self.cfg.precision,
        )

    def get_pretrainer(self):
        """Build a pretrainer for pretraining backbone models."""
        log.info("Building Pre-Trainer")
        return pl.Trainer(
            gpus=self.cfg.gpus,
            accelerator=self.cfg.accelerator,
            resume_from_checkpoint=self.pretrain_checkpoint,
            max_steps=self.cfg.dataset.pretrain.max_steps,
            check_val_every_n_epoch=self.cfg.check_val_every_n_epoch,
            callbacks=[get_pretrain_checkpoint_callback(), lr_monitor],
            limit_train_batches=self.cfg.limit_train_batches,
            accumulate_grad_batches=self.cfg.dataset.pretrain.accumulate_grad_batches,
            deterministic=True,
        )

    def get_datamodule(self):
        if self.data_loader:
            return self.data_loader
        else:
            log.info("Creating DataModule")
            self.data_loader = DataModule(
                self.data_interface,
                self.cfg,
                dataset_type=self.model_factory.get_dataset_type(),
            )
            return self.data_loader

    def pretrain(self):
        """Pretrain a network with an unsupervised objective."""

        log.info("Loading backbone dataloader")
        dataset_type = self.model_factory.get_backbone_dataset_type()
        pretrain_data_loader = DataModule(
            self.data_interface, self.cfg, dataset_type=dataset_type, is_pretrain=True
        )

        log.info("Creating backbone model")
        backbone_wrapper_type = self.model_factory.get_backbone_wrapper_type()
        if self.pretrain_checkpoint:
            log.info("Continuing pretraining from checkpoint.")
            backbonewraper = backbone_wrapper_type.load_from_checkpoint(
                cfg=self.cfg,
                checkpoint_path=self.pretrain_checkpoint,
            )
        else:
            backbonewraper = backbone_wrapper_type(self.cfg)

        trainer = self.get_pretrainer()

        log.info("starting pre-training")
        trainer.fit(
            backbonewraper,
            pretrain_data_loader.pretrain_dataloader(),
            pretrain_data_loader.pretrain_val_dataloader(),
        )
        log.info("finished pretraining")

        # Store pretrained backbone model for use in future tasks
        self.model.backbone = backbonewraper.model

    def train(self):
        """Train on superivised data."""

        data_loader = self.get_datamodule()

        log.info("starting training")
        self.trainer.fit(
            self.model,
            data_loader.train_dataloader(),
            data_loader.val_dataloader(),
        )
        log.info("Finished training")

        # Load the best checkpoint so far if desired
        if self.cfg.eval_on_best:
            self.checkpoint_path = get_checkpoint_callback().best_model_path

    def eval(self):
        """Evaluate full pipline on test set."""
        log.info(f"Running evaluation on model {Path(self.checkpoint_path).name}")

        data_loader = self.get_datamodule()

        if self.checkpoint_path:
            self.model = self.model_factory.load_from_checkpoint(self.checkpoint_path)

        log.info("Running on test set")
        self.trainer.test(self.model, data_loader.test_dataloader())

    def visualize(self):
        """Visualize the semantic and instance predicitons."""
        log.info("Visualizing output")
        predictions_dir = Path.cwd() / "predictions"

        Visualizer = utils.Visualizer(predictions_dir)
        Visualizer.visualize_results()

    def run_tasks(self):
        """Run all the tasks specified in configuration."""
        for task in self.cfg.tasks:
            if hasattr(self, task):
                log.info(f"Performing task: {task}")
                getattr(self, task)()
            else:
                raise NotImplementedError(f"Task {task} does not exist")


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Set random seeds for reproductability
    pl.seed_everything(42, workers=True)

    # Create trainer and go through all desired tasks
    trainer = Trainer(cfg)
    trainer.run_tasks()


if __name__ == "__main__":
    main()
