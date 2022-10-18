import logging
from pathlib import Path
import random

import hydra
from omegaconf import DictConfig
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from util import utils
from models.factory import ModelFactory
from dataloaders.datamodule import DataModule
from datasets.interface import DataInterfaceFactory
import MinkowskiEngine as ME
import open3d as o3d

# Visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import PIL

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
        filename="{epoch}-{step}-{val_semantic_mIOU:.3f}",
        save_last=True,
        monitor="val_semantic_mIOU",
        mode="max",
        save_top_k=3,
    )


def embed_tsne(data):
    """
    N x D np.array data
    """
    tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(data)
    tsne_results = np.squeeze(tsne_results)
    tsne_min = np.min(tsne_results)
    tsne_max = np.max(tsne_results)
    return (tsne_results - tsne_min) / (tsne_max - tsne_min)


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

        self.checkpoint_2d_path = None
        if cfg.pretrain_checkpoint_2d:
            self.checkpoint_2d_path = str(
                Path.cwd() / "pretrain_checkpoints_2d" / cfg.pretrain_checkpoint_2d
            )
            log.info(f"Resuming 2D checkpoint: {Path(self.checkpoint_2d_path).name}")

        log.info("Loading data module")
        self.data_interface = DataInterfaceFactory(cfg).get_interface()
        self.model_factory = ModelFactory(
            cfg, self.data_interface, checkpoint_2d_path=self.checkpoint_2d_path
        )

        if self.pretrain_checkpoint:
            log.info("Continuing pretraining from checkpoint.")
            backbone_wrapper_type = self.model_factory.get_backbone_wrapper_type()

            # Load backbone parameters only
            # self.pretrain_checkpoint = self.pretrain_checkpoint.replace(
            #     "pretrain_checkpoints", "pretrain_checkpoints_2d"
            # )
            checkpoint = torch.load(self.pretrain_checkpoint)
            head_params = [
                key
                for key in checkpoint["state_dict"]
                if "_model.head" in key or "head" in key
            ]
            for key in head_params:
                del checkpoint["state_dict"][key]

            backbonewraper = backbone_wrapper_type._load_model_state(
                cfg=self.cfg, checkpoint=checkpoint, strict=False
            )
            self.model_factory = ModelFactory(
                cfg, self.data_interface, backbone=backbonewraper.model
            )

        # Create model
        log.info("Creating Model")
        self.model = self.model_factory.get_model()

        # Init variables
        self.trainer = self.get_trainer()
        self.pretrainer = self.get_pretrainer()
        self.data_loader = None

    def get_trainer(self):
        """Build a trainer for regular training"""
        log.info("Building Trainer")
        tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/train")
        return pl.Trainer(
            logger=tb_logger,
            gpus=self.cfg.gpus,
            accelerator=self.cfg.accelerator,
            max_epochs=self.cfg.max_epochs,
            resume_from_checkpoint=self.checkpoint_path,
            check_val_every_n_epoch=int(self.cfg.check_val_every_n_epoch),
            callbacks=[get_checkpoint_callback(), lr_monitor],
            limit_train_batches=self.cfg.limit_train_batches,
            limit_val_batches=self.cfg.limit_val_batches,
            limit_test_batches=self.cfg.limit_test_batches,
            accumulate_grad_batches=self.cfg.dataset.accumulate_grad_batches,
            deterministic=True,
            precision=self.cfg.precision,
            max_time=self.cfg.max_time,
            val_check_interval=self.cfg.val_check_interval,
        )

    def get_pretrainer(self):
        """Build a pretrainer for pretraining backbone models."""
        log.info("Building Pre-Trainer")
        tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/pretrain")
        return pl.Trainer(
            logger=tb_logger,
            gpus=self.cfg.gpus,
            accelerator=self.cfg.accelerator,
            resume_from_checkpoint=self.pretrain_checkpoint,
            max_steps=self.cfg.dataset.pretrain.max_steps,
            check_val_every_n_epoch=self.cfg.check_val_every_n_epoch,
            callbacks=[get_pretrain_checkpoint_callback(), lr_monitor],
            limit_train_batches=self.cfg.limit_train_batches,
            limit_val_batches=self.cfg.limit_val_batches,
            limit_test_batches=self.cfg.limit_test_batches,
            accumulate_grad_batches=self.cfg.dataset.pretrain.accumulate_grad_batches,
            deterministic=True,
            max_time=self.cfg.max_time,
            val_check_interval=self.cfg.val_check_interval,
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
        elif self.checkpoint_2d_path:
            log.info("Loading pretrained 2D network")
            backbonewraper = backbone_wrapper_type(
                self.cfg, checkpoint_2d_path=self.checkpoint_2d_path
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

        # if self.checkpoint_path:
        #     checkpoint = torch.load(self.checkpoint_path)
        #     self.model.load_state_dict(checkpoint["state_dict"])

        log.info("starting training")
        self.trainer.fit(
            self.model,
            data_loader.train_dataloader(),
            data_loader.val_dataloader(),
        )
        log.info("Finished training")

        # Load the best checkpoint so far if desired
        if self.cfg.eval_on_best:
            self.checkpoint_path = self.trainer.checkpoint_callback.best_model_path

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

    def pretrain_vis(self):
        """Visualize the pretrained feature embeddings."""
        log.info("Visualizing 2D feature embeddings")

        # Load model
        backbone_wrapper_type = self.model_factory.get_backbone_wrapper_type()
        if self.checkpoint_2d_path:
            log.info("Continuing pretraining from checkpoint.")
            backbonewraper = backbone_wrapper_type.load_from_checkpoint(
                cfg=self.cfg,
                checkpoint_path=self.checkpoint_2d_path,
            )
        else:
            backbonewraper = backbone_wrapper_type(self.cfg)

        # Get data
        dataset_type = self.model_factory.get_backbone_dataset_type()
        dataset = dataset_type(self.data_interface.pretrain_val_data, self.cfg)
        collate_fn = dataset.collate
        scene = collate_fn([dataset[2100]])

        # Generate feature embeddings
        z1 = backbonewraper.model(scene.images2).detach().cpu().numpy()

        # TSNE
        test = z1.reshape(16, -1).T
        embeddings = embed_tsne(test)
        embeddings = embeddings.reshape(z1.shape[2], z1.shape[3])

        # Plot TSNE results
        plt.imshow(embeddings, cmap="autumn", interpolation="nearest")
        plt.title("2-D Heat Map")
        plt.show()

        waithere = 1

    def pretrain_vis_3d(self):
        """Visualize the pretrained feature embeddings."""
        log.info("Visualizing 3D feature embeddings")

        # Load model
        backbone_wrapper_type = self.model_factory.get_backbone_wrapper_type()
        if self.pretrain_checkpoint:
            log.info("Continuing pretraining from checkpoint.")
            backbonewraper = backbone_wrapper_type.load_from_checkpoint(
                cfg=self.cfg,
                checkpoint_path=self.pretrain_checkpoint,
            )
        else:
            backbonewraper = backbone_wrapper_type(self.cfg)

        # Get data
        dataset_type = self.model_factory.get_backbone_dataset_type()
        dataset = dataset_type(self.data_interface.pretrain_val_data, self.cfg)
        collate_fn = dataset.collate
        scene = collate_fn([dataset[2100]])

        # Generate feature embeddings
        input_3d_1 = ME.SparseTensor(scene.features1, scene.points1)
        output_3d_1 = backbonewraper.model(input_3d_1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            scene.points1[:, 1:4].detach().cpu().numpy()
        )
        pcd.colors = o3d.utility.Vector3dVector(scene.features1.detach().cpu().numpy())

        # from util.utils import mesh_sphere

        # pcd = mesh_sphere(pcd, 0.02)
        # o3d.visualization.draw_geometries([pcd])

        vis_pcd = utils.get_colored_point_cloud_feature(
            pcd,
            output_3d_1.F.detach().cpu().numpy(),
            0.02,
        )

        # o3d.io.write_triangle_mesh("vis_3d.obj", vis_pcd, write_vertex_normals=False)
        o3d.visualization.draw_geometries([vis_pcd])

        waithere = 1

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
    seed = 42
    torch.manual_seed(seed)
    random.seed(0)
    np.random.seed(0)
    # torch.use_deterministic_algorithms(True)
    pl.seed_everything(seed, workers=True)

    # Create trainer and go through all desired tasks
    trainer = Trainer(cfg)
    trainer.run_tasks()


if __name__ == "__main__":
    main()
