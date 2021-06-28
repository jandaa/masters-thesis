import logging
from pathlib import Path
from dateutil.parser import parse

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from util import utils
from dataloaders.dataloader import DataModule
from dataloaders.data_interface import DataInterfaceFactory
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
        every_n_val_epochs=1,
    )


@hydra.main(config_path="config", config_name="config")
def semantics(cfg: DictConfig) -> None:

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
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
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
            model = PointGroupWrapper.load_from_checkpoint(
                cfg=cfg, data_interface=data_interface, checkpoint_path=checkpoint_path
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


## Just in case
# scannet = DataModule(scannet_interface, cfg)
# scannet_original = OriginalDataModule(scannet_interface, cfg)

# # Make sure both output the same thing
# import random
# import torch
# import numpy as np

# for i in range(len(scannet_interface.train_data)):
#     ids = random.sample(range(len(scannet_interface.train_data)), 3)

#     np.random.seed(42)
#     batch = scannet.merge(ids, scannet_interface.train_data)
#     batch_original = scannet_original.merge(ids, scannet_interface.train_data)

#     # Make sure all parts are the same
#     assert torch.all(batch.coordinates == batch_original.coordinates)
#     assert torch.all(batch.voxel_coordinates == batch_original.voxel_coordinates)
#     assert torch.all(batch.point_to_voxel_map == batch_original.point_to_voxel_map)
#     assert torch.all(batch.voxel_to_point_map == batch_original.voxel_to_point_map)
#     assert torch.all(batch.point_coordinates == batch_original.point_coordinates)
#     assert torch.all(batch.features == batch_original.features)
#     assert torch.all(batch.labels == batch_original.labels)
#     assert torch.all(batch.instance_labels == batch_original.instance_labels)
#     assert torch.all(
#         batch.instance_centers == batch_original.instance_centers[:, 0:3]
#     )
#     assert torch.all(batch.instance_pointnum == batch_original.instance_pointnum)
#     assert torch.all(batch.offsets == batch_original.offsets)
#     assert batch.id == batch_original.id
#     assert np.all(batch.spatial_shape == batch_original.spatial_shape)
#     assert batch.test_filename == batch_original.test_filename
