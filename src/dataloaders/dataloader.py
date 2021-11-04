"""
(Modified from PointGroup dataloader)
"""
import logging
from omegaconf import DictConfig

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

from util.types import DataInterface
from dataloaders.datasets import MinkowskiDataset


log = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_interface: DataInterface, cfg: DictConfig):
        super().__init__()

        # Dataloader specific parameters
        self.cfg = cfg
        self.batch_size = cfg.dataset.batch_size
        self.scale = cfg.dataset.scale  # voxel_size = 1 / scale, scale 50(2cm)
        self.max_npoint = cfg.dataset.max_npoint
        self.max_pointcloud_size = cfg.model.test.max_pointcloud_size
        self.ignore_label = cfg.dataset.ignore_label

        # Number of workers
        self.num_workers = cfg.model.train.train_workers

        # Load data from interface
        self.train_data = data_interface.train_data
        self.val_data = data_interface.val_data
        self.test_data = data_interface.test_data

        # Grab label to index map
        self.label_to_index_map = data_interface.label_to_index_map

        log.info(f"Training samples: {len(self.train_data)}")
        log.info(f"Validation samples: {len(self.val_data)}")
        log.info(f"Testing samples: {len(self.test_data)}")

        self.dataset_type = MinkowskiDataset

    def train_dataloader(self):
        return DataLoader(
            MinkowskiDataset(self.train_data, self.cfg, is_test=False),
            batch_size=self.batch_size,
            collate_fn=self.dataset_type.collate,
            num_workers=self.num_workers,
            shuffle=True,
            sampler=None,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            MinkowskiDataset(self.val_data, self.cfg, is_test=False),
            batch_size=self.batch_size,
            collate_fn=self.dataset_type.collate,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            MinkowskiDataset(self.test_data, self.cfg, is_test=False),
            batch_size=1,
            collate_fn=self.dataset_type.collate,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
