"""
(Modified from PointGroup dataloader)
"""
import math
import logging
from omegaconf import DictConfig, listconfig

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from util.types import DataInterface
from dataloaders.datasets import SegmentationDataset


log = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_interface: DataInterface,
        cfg: DictConfig,
        dataset_type: SegmentationDataset,
    ):
        super().__init__()

        # Get number of gpus
        if type(cfg.gpus) == listconfig.ListConfig:
            num_gpus = len(cfg.gpus)
        else:
            num_gpus = cfg.gpus

        # Get batch sizes
        self.batch_size = math.floor(cfg.dataset.batch_size / num_gpus)
        self.pretrain_batch_size = math.floor(
            cfg.dataset.pretrain.batch_size / num_gpus
        )
        log.info(f"Using batch size of {self.batch_size}")
        log.info(f"Using pretraining batch size of {self.pretrain_batch_size}")

        # Dataloader specific parameters
        self.cfg = cfg
        self.max_npoint = cfg.dataset.max_npoint
        self.max_pointcloud_size = cfg.model.test.max_pointcloud_size
        self.ignore_label = cfg.dataset.ignore_label

        # Number of workers
        self.num_workers = cfg.model.train.train_workers

        # Load data from interface
        self.pretrain_data = data_interface.pretrain_data
        self.pretrain_val_data = data_interface.pretrain_val_data
        self.train_data = data_interface.train_data
        self.val_data = data_interface.val_data
        self.test_data = data_interface.test_data

        # Grab label to index map
        self.label_to_index_map = data_interface.label_to_index_map

        log.info(f"Pretraining samples: {len(self.pretrain_data)}")
        log.info(f"Training samples: {len(self.train_data)}")
        log.info(f"Validation samples: {len(self.val_data)}")
        log.info(f"Testing samples: {len(self.test_data)}")

        self.dataset_type = dataset_type

    def pretrain_dataloader(self):
        dataset = self.dataset_type(self.pretrain_data, self.cfg)
        return DataLoader(
            dataset,
            batch_size=self.pretrain_batch_size,
            collate_fn=dataset.collate,
            num_workers=self.num_workers,
            shuffle=True,
            sampler=None,
            drop_last=True,
            pin_memory=True,
        )

    def pretrain_val_dataloader(self):
        dataset = self.dataset_type(self.pretrain_val_data, self.cfg)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def train_dataloader(self):
        dataset = self.dataset_type(self.train_data, self.cfg, is_test=False)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate,
            num_workers=self.num_workers,
            shuffle=True,
            sampler=None,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = self.dataset_type(self.val_data, self.cfg, is_test=False)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = self.dataset_type(self.test_data, self.cfg, is_test=True)
        return DataLoader(
            dataset,
            batch_size=1,
            collate_fn=dataset.collate,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
