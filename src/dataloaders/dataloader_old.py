"""
(Modified from PointGroup dataloader)
"""
import logging
import math
import random

from omegaconf import DictConfig

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np


from util.types import DataInterface, SceneWithLabels
from util.utils import (
    apply_data_operation_in_parallel,
)

from dataloaders.merge import (
    pointgroup_merge,
    pointgroup_pretrain_merge,
    minkowski_merge,
)


log = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_interface: DataInterface, cfg: DictConfig):
        super().__init__()

        # Dataloader specific parameters
        self.batch_size = cfg.dataset.batch_size
        self.pretrain_batch_size = cfg.dataset.pretrain.batch_size
        self.scale = cfg.dataset.scale  # voxel_size = 1 / scale, scale 50(2cm)
        self.max_npoint = cfg.dataset.max_npoint
        self.max_pointcloud_size = cfg.model.test.max_pointcloud_size
        self.mode = cfg.dataset.mode
        self.ignore_label = cfg.dataset.ignore_label
        self.are_scenes_preloaded = cfg.preload_data

        log.info(f"preload_data: {self.are_scenes_preloaded}")

        # What kind of test?
        # val == with labels
        # test == without labels
        self.test_split = cfg.model.test.split  # val or test

        # Number of workers
        self.train_workers = cfg.model.train.train_workers
        self.val_workers = cfg.model.train.train_workers
        self.test_workers = cfg.model.test.test_workers

        # Load data from interface
        self.pretrain_data = data_interface.pretrain_data
        self.train_data = data_interface.train_data
        self.val_data = data_interface.val_data
        self.test_data = data_interface.test_data

        # Grab label to index map
        self.label_to_index_map = data_interface.label_to_index_map

        if self.are_scenes_preloaded:
            self.train_data = apply_data_operation_in_parallel(
                self.preload_scenes_with_crop,
                self.train_data,
                self.train_workers,
            )
            self.val_data = apply_data_operation_in_parallel(
                self.preload_scenes_with_crop,
                self.val_data,
                self.train_workers,
            )
            self.test_data = apply_data_operation_in_parallel(
                self.preload_scenes,
                self.test_data,
                self.train_workers,
            )

        else:
            # Duplicate large scenes because they will ultimately be cropped
            self.train_data = self.duplicate_large_scenes(self.train_data)
            self.val_data = self.duplicate_large_scenes(self.val_data)

        log.info(f"Training samples: {len(self.train_data)}")
        log.info(f"Validation samples: {len(self.val_data)}")
        log.info(f"Testing samples: {len(self.test_data)}")

        # set merge function based on model
        if cfg.model.name == "pointgroup":
            self.merge = pointgroup_merge
            self.pretrain_merge = pointgroup_pretrain_merge
        elif cfg.model.name == "minkowski":
            self.merge = minkowski_merge
            self.pretrain_merge = pointgroup_pretrain_merge
        else:
            raise RuntimeError(f"model {cfg.model.name} not supported")

    def pretrain_dataloader(self):
        return DataLoader(
            list(range(len(self.pretrain_data))),
            batch_size=self.pretrain_batch_size,
            collate_fn=self.pretrain_merge,
            num_workers=self.train_workers,
            shuffle=True,
            sampler=None,
            drop_last=True,
            pin_memory=True,
        )

    def train_dataloader(self):
        return DataLoader(
            list(range(len(self.train_data))),
            batch_size=self.batch_size,
            collate_fn=self.train_merge,
            num_workers=self.train_workers,
            shuffle=True,
            sampler=None,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            list(range(len(self.val_data))),
            batch_size=self.batch_size,
            collate_fn=self.val_merge,
            num_workers=self.val_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            list(range(len(self.test_data))),
            batch_size=1,
            collate_fn=self.test_merge,
            num_workers=self.test_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def get_instance_centers(self, xyz, instance_labels):
        """
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        """

        instance_centers = np.ones(xyz.shape, dtype=np.float32) * -100
        instance_pointnum = []  # (nInst), int

        # unique_instances = np.unique(instance_labels)
        number_of_instances = int(instance_labels.max()) + 1
        for instance_label in range(number_of_instances):
            instance_indices = np.where(instance_labels == instance_label)
            instance_centers[instance_indices] = xyz[instance_indices].mean(0)
            instance_pointnum.append(instance_indices[0].size)

        return number_of_instances, instance_pointnum, instance_centers

    def train_merge(self, id):
        return self.merge(self, id, self.train_data, crop=True)

    def val_merge(self, id):
        return self.merge(self, id, self.val_data, crop=True)

    def test_merge(self, id):
        return self.merge(self, id, self.test_data, is_test=True)
