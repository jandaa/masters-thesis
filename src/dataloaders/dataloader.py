"""
(Modified from PointGroup dataloader)
"""
import logging
import math

from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

import scipy
import scipy.ndimage
import scipy.interpolate
from packages.pointgroup_ops.functions import pointgroup_ops

from util.types import PointGroupBatch, DataInterface

log = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_interface: DataInterface,
        cfg: DictConfig,
    ):
        super().__init__()

        # Dataloader specific parameters
        self.batch_size = cfg.dataset.batch_size
        self.full_scale = cfg.dataset.full_scale
        self.scale = cfg.dataset.scale  # voxel_size = 1 / scale, scale 50(2cm)
        self.max_npoint = cfg.dataset.max_npoint
        self.mode = cfg.dataset.mode

        # What kind of test?
        # val == with labels
        # test == without labels
        self.test_split = cfg.model.test.split  # val or test

        # Number of workers
        self.train_workers = cfg.model.train.train_workers
        self.val_workers = cfg.model.train.train_workers
        self.test_workers = cfg.model.test.test_workers

        # Load data from interface
        self.train_data = data_interface.train_data
        self.val_data = data_interface.val_data
        self.test_data = data_interface.test_data

        log.info(f"Training samples: {len(self.train_data)}")
        log.info(f"Validation samples: {len(self.train_data)}")
        log.info(f"Testing samples: {len(self.train_data)}")

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

    def elastic_distortion(self, x, granularity, magnitude):
        # rng = np.random
        rng = np.random.RandomState(2)

        blurs = [
            np.ones(shape).astype("float32") / 3
            for shape in [(3, 1, 1), (1, 3, 1), (1, 1, 3)]
        ]

        # Select random noise for each voxel of bounding box
        bounding_box = np.abs(x).max(0).astype(np.int32) // granularity + 3
        noise = [rng.randn(*list(bounding_box)).astype("float32") for _ in range(3)]

        # Apply bluring filters on the noise
        for _ in range(2):
            for blur in blurs:
                noise = [
                    scipy.ndimage.filters.convolve(n, blur, mode="constant", cval=0)
                    for n in noise
                ]

        # Interpolate between the axes
        ax = [
            np.linspace(
                -(side_length - 1) * granularity,
                (side_length - 1) * granularity,
                side_length,
            )
            for side_length in bounding_box
        ]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0
            )
            for n in noise
        ]

        return x + np.hstack([i(x)[:, None] for i in interp]) * magnitude

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

    def augment_data(self, xyz, jitter=False, flip=False, rot=False):
        # rng = np.random
        rng = np.random.RandomState(2)

        m = np.eye(3)
        if jitter:
            m += rng.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= rng.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = rng.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [
                    [math.cos(theta), math.sin(theta), 0],
                    [-math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1],
                ],
            )  # rotation
        return np.matmul(xyz, m)

    def crop(self, xyz):
        """
        :param xyz: (n, 3) >= 0
        """
        # rng = np.random
        rng = np.random.RandomState(2)

        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.max_npoint:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * rng.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * (
                (xyz_offset < full_scale).sum(1) == 3
            )
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def train_merge(self, id):
        return self.merge(id, self.train_data)

    def val_merge(self, id):
        return self.merge(id, self.val_data)

    def test_merge(self, id):
        return self.merge(id, self.test_data, is_test=True)

    def merge(self, id, scenes, is_test=False):

        # Make sure valid test split option is specified
        if is_test and self.test_split not in ["val", "test"]:
            raise RuntimeError(f"Wrong test split: {self.test_split}")

        # Whether semantics and instance labels are available
        are_labels_available = is_test and self.test_split == "val" or not is_test

        batch_coordinates = []
        batch_point_coordinates = []
        batch_features = []
        batch_semantic_labels = []
        batch_instance_labels = []

        batch_instance_centers = []  # (N, 9)
        batch_instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):

            scene = scenes[idx]

            if is_test:

                xyz_middle = self.augment_data(scene.points, False, True, True)

                xyz = xyz_middle * self.scale

                xyz -= xyz.min(0)

                batch_features.append(torch.from_numpy(scene.features))

                semantic_labels = scene.semantic_labels
                instance_labels = scene.instance_labels

            else:
                ### jitter / flip x / rotation
                xyz_middle = self.augment_data(scene.points, True, True, True)

                ### scale
                xyz = xyz_middle * self.scale

                ### elastic
                xyz = self.elastic_distortion(
                    xyz, 6 * self.scale // 50, 40 * self.scale / 50
                )
                xyz = self.elastic_distortion(
                    xyz, 20 * self.scale // 50, 160 * self.scale / 50
                )

                ### offset
                xyz -= xyz.min(0)

                ### crop
                xyz, valid_idxs = self.crop(xyz)

                xyz_middle = xyz_middle[valid_idxs]
                xyz = xyz[valid_idxs]
                rgb = scene.features[valid_idxs]
                semantic_labels = scene.semantic_labels[valid_idxs]
                instance_labels = self.getCroppedInstLabel(
                    scene.instance_labels, valid_idxs
                )

                torch.set_rng_state(torch.manual_seed(10).get_state())
                batch_features.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)

            if are_labels_available:
                ### get instance information
                (
                    number_of_instances,
                    instance_pointnum,
                    instance_centers,
                ) = self.get_instance_centers(
                    xyz_middle, instance_labels.astype(np.int32)
                )

                # TODO: why do this?
                # They do this because they combine all the scenes in the batch into one vector
                instance_labels[np.where(instance_labels != -100)] += total_inst_num
                total_inst_num += number_of_instances

                # Add training and validation info
                batch_instance_centers.append(torch.from_numpy(instance_centers))
                batch_instance_pointnum.extend(instance_pointnum)

                batch_semantic_labels.append(torch.from_numpy(semantic_labels))
                batch_instance_labels.append(torch.from_numpy(instance_labels))

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            batch_coordinates.append(
                torch.cat(
                    [
                        torch.LongTensor(xyz.shape[0], 1).fill_(i),
                        torch.from_numpy(xyz).long(),
                    ],
                    1,
                )
            )
            batch_point_coordinates.append(torch.from_numpy(xyz_middle))

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        coordinates = torch.cat(
            batch_coordinates, 0
        )  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        point_coordinates = torch.cat(batch_point_coordinates, 0).to(
            torch.float32
        )  # float (N, 3)
        features = torch.cat(batch_features, 0)  # float (N, C)

        spatial_shape = np.clip(
            (coordinates.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None
        )  # long (3)

        ### voxelize
        (
            voxel_coordinates,
            point_to_voxel_map,
            voxel_to_point_map,
        ) = pointgroup_ops.voxelization_idx(coordinates, self.batch_size, self.mode)

        if are_labels_available:
            semantic_labels = torch.cat(batch_semantic_labels, 0).long()  # long (N)
            instance_labels = torch.cat(batch_instance_labels, 0).long()  # long (N)

            instance_centers = torch.cat(batch_instance_centers, 0).to(
                torch.float32
            )  # float (N, 9) (meanxyz, minxyz, maxxyz)
            instance_pointnum = torch.tensor(
                batch_instance_pointnum, dtype=torch.int
            )  # int (total_nInst)

        if is_test:
            test_filename = scenes[id[0]].name
        else:
            test_filename = None

        return PointGroupBatch(
            coordinates=coordinates,
            voxel_coordinates=voxel_coordinates,
            point_to_voxel_map=point_to_voxel_map,
            voxel_to_point_map=voxel_to_point_map,
            point_coordinates=point_coordinates,
            features=features,
            labels=semantic_labels,
            instance_labels=instance_labels,
            instance_centers=instance_centers,
            instance_pointnum=instance_pointnum,
            offsets=batch_offsets,
            id=id,
            spatial_shape=spatial_shape,
            test_filename=test_filename,
        )
