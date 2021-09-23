"""
(Modified from PointGroup dataloader)
"""
import logging
import math
import random

from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import open3d as o3d

import scipy
import scipy.ndimage
import scipy.interpolate
from scipy.spatial import KDTree
from packages.pointgroup_ops.functions import pointgroup_ops

from util.types import PointGroupBatch, DataInterface, PretrainInput, SceneWithLabels

from util.utils import apply_data_operation_in_parallel, get_random_colour

log = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_interface: DataInterface, cfg: DictConfig):
        super().__init__()

        # Dataloader specific parameters
        self.batch_size = cfg.dataset.batch_size
        self.scale = cfg.dataset.scale  # voxel_size = 1 / scale, scale 50(2cm)
        self.max_npoint = cfg.dataset.max_npoint
        self.max_pointcloud_size = cfg.model.test.max_pointcloud_size
        self.mode = cfg.dataset.mode
        self.ignore_label = cfg.dataset.ignore_label
        self.are_scenes_preloaded = cfg.preload_data
        self.force_reload = cfg.force_reload
        log.info(f"preload_data: {self.are_scenes_preloaded}")
        log.info(f"force_reload: {self.force_reload}")

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

        # Preprocess all data in parallel
        # TODO: put back test_data
        # all_datapoints = self.train_data + self.val_data + self.test_data
        all_datapoints = self.train_data + self.val_data
        apply_data_operation_in_parallel(
            self.preprocess_batch, all_datapoints, self.train_workers
        )

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

    def pretrain_dataloader(self):
        return DataLoader(
            list(range(len(self.pretrain_data))),
            batch_size=self.batch_size,
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

    def preload_scene(self, scene, crop_scene):
        """Preload a batch of scenes into memory and crop where necessary."""
        scene = scene.load()
        if crop_scene:
            return self.crop_multiple(scene)
        else:
            return [scene]

    def preload_scenes(self, scenes, crop_scene=False):
        """Preload a batch of scenes into memory and crop where necessary."""
        preloaded_scenes = []
        for scene in scenes:
            preloaded_scenes += self.preload_scene(scene, crop_scene)

        return preloaded_scenes

    def preload_scenes_with_crop(self, scenes):
        """Preload a batch of scenes into memory and crop where necessary."""
        return self.preload_scenes(scenes, crop_scene=True)

    def duplicate_large_scenes(self, datapoints):
        """
        Duplicate large scenes by the number of times they are larger than the
        max number of points. This is because they will be cropped later in the
        merge function to the max points, but we want to use more of the scene.
        """
        new_datapoints = []
        for datapoint in datapoints:
            number_of_splits = math.floor(datapoint.num_points / self.max_npoint)
            number_of_splits = max(number_of_splits, 1)
            new_datapoints += [datapoint] * number_of_splits

        return new_datapoints

    def preprocess_batch(self, datapoints):
        """Run the preprocess function on a batch of datapoints"""
        for datapoint in datapoints:
            datapoint.preprocess(force_reload=self.force_reload)

    def elastic_distortion(self, x, granularity, magnitude):
        blurs = [
            np.ones(shape).astype("float32") / 3
            for shape in [(3, 1, 1), (1, 3, 1), (1, 1, 3)]
        ]

        # Select random noise for each voxel of bounding box
        bounding_box = np.abs(x).max(0).astype(np.int32) // granularity + 3
        noise = [
            np.random.randn(*list(bounding_box)).astype("float32") for _ in range(3)
        ]

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
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [
                    [math.cos(theta), math.sin(theta), 0],
                    [-math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1],
                ],
            )  # rotation
        return np.matmul(xyz, m)

    def crop_multiple(self, scene: SceneWithLabels):

        num_points = scene.points.shape[0]
        if num_points <= self.max_npoint:
            return [scene]

        num_splits = math.floor(num_points / self.max_npoint)
        scenes = self.crop(scene, num_splits=num_splits)
        if not scenes:
            return []
        return scenes

    def crop_single(self, scene: SceneWithLabels, max_npoint: int):
        if scene.points.shape[0] > max_npoint:
            scene = self.crop(scene, max_npoint=max_npoint)
            scene = scene[0]
        return scene

    def crop(self, scene: SceneWithLabels, num_splits: int = 1, max_npoint: int = None):
        """
        Crop by picking a random point and selecting all
        neighbouring points up to a max number of points
        """
        if not max_npoint:
            max_npoint = self.max_npoint

        # Build KDTree
        kd_tree = KDTree(scene.points)

        valid_instance_idx = scene.instance_labels != self.ignore_label
        unique_instance_labels = np.unique(scene.instance_labels[valid_instance_idx])

        if unique_instance_labels.size == 0:
            return False

        cropped_scenes = []
        for i in range(num_splits):

            # Randomly select a query point
            query_instance = np.random.choice(unique_instance_labels)
            query_points = scene.points[scene.instance_labels == query_instance]
            query_point_ind = random.randint(0, query_points.shape[0] - 1)
            query_point = query_points[query_point_ind]

            # select subset of neighbouring points from the random center point
            [_, idx] = kd_tree.query(query_point, k=max_npoint)

            # Make sure there is at least one instance in the scene
            current_instances = np.unique(scene.instance_labels[idx])
            if (
                current_instances.size == 1
                and current_instances[0] == self.ignore_label
            ):
                raise RuntimeError("No instances in scene")

            cropped_scene = SceneWithLabels(
                name=scene.name + f"_crop_{i}",
                points=scene.points[idx],
                features=scene.features[idx],
                semantic_labels=scene.semantic_labels[idx],
                instance_labels=scene.instance_labels[idx],
            )

            # Remap instance numbers
            instance_ids = np.unique(cropped_scene.instance_labels)
            new_index = 0
            for old_index in instance_ids:
                if old_index != self.ignore_label:
                    instance_indices = np.where(
                        cropped_scene.instance_labels == old_index
                    )
                    cropped_scene.instance_labels[instance_indices] = new_index
                    new_index += 1

            cropped_scenes.append(cropped_scene)

        return cropped_scenes

    def train_merge(self, id):
        return self.merge(id, self.train_data, crop=True)

    def val_merge(self, id):
        return self.merge(id, self.val_data, crop=True)

    def test_merge(self, id):
        return self.merge(id, self.test_data, is_test=True)

    def merge(self, id, scenes, crop=False, is_test=False):

        # Make sure valid test split option is specified
        if is_test and self.test_split not in ["val", "test"]:
            raise RuntimeError(f"Wrong test split: {self.test_split}")

        # Whether semantics and instance labels are available
        are_labels_available = is_test and self.test_split == "val" or not is_test

        batch_coordinates = []
        batch_points = []
        batch_features = []
        batch_semantic_labels = []
        batch_instance_labels = []

        batch_instance_centers = []  # (N, 9)
        batch_instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):

            scene = scenes[idx]
            if not self.are_scenes_preloaded:
                scene = scene.load()

                if crop:
                    scene = self.crop_single(scene, max_npoint=self.max_npoint)
                    if not scene:
                        continue

            if is_test:

                # Make sure the scene is not way too big
                scene = self.crop_single(scene, max_npoint=self.max_pointcloud_size)

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

                rgb = scene.features
                semantic_labels = scene.semantic_labels
                instance_labels = scene.instance_labels

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

                # They do this because they combine all the scenes in the batch into one vector
                instance_labels[
                    np.where(instance_labels != self.ignore_label)
                ] += total_inst_num
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

            batch_points.append(torch.from_numpy(xyz_middle))

        ### merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        coordinates = torch.cat(
            batch_coordinates, 0
        )  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        points = torch.cat(batch_points, 0).to(torch.float32)  # float (N, 3)
        features = torch.cat(batch_features, 0)  # float (N, C)

        spatial_shape = (coordinates.max(0)[0][1:] + 1).numpy()

        # Minimum spatial shape must be at least smallest kernel size in UNet
        for i in range(spatial_shape.size):
            spatial_shape[i] = max(spatial_shape[i], 128)

        ### voxelize
        (
            voxel_coordinates,
            point_to_voxel_map,
            voxel_to_point_map,
        ) = pointgroup_ops.voxelization_idx(coordinates, len(id), self.mode)

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
            test_filename = scenes[id[0]].scene_name
        else:
            test_filename = None

        return PointGroupBatch(
            batch_indices=coordinates[:, 0].int(),
            voxel_coordinates=voxel_coordinates,
            point_to_voxel_map=point_to_voxel_map,
            voxel_to_point_map=voxel_to_point_map,
            points=points,
            features=features,
            labels=semantic_labels,
            instance_labels=instance_labels,
            instance_centers=instance_centers,
            instance_pointnum=instance_pointnum,
            offsets=batch_offsets,
            batch_size=len(id),
            spatial_shape=spatial_shape,
            test_filename=test_filename,
        )

    def pretrain_merge(self, id):

        batch_coordinates = []
        batch_points = []
        batch_features = []

        batch_offsets = [0]

        batch_correspondances = []

        for i, idx in enumerate(id):

            scene = self.pretrain_data[idx]
            if not self.are_scenes_preloaded:
                scene = scene.load_measurements()

            # pick matching scenes at random
            frame1 = random.choice(list(scene.matching_frames_map.keys()))
            frame2 = random.choice(scene.matching_frames_map[frame1])

            frame1 = scene.measurements[frame1]
            frame2 = scene.measurements[frame2]

            # compute matching points
            frame1_kd_tree = KDTree(frame1.points)
            frame2_kd_tree = KDTree(frame2.points)
            indexes = frame1_kd_tree.query_ball_tree(frame2_kd_tree, 2.0 * 0.05, p=1)

            correspondances = {i: index[0] for i, index in enumerate(indexes) if index}

            # select a max number of correspondances
            keys = list(correspondances.keys())
            if len(keys) > 4092:
                keys = random.choices(keys, k=min(4092, len(keys)))
                correspondances = {k: correspondances[k] for k in keys}
            batch_correspondances.append(correspondances)

            for frame in [frame1, frame2]:

                # Randomly rotate each frame
                xyz_middle = frame.points - frame.points.mean(0)
                xyz_middle = self.augment_data(xyz_middle, rot=True)

                ### offset
                xyz = xyz_middle - xyz_middle.min(0)

                # Append
                batch_features.append(torch.from_numpy(frame.point_colors))

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

                batch_points.append(torch.from_numpy(xyz_middle))

        coordinates = torch.cat(
            batch_coordinates, 0
        )  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        points = torch.cat(batch_points, 0).to(torch.float32)  # float (N, 3)
        features = torch.cat(batch_features, 0).to(torch.float32)  # float (N, C)

        ### voxelize
        (
            voxel_coordinates,
            point_to_voxel_map,
            voxel_to_point_map,
        ) = pointgroup_ops.voxelization_idx(coordinates, len(id), self.mode)

        spatial_shape = (coordinates.max(0)[0][1:] + 1).numpy()

        # Minimum spatial shape must be at least smallest kernel size in UNet
        for i in range(spatial_shape.size):
            spatial_shape[i] = max(spatial_shape[i], 128)

        pretrain_input = PretrainInput(
            points=points,
            features=features,
            voxel_coordinates=voxel_coordinates,
            point_to_voxel_map=point_to_voxel_map,
            voxel_to_point_map=voxel_to_point_map,
            batch_indices=coordinates[:, 0].int(),
            spatial_shape=spatial_shape,
            correspondances=batch_correspondances,
            batch_size=2 * len(id),
            offsets=batch_offsets,
        )

        # Verify that point correspondences are correct
        # for debugging
        # self.visualize_correspondances(pretrain_input)

        return pretrain_input

    def visualize_correspondances(self, pretrain_input: PretrainInput):
        """Visualize the point correspondances between the matched scans in
        the pretrain input"""

        for i, matches in enumerate(pretrain_input.correspondances):
            points1_start = pretrain_input.offsets[2 * i]
            points2_start = pretrain_input.offsets[2 * i + 1]

            # verify that point corresponsances are correct
            # first verify that the point clouds are correct
            points1 = pretrain_input.points[points1_start:points2_start]
            colors1 = pretrain_input.features[points1_start:points2_start]
            points2 = pretrain_input.points[
                points2_start : pretrain_input.offsets[2 * (i + 1)]
            ]
            colors2 = pretrain_input.features[
                points2_start : pretrain_input.offsets[2 * (i + 1)]
            ]
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(points1.cpu().numpy())
            pcd1.colors = o3d.utility.Vector3dVector(colors1.cpu().numpy())

            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(points2.cpu().numpy())
            pcd2.colors = o3d.utility.Vector3dVector(colors2.cpu().numpy())
            pcd2 = pcd2.translate([1.0, 0, 0])

            correspondences = [(k, v) for k, v in matches.items()]

            correspondences = random.choices(correspondences, k=100)
            lineset = o3d.geometry.LineSet()
            lineset = lineset.create_from_point_cloud_correspondences(
                pcd1, pcd2, correspondences
            )
            colors = [get_random_colour() for i in range(len(correspondences))]
            lineset.colors = o3d.utility.Vector3dVector(colors)

            o3d.visualization.draw_geometries([pcd1, pcd2, lineset])
