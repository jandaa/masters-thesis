import math
import random

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d

import dataloaders.transforms as transforms
from dataloaders.crop import crop_single

from model.minkowski.types import MinkowskiInput, MinkowskiPretrainInput
from model.pointgroup.types import PointGroupBatch
from packages.pointgroup_ops.functions import pointgroup_ops


class SegmentationDataset(Dataset):
    """Base segmentation dataset"""

    def __init__(self, scenes, cfg, is_test=False):
        super(SegmentationDataset, self).__init__()

        self.cfg = cfg
        self.num_workers = cfg.model.train.train_workers
        self.ignore_label = cfg.dataset.ignore_label
        self.is_test = is_test
        self.scale = cfg.dataset.scale
        self.mode = cfg.dataset.mode

        self.max_npoint = cfg.dataset.max_npoint
        self.max_pointcloud_size = cfg.model.test.max_pointcloud_size

        # Duplicate large scenes because they will be
        if not is_test:
            scenes = self.duplicate_large_scenes(scenes)
        self.scenes = scenes

    def __len__(self):
        return len(self.scenes)

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


class PretrainDataset(Dataset):
    """Base dataset for pretraining tasks."""

    def __init__(self, scenes, cfg, is_test=False):
        super(PretrainDataset, self).__init__()

        self.cfg = cfg
        self.scenes = scenes
        self.num_workers = cfg.model.train.train_workers
        self.ignore_label = cfg.dataset.ignore_label
        self.scale = cfg.dataset.scale

    def __len__(self):
        return len(self.scenes)


class MinkowskiPretrainDataset(PretrainDataset):
    def __init__(self, scenes, cfg):
        super(MinkowskiPretrainDataset, self).__init__(scenes, cfg)

        color_jitter_std = 0.05
        color_trans_ratio = 0.1
        self.scale_range = (0.9, 1.1)
        self.augmentations = transforms.Compose(
            [
                transforms.ChromaticTranslation(color_trans_ratio),
                transforms.ChromaticJitter(color_jitter_std),
                transforms.RandomRotate(),
            ]
        )

    def __len__(self):
        return super(MinkowskiPretrainDataset, self).__len__()

    def collate(self, batch):
        correspondences = [
            datapoint["correspondences"] for datapoint in batch if datapoint
        ]
        coords_list = [
            frame["discrete_coords"]
            for datapoint in batch
            for frame in datapoint["quantized_frames"]
            if datapoint
        ]
        features_list = [
            frame["unique_feats"]
            for datapoint in batch
            for frame in datapoint["quantized_frames"]
            if datapoint
        ]

        coordinates_batch, features_batch = ME.utils.sparse_collate(
            coords_list, features_list
        )

        pretrain_input = MinkowskiPretrainInput(
            points=coordinates_batch,
            features=features_batch.float(),
            correspondences=correspondences,
            batch_size=2 * len(batch),
        )

        return pretrain_input

    def __getitem__(self, index):

        scene = self.scenes[index].load_measurements()

        if not scene.matching_frames_map:
            return {}

        # pick matching scenes at random
        frame1 = random.choice(list(scene.matching_frames_map.keys()))
        frame2 = random.choice(scene.matching_frames_map[frame1])

        correspondences = scene.correspondance_map[frame1][frame2]

        frame1 = scene.get_measurement(frame1)
        frame2 = scene.get_measurement(frame2)

        # recursively try to select a sample correspondance
        # with a minimum overlap ratio
        if len(correspondences) / frame1.points.shape[0] < 0.3:
            return self.__getitem__(index)

        quantized_frames = []
        random_scale = np.random.uniform(*self.scale_range)
        for frame in [frame1, frame2]:

            # Extract data
            xyz = np.ascontiguousarray(frame.points)
            features = torch.from_numpy(frame.point_colors)

            # apply a random scalling
            xyz *= random_scale

            # Randomly rotate each frame
            xyz = xyz - xyz.mean(0)
            xyz, features, _ = self.augmentations(xyz, features, None)

            # Voxelize input
            (discrete_coords, mapping, inv_mapping,) = ME.utils.sparse_quantize(
                coordinates=xyz,
                quantization_size=(1 / self.scale),
                return_index=True,
                return_inverse=True,
            )

            unique_feats = features[mapping]

            # Get the point to voxel mapping
            mapping = {
                point_ind: voxel_ind
                for voxel_ind, point_ind in enumerate(mapping.numpy())
            }

            # Append to quantized frames
            quantized_frames.append(
                {
                    "discrete_coords": discrete_coords,
                    "unique_feats": unique_feats,
                    "mapping": mapping,
                }
            )

        # Remap the correspondances into voxel world
        mapping1 = quantized_frames[0]["mapping"]
        mapping2 = quantized_frames[1]["mapping"]
        correspondences = [
            (mapping1[k], mapping2[v])
            for k, v in correspondences.items()
            if k in mapping1.keys() and v in mapping2.keys()
        ]

        # select a max number of correspondances
        max_correspondances = 4092
        if len(correspondences) > max_correspondances:
            correspondences = random.sample(correspondences, max_correspondances)

        # visualize_correspondances(quantized_frames, correspondences)
        return {
            "correspondences": correspondences,
            "quantized_frames": quantized_frames,
        }


from util.utils import get_random_colour


def visualize_correspondances(quantized_frames, correspondances):
    """Visualize the point correspondances between the matched scans in
    the pretrain input"""

    # for i, matches in enumerate(pretrain_input.correspondances):
    points1 = quantized_frames[0]["discrete_coords"]
    colors1 = quantized_frames[0]["unique_feats"]

    points2 = quantized_frames[1]["discrete_coords"]
    colors2 = quantized_frames[1]["unique_feats"]

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)
    pcd2 = pcd2.translate([100.0, 0, 0])

    correspondences = random.choices(correspondances, k=100)
    lineset = o3d.geometry.LineSet()
    lineset = lineset.create_from_point_cloud_correspondences(
        pcd1, pcd2, correspondences
    )
    colors = [get_random_colour() for i in range(len(correspondences))]
    lineset.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd1, pcd2, lineset])


class MinkowskiDataset(SegmentationDataset):
    def __init__(self, scenes, cfg, is_test=False):
        super(MinkowskiDataset, self).__init__(scenes, cfg, is_test=is_test)

        color_jitter_std = 0.05
        color_trans_ratio = 0.1
        elastic_distortion_params = ((0.2, 0.4), (0.8, 1.6))
        self.augmentations = transforms.Compose(
            [
                transforms.Crop(self.max_npoint, self.ignore_label),
                transforms.RandomDropout(0.2),
                transforms.RandomHorizontalFlip("z", False),
                transforms.ChromaticTranslation(color_trans_ratio),
                transforms.ChromaticJitter(color_jitter_std),
                transforms.RandomScale(),
                transforms.RandomRotate(),
                transforms.ElasticDistortion(elastic_distortion_params),
            ]
        )

        self.test_augmentations = transforms.Compose(
            [
                transforms.Crop(self.max_pointcloud_size, self.ignore_label),
            ]
        )

    def collate(self, batch):
        coords_list = [datapoint.points for datapoint in batch]
        features_list = [datapoint.features for datapoint in batch]
        labels_list = [datapoint.labels for datapoint in batch]

        coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
            coords_list, features_list, labels_list
        )

        return MinkowskiInput(
            points=coordinates_batch,
            features=features_batch.float(),
            labels=labels_batch.int(),
            test_filename=batch[0].test_filename,
        )

    def __len__(self):
        return super(MinkowskiDataset, self).__len__()

    def __getitem__(self, id):

        # Limit absolute max size of point cloud
        scene = self.scenes[id].load()

        xyz = np.ascontiguousarray(scene.points)
        features = torch.from_numpy(scene.features)
        labels = np.array([scene.semantic_labels, scene.instance_labels]).T

        if self.is_test:
            xyz, features, labels = self.test_augmentations(xyz, features, labels)
        else:
            xyz, features, labels = self.augmentations(xyz, features, labels)

        coords, feats, labels = ME.utils.sparse_quantize(
            xyz,
            features=features,
            labels=labels[:, 0],
            quantization_size=(1 / self.scale),
        )

        return MinkowskiInput(
            points=coords,
            features=feats,
            labels=labels,
            test_filename=scene.name,
        )


class SpconvDataset(SegmentationDataset):
    def __init__(self, scenes, cfg, is_test=False):
        super(SpconvDataset, self).__init__(scenes, cfg, is_test=is_test)

    def collate(self, batch):

        # Whether semantics and instance labels are available
        are_labels_available = True

        batch_coordinates = []
        batch_points = []
        batch_features = []
        batch_semantic_labels = []
        batch_instance_labels = []

        batch_instance_centers = []  # (N, 9)
        batch_instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]

        total_inst_num = 0
        for i, scene in enumerate(batch):

            if self.is_test:

                # Make sure the scene is not way too big
                scene = crop_single(scene, self.max_pointcloud_size, self.ignore_label)

                xyz_middle = transforms.augment_data(scene.points, False, True, True)

                xyz = xyz_middle * self.scale

                xyz -= xyz.min(0)

                batch_features.append(torch.from_numpy(scene.features))

                semantic_labels = scene.semantic_labels
                instance_labels = scene.instance_labels

            else:

                ### jitter / flip x / rotation
                xyz_middle = transforms.augment_data(scene.points, True, True, True)

                ### scale
                xyz = xyz_middle * self.scale

                ### elastic
                xyz = transforms.elastic_distortion(
                    xyz, 6 * self.scale // 50, 40 * self.scale / 50
                )
                xyz = transforms.elastic_distortion(
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
        ) = pointgroup_ops.voxelization_idx(coordinates, len(batch), self.mode)

        instance_centers = None
        if are_labels_available:
            semantic_labels = torch.cat(batch_semantic_labels, 0).long()  # long (N)
            instance_labels = torch.cat(batch_instance_labels, 0).long()  # long (N)

            instance_centers = torch.cat(batch_instance_centers, 0).to(
                torch.float32
            )  # float (N, 9) (meanxyz, minxyz, maxxyz)
            instance_pointnum = torch.tensor(
                batch_instance_pointnum, dtype=torch.int
            )  # int (total_nInst)

        if self.is_test:
            test_filename = batch[0].name
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
            batch_size=len(batch),
            spatial_shape=spatial_shape,
            test_filename=test_filename,
        )

    def collate_new(self, batch):

        coordinates = torch.cat(
            [
                torch.cat(
                    [
                        torch.LongTensor(datapoint["xyz"].shape[0], 1).fill_(i),
                        datapoint["xyz"].long(),
                    ],
                    1,
                )
                for i, datapoint in enumerate(batch)
            ],
            0,
        )
        points = torch.cat([datapoint["xyz_middle"] for datapoint in batch], 0).to(
            torch.float32
        )
        features = torch.cat([datapoint["features"] for datapoint in batch], 0)
        semantic_labels = torch.cat(
            [datapoint["semantic_labels"] for datapoint in batch], 0
        )
        instance_centers = torch.cat(
            [datapoint["instance_centers"] for datapoint in batch], 0
        ).to(torch.float32)
        instance_pointnum = torch.cat(
            [datapoint["instance_pointnum"] for datapoint in batch], 0
        ).to(torch.int)

        total_inst_num = 0
        batch_instance_labels = []
        batch_offsets = [0]
        for datapoint in batch:

            # They do this because they combine all the scenes in the batch into one vector
            instance_labels = datapoint["instance_labels"]
            instance_labels[
                np.where(datapoint["instance_labels"] != self.ignore_label)
            ] += total_inst_num
            total_inst_num += datapoint["number_of_instances"]
            batch_instance_labels.append(instance_labels)

            batch_offsets.append(batch_offsets[-1] + datapoint["xyz"].shape[0])

        # voxelize
        (
            voxel_coordinates,
            point_to_voxel_map,
            voxel_to_point_map,
        ) = pointgroup_ops.voxelization_idx(coordinates, len(batch), self.mode)

        spatial_shape = (coordinates.max(0)[0][1:] + 1).numpy()
        test_filename = batch[0]["filename"]
        return PointGroupBatch(
            batch_indices=coordinates[:, 0].int(),
            voxel_coordinates=voxel_coordinates,
            point_to_voxel_map=point_to_voxel_map,
            voxel_to_point_map=voxel_to_point_map,
            points=points,
            features=features,
            labels=semantic_labels.long(),
            instance_labels=instance_labels,
            instance_centers=instance_centers,
            instance_pointnum=instance_pointnum,
            offsets=batch_offsets,
            batch_size=len(batch),
            spatial_shape=spatial_shape,
            test_filename=test_filename,
        )

    def __len__(self):
        return super(SpconvDataset, self).__len__()

    def __getitem__(self, id):
        scene = self.scenes[id].load()
        scene = crop_single(scene, self.max_pointcloud_size, self.ignore_label)
        return scene

    def getitem_old(self, id):

        # Limit absolute max size of point cloud
        scene = self.scenes[id].load()
        scene = crop_single(scene, self.max_pointcloud_size, self.ignore_label)

        if not self.is_test:
            scene = crop_single(scene, self.max_npoint, self.ignore_label)

            xyz_middle = transforms.augment_data(scene.points, True, True, True)

            xyz = xyz_middle * self.scale

            ### elastic
            xyz = transforms.elastic_distortion(
                xyz, 6 * self.scale // 50, 40 * self.scale / 50
            )
            xyz = transforms.elastic_distortion(
                xyz, 20 * self.scale // 50, 160 * self.scale / 50
            )

            (
                number_of_instances,
                instance_pointnum,
                instance_centers,
            ) = self.get_instance_centers(
                xyz_middle, scene.instance_labels.astype(np.int32)
            )

        else:

            # Scale
            xyz_middle = scene.points
            xyz = xyz_middle * self.scale

        # Offset points
        xyz -= xyz.min(0)

        features = torch.from_numpy(scene.features) + torch.randn(3) * 0.1
        return {
            "xyz_middle": torch.from_numpy(xyz_middle),
            "xyz": torch.from_numpy(xyz),
            "features": features,
            "semantic_labels": torch.from_numpy(scene.semantic_labels),
            "instance_labels": torch.from_numpy(scene.instance_labels),
            "number_of_instances": number_of_instances,
            "instance_pointnum": torch.tensor(instance_pointnum),
            "instance_centers": torch.from_numpy(instance_centers),
            "filename": scene.name,
        }

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
