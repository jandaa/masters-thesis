import torch
import numpy as np

from dataloaders.datasets import SegmentationDataset
import dataloaders.transforms as transforms
from models.pointgroup.types import PointGroupBatch
from packages.pointgroup_ops.functions import pointgroup_ops


class SpconvDataset(SegmentationDataset):
    def __init__(self, scenes, cfg, is_test=False):
        super(SpconvDataset, self).__init__(scenes, cfg, is_test=is_test)

        self.max_npoint = min(self.max_npoint, self.max_pointcloud_size)
        color_jitter_std = 0.05
        color_trans_ratio = 0.1
        scale_range = (0.8, 1.2)
        elastic_distortion_params = ((0.2, 0.4), (0.8, 1.6))

        if self.is_test:
            self.augmentations = transforms.Compose(
                [
                    transforms.Crop(self.max_pointcloud_size, self.ignore_label),
                ]
            )
        else:
            self.augmentations = transforms.Compose(
                [
                    transforms.Crop(self.max_npoint, self.ignore_label),
                    transforms.RandomDropout(0.2),
                    transforms.RandomHorizontalFlip("z", False),
                    transforms.ChromaticTranslation(color_trans_ratio),
                    transforms.ChromaticJitter(color_jitter_std),
                    transforms.RandomScale(scale_range),
                    transforms.RandomRotate(),
                    transforms.ElasticDistortion(elastic_distortion_params),
                ]
            )

    def collate(self, batch):

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

        instance_labels = torch.cat(batch_instance_labels, 0).to(torch.int)

        # voxelize
        (
            voxel_coordinates,
            point_to_voxel_map,
            voxel_to_point_map,
        ) = pointgroup_ops.voxelization_idx(coordinates, len(batch), self.mode)

        spatial_shape = (coordinates.max(0)[0][1:] + 1).numpy()
        spatial_shape = spatial_shape.clip(128, None)
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

    def __getitem__(self, id):

        scene = self.scenes[id].load()
        labels = np.array([scene.semantic_labels, scene.instance_labels]).T

        xyz_middle, features, labels = self.augmentations(
            scene.points, scene.features, labels
        )
        xyz = xyz_middle * self.scale
        xyz -= xyz.min(0)

        semantic_labels = labels[:, 0]
        instance_labels = labels[:, 1].astype(np.int32)

        (
            number_of_instances,
            instance_pointnum,
            instance_centers,
        ) = self.get_instance_centers(xyz_middle, instance_labels)

        return {
            "xyz_middle": torch.from_numpy(xyz_middle),
            "xyz": torch.from_numpy(xyz),
            "features": torch.from_numpy(features),
            "semantic_labels": torch.from_numpy(semantic_labels),
            "instance_labels": torch.from_numpy(instance_labels),
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
