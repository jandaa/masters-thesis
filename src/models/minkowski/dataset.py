import random

import torch
import numpy as np
import MinkowskiEngine as ME

# Sampling
import open3d as o3d
from math import log, e

import dataloaders.transforms as transforms
from models.minkowski.types import MinkowskiInput, MinkowskiPretrainInput
from dataloaders.datasets import PretrainDataset, SegmentationDataset


class MinkowskiPretrainDataset(PretrainDataset):
    def __init__(self, scenes, cfg):
        super(MinkowskiPretrainDataset, self).__init__(scenes, cfg)

        color_jitter_std = 0.05
        self.scale_range = (0.9, 1.1)
        self.augmentations = transforms.Compose(
            [
                transforms.ChromaticJitter(color_jitter_std),
                transforms.RandomRotateZ(),
            ]
        )

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

        # pick matching scenes at random
        frame1 = random.choice(list(scene.matching_frames_map.keys()))
        frame2 = random.choice(scene.matching_frames_map[frame1])

        correspondences = scene.correspondance_map[frame1][frame2]

        frame1 = scene.get_measurement(frame1)
        frame2 = scene.get_measurement(frame2)

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
            discrete_coords, mapping = ME.utils.sparse_quantize(
                coordinates=xyz, quantization_size=self.voxel_size, return_index=True
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

        # visualize_correspondances(quantized_frames, correspondences)
        return {
            "correspondences": correspondences,
            "quantized_frames": quantized_frames,
        }


class MinkowskiEntropyPretrainDataset(PretrainDataset):
    def __init__(self, scenes, cfg):
        super(MinkowskiEntropyPretrainDataset, self).__init__(scenes, cfg)

        color_jitter_std = 0.05
        self.scale_range = (0.9, 1.1)
        self.augmentations = transforms.Compose(
            [
                transforms.ChromaticJitter(color_jitter_std),
                transforms.RandomRotateZ(),
            ]
        )

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

    def entropy(self, feature, base=None):
        """Compute entropy of a feature vector."""

        feature = feature / feature.sum()

        entropy = 0

        # Compute entropy
        base = e if base is None else base
        for i in feature:
            if i != 0:
                entropy -= i * log(i, base)

        return entropy

    def __getitem__(self, index):

        scene = self.scenes[index].load_measurements()

        # pick matching scenes at random
        frame1 = random.choice(list(scene.matching_frames_map.keys()))
        frame2 = random.choice(scene.matching_frames_map[frame1])

        correspondences = scene.correspondance_map[frame1][frame2]

        frame1 = scene.get_measurement(frame1)
        frame2 = scene.get_measurement(frame2)

        quantized_frames = []
        random_scale = np.random.uniform(*self.scale_range)
        for frame in [frame1, frame2]:

            # Extract data
            xyz = np.ascontiguousarray(frame.points)
            features = torch.from_numpy(frame.point_colors)

            # Compute FPFH features
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=20 * self.voxel_size, max_nn=500
                )
            )

            search_param = o3d.geometry.KDTreeSearchParamHybrid(
                radius=30 * self.voxel_size, max_nn=200
            )
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)
            fpfh = fpfh.data.T

            entropies = np.array([self.entropy(feature) for feature in fpfh])
            entropies = entropies - entropies.min()
            entropies = entropies / entropies.max()

            entropies[np.where(entropies < 0.4)[0]] = 0.0

            # apply a random scalling
            xyz *= random_scale

            # Randomly rotate each frame
            xyz = xyz - xyz.mean(0)
            xyz, features, _ = self.augmentations(xyz, features, None)

            # Voxelize input
            discrete_coords, mapping = ME.utils.sparse_quantize(
                coordinates=xyz, quantization_size=self.voxel_size, return_index=True
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
                    "entropies": entropies,
                }
            )

        # Remap the correspondances into voxel world
        mapping1 = quantized_frames[0]["mapping"]
        mapping2 = quantized_frames[1]["mapping"]
        entropies1 = quantized_frames[0]["entropies"]
        correspondences = [
            (mapping1[k], mapping2[v], entropies1[k])
            for k, v in correspondences.items()
            if k in mapping1.keys() and v in mapping2.keys()
        ]

        # visualize_correspondances(quantized_frames, correspondences)
        return {
            "correspondences": correspondences,
            "quantized_frames": quantized_frames,
        }


class MinkowskiDataset(SegmentationDataset):
    def __init__(self, scenes, cfg, is_test=False):
        super(MinkowskiDataset, self).__init__(scenes, cfg, is_test=is_test)

        color_jitter_std = 0.05
        color_trans_ratio = 0.1
        scale_range = (0.8, 1.2)
        elastic_distortion_params = ((0.2, 0.4), (0.8, 1.6))
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
            batch_size=len(batch),
            test_filename=batch[0].test_filename,
        )

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
            quantization_size=self.voxel_size,
        )

        return MinkowskiInput(
            points=coords,
            features=feats,
            labels=labels,
            batch_size=1,
            test_filename=scene.name,
        )
