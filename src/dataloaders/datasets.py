import math

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import MinkowskiEngine as ME

import dataloaders.transforms as transforms
from dataloaders.crop import crop_single

from util.types import MinkowskiInput

color_jitter_std = 0.05
color_trans_ratio = 0.1
elastic_distortion_params = ((0.2, 0.4), (0.8, 1.6))
augmentations = transforms.Compose(
    [
        transforms.RandomDropout(0.2),
        transforms.RandomHorizontalFlip("z", False),
        transforms.ChromaticTranslation(color_trans_ratio),
        transforms.ChromaticJitter(color_jitter_std),
        transforms.RandomScale(),
        transforms.RandomRotate(),
        transforms.ElasticDistortion(elastic_distortion_params),
    ]
)


class SegmentationDataset(Dataset):
    """Base segmentation dataset"""

    def __init__(self, scenes, cfg, is_test=False):
        Dataset.__init__(self)

        self.cfg = cfg
        self.num_workers = cfg.model.train.train_workers

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


class MinkowskiDataset(Dataset):
    def __init__(self, scenes, cfg, is_test=False, augmentations=augmentations):
        super(MinkowskiDataset, self).__init__(scenes, cfg, is_test=is_test)
        self.augmentations = augmentations

    @classmethod
    def collate(batch):
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
        return len(super(self))

    def __getitem__(self, id):

        scene = self.scenes[id]
        test_filename = scene.test_filename
        if not self.are_scenes_preloaded:
            scene = scene.load()

            if not self.is_test:
                scene = crop_single(scene, self.max_npoint, self.ignore_label)

        # Limit absolute max size of point cloud
        scene = crop_single(scene, self.max_pointcloud_size, self.ignore_label)

        xyz = np.ascontiguousarray(scene.points)
        features = torch.from_numpy(scene.features)
        labels = scene.semantic_labels

        if self.augmentations:
            xyz, features, labels = self.augmentations(xyz, features, labels)

        coords, feats, labels = ME.utils.sparse_quantize(
            xyz,
            features=features,
            labels=labels,
            quantization_size=(1 / self.scale),
        )

        return MinkowskiInput(
            points=coords,
            features=feats.float(),
            labels=labels.int(),
            test_filename=test_filename,
        )
