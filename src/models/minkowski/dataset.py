import random
import pickle
import torch
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data.dataset import Dataset
import PIL

# Sampling
import open3d as o3d

# Transforms
from torchvision import transforms as T
from dataloaders import transform_coord
from dataloaders import transforms

from scipy.spatial import KDTree

from models.minkowski.types import (
    MinkowskiInput,
    MinkowskiPretrainInput,
    ImagePretrainInput,
)
from dataloaders.datasets import PretrainDataset, SegmentationDataset


class ImagePretrainDataset(Dataset):
    def __init__(self, scenes, cfg):
        super(ImagePretrainDataset, self).__init__()

        self.scenes = scenes
        image_size = 224
        crop = 0.2
        self.image_augmentations = transform_coord.Compose(
            [
                transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.0)),
                transform_coord.RandomHorizontalFlipCoord(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([transforms.GaussianBlur([0.1, 2.0])], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def collate(self, batch):

        # Collate images
        images1 = torch.cat([datapoint["image1"] for datapoint in batch], 0)
        images2 = torch.cat([datapoint["image2"] for datapoint in batch], 0)
        coords1 = torch.vstack([datapoint["coords1"] for datapoint in batch])
        coords2 = torch.vstack([datapoint["coords2"] for datapoint in batch])

        # correspondences = torch.cat([datapoint["correspondences"] for datapoint in batch], 0)
        correspondences = [
            datapoint["correspondences"] for datapoint in batch if datapoint
        ]

        pretrain_input = ImagePretrainInput(
            images1=images1,
            images2=images2,
            coords1=coords1,
            coords2=coords2,
            correspondences=correspondences,
            batch_size=len(images1),
        )

        return pretrain_input

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):

        scene = self.scenes[index]

        with scene.open("rb") as scene_pickle:
            scene = pickle.load(scene_pickle)

        # Normalize image
        image = PIL.Image.fromarray(scene.color_image)

        # Do two sets of augmentations
        # self.image_augmentations(image).unsqueeze(dim=0).to(torch.float32)
        image1, coords1 = self.image_augmentations(image)
        image2, coords2 = self.image_augmentations(image)

        # visualize_image(image1)

        # Generate a mapping
        coords1 = coords1.view(2, -1).T.detach().numpy()
        coords2 = coords2.view(2, -1).T.detach().numpy()

        kdtree_1 = KDTree(coords1)
        kdtree_2 = KDTree(coords2)

        correspondences = kdtree_1.query_ball_tree(kdtree_2, 1.5, p=2)
        correspondences = [
            [ind, matches[0]] for ind, matches in enumerate(correspondences) if matches
        ]
        correspondences = torch.tensor(correspondences, dtype=torch.long)

        return {
            "image1": image1.unsqueeze(dim=0),
            "image2": image2.unsqueeze(dim=0),
            "coords1": torch.from_numpy(coords1).unsqueeze(dim=0),
            "coords2": torch.from_numpy(coords2).unsqueeze(dim=0),
            "correspondences": correspondences,
        }


def visualize_mapping(image, image1, image2, coords1, coords2, correspondances):
    import math

    image = np.array(image)
    image = np.zeros(image.shape, dtype=np.uint8)

    # Verify the correspondances
    # by swapping raw pixel values and see if image makes sense
    for ind, matches in enumerate(correspondances):
        if matches:

            # Get rgb in cropped image
            pixel = matches[0]
            y = pixel % 224
            x = math.floor(pixel / 224)

            pixel = coords2[pixel].astype("int")
            image[pixel[1], pixel[0], :] = image2[:, x, y] * 255

            pixel = ind
            y = pixel % 224
            x = math.floor(pixel / 224)

            pixel = coords1[pixel].astype("int")
            image[pixel[1], pixel[0], :] = image1[:, x, y] * 255

    # # # Visualize image
    # image2 = image2.transpose(2, 0) * 255.0
    # image2 = image2.transpose(1, 0)
    # image2 = image2.to(torch.uint8)
    # vis = PIL.Image.fromarray(image2.detach().cpu().numpy())
    # vis.show()

    # Visualize image
    vis = PIL.Image.fromarray(image)
    vis.show()


def visualize_image(image):

    # Visualize image
    image = image.transpose(2, 0) * 255.0
    image = image.transpose(1, 0)
    image = image.to(torch.uint8)
    vis = PIL.Image.fromarray(image.detach().cpu().numpy())
    vis.show()


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

        self.image_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

        # Collate images
        images = [
            frame["image"]
            for datapoint in batch
            for frame in datapoint["quantized_frames"]
        ]
        images = torch.cat(images, 0)

        pretrain_input = MinkowskiPretrainInput(
            points=coordinates_batch,
            features=features_batch.float(),
            images=images,
            correspondences=correspondences,
            batch_size=2 * len(batch),
        )

        return pretrain_input

    def __getitem__(self, index):

        scene = self.scenes[index]

        with scene.open("rb") as scene_pickle:
            scene = pickle.load(scene_pickle)

        if scene.points.shape[0] == 0:
            new_ind = random.randint(0, len(self.scenes))
            return self[new_ind]

        quantized_frames = []
        random_scale = np.random.uniform(*self.scale_range)
        for frame in [scene]:

            image = frame.color_image / 255.0
            image = self.image_transforms(image).unsqueeze(dim=0).to(torch.float32)

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
                    "image": image,
                }
            )

        # # Randomly pick points as correspondances
        # max_pos = min(2024, scene.points.shape[0])
        # point_indices = np.random.choice(scene.points.shape[0], max_pos, replace=False)

        # # Remap the correspondances into voxel world
        # mapping1 = quantized_frames[0]["mapping"]
        # mapping2 = quantized_frames[1]["mapping"]
        # correspondences = [
        #     {
        #         "frame1": {
        #             "voxel_inds": mapping1[point_ind],
        #         },
        #         "frame2": {
        #             "voxel_inds": mapping2[point_ind],
        #         },
        #     }
        #     for point_ind in point_indices
        #     if point_ind in mapping1.keys() and point_ind in mapping2.keys()
        # ]

        # visualize_mapping(
        #     quantized_frames[0]["discrete_coords"],
        #     quantized_frames[1]["discrete_coords"],
        #     correspondences,
        # )

        return {
            "correspondences": None,
            "quantized_frames": quantized_frames,
        }


import open3d as o3d
from matplotlib import pyplot as plt


def get_color_map(x):
    colours = plt.cm.Spectral(x)
    return colours[:, :3]


def visualize_mapping(points1, points2, correspondences):
    points1 = points1.detach().cpu().numpy()
    points2 = points2.detach().cpu().numpy()

    # offset points1
    points1 += np.array([0, 0, 100])

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)

    point_correspondances = [
        (match["frame1"]["voxel_inds"], match["frame2"]["voxel_inds"])
        for match in correspondences
    ]

    lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        pcd1, pcd2, point_correspondances
    )

    o3d.visualization.draw_geometries([pcd1, pcd2, lines])


class MinkowskiFrameDataset(SegmentationDataset):
    def __init__(self, scenes, cfg, is_test=False):
        super(MinkowskiFrameDataset, self).__init__(scenes, cfg, is_test=True)

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

        self.is_test = is_test
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

    def __getitem__(self, index):

        # Limit absolute max size of point cloud
        scene = self.scenes[index]

        with scene.open("rb") as scene_pickle:
            scene = pickle.load(scene_pickle)

        if scene.points.shape[0] == 0:
            new_ind = random.randint(0, len(self.scenes) - 1)
            return self[new_ind]

        xyz = np.ascontiguousarray(scene.points)
        features = torch.from_numpy(scene.point_colors)
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
            test_filename="frame" + str(index),
        )


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
