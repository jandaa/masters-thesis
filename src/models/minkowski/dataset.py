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
    MinkowskiPretrainInputNew,
    ImagePretrainInput,
)
from dataloaders.datasets import PretrainDataset, SegmentationDataset


class ImagePretrainDataset(PretrainDataset):
    def __init__(self, scenes, cfg):
        super(ImagePretrainDataset, self).__init__(scenes, cfg)

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

        if scene.points.shape[0] == 0:
            new_ind = random.randint(0, len(self.scenes) - 1)
            return self[new_ind]

        # Normalize image
        image = PIL.Image.fromarray(scene.color_image)

        # Do two sets of augmentations for 2D training
        image1, coords1 = self.image_augmentations(image)
        image2, coords2 = self.image_augmentations(image)

        # # Visualize image
        # image1 = image1.transpose(2, 0) * 255.0
        # image1 = image1.transpose(1, 0)
        # image1 = image1.to(torch.uint8)
        # vis = PIL.Image.fromarray(image1.detach().cpu().numpy())
        # vis.show()

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


class PointContrastPretrainDataset(PretrainDataset):
    def __init__(self, scenes, cfg):
        super(PointContrastPretrainDataset, self).__init__(scenes, cfg)

        color_trans_ratio = 0.05
        color_jitter_std = 0.005
        self.scale_range = (0.9, 1.1)

        ROTATION_AUGMENTATION_BOUND = (
            (-np.pi / 32, np.pi / 32),
            (-np.pi / 32, np.pi / 32),
            (-np.pi, np.pi),
        )

        self.augmentations = transforms.Compose(
            [
                transforms.RandomHorizontalFlip("z", False),
                transforms.ChromaticAutoContrast(),
                transforms.ChromaticTranslation(color_trans_ratio),
                transforms.ChromaticJitter(color_jitter_std),
                transforms.RandomRotateZ(
                    ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND
                ),
            ]
        )

    def collate(self, batch):

        # Points stuff
        coords1_list = [
            frame["discrete_coords"]
            for datapoint in batch
            for frame in datapoint["quantized_frames"]
            if datapoint
        ]
        features1_list = [
            frame["unique_feats"]
            for datapoint in batch
            for frame in datapoint["quantized_frames"]
            if datapoint
        ]
        # coords2_list = [
        #     datapoint["quantized_frames"][1]["discrete_coords"]
        #     for datapoint in batch
        #     if datapoint
        # ]
        # features2_list = [
        #     datapoint["quantized_frames"][1]["unique_feats"]
        #     for datapoint in batch
        #     if datapoint
        # ]

        points1, features1 = ME.utils.sparse_collate(coords1_list, features1_list)
        # points2, features2 = ME.utils.sparse_collate(coords2_list, features2_list)

        # Point to Point map
        point_to_point_map = [
            datapoint["point_to_point_map"] for datapoint in batch if datapoint
        ]

        pretrain_input = MinkowskiPretrainInputNew(
            points1=points1,
            features1=features1.float(),
            points2=None,
            features2=None,
            point_to_pixel_maps1=None,
            point_to_pixel_maps2=None,
            images=None,
            image_coordinates=None,
            point_to_point_map=point_to_point_map,
            batch_size=len(batch),
        )

        return pretrain_input

    def __getitem__(self, index):

        scene = self.scenes[index]

        with scene.open("rb") as scene_pickle:
            scene = pickle.load(scene_pickle)

        if scene.scan_points.shape[0] == 0:
            new_ind = random.randint(0, len(self.scenes) - 1)
            return self[new_ind]

        quantized_frames = []
        random_scale = np.random.uniform(*self.scale_range)
        for frame in [scene, scene]:

            # Extract data
            xyz = np.ascontiguousarray(frame.scan_points[:, 0:3])
            features = torch.from_numpy(frame.scan_point_colors)

            # apply a random scalling
            xyz *= random_scale

            # Unnormalize features for a sec
            features = (features + 1.0) * 127.5

            # Randomly rotate each frame
            xyz = xyz - xyz.mean(0)
            xyz, features, _ = self.augmentations(xyz, features, None)
            xyz = xyz - xyz.mean(0)

            # Renormalize Colour
            features = features / 255.0 - 0.5

            # Voxelize input
            discrete_coords, mapping, inverse_mapping = ME.utils.sparse_quantize(
                coordinates=xyz,
                quantization_size=self.voxel_size,
                return_index=True,
                return_inverse=True,
            )

            unique_feats = features[mapping]

            # Append to quantized frames
            quantized_frames.append(
                {
                    "discrete_coords": discrete_coords,
                    "unique_feats": unique_feats,
                    "inverse_mapping": inverse_mapping,
                }
            )

        # Randomly pick points of original scene
        max_pos = 1000
        point_indices = np.random.choice(
            frame.scan_points.shape[0], max_pos, replace=False
        )

        point_to_point_map = np.array(
            [
                [
                    quantized_frames[0]["inverse_mapping"][point_ind],
                    quantized_frames[1]["inverse_mapping"][point_ind],
                ]
                for point_ind in point_indices
            ]
        )

        return {
            "quantized_frames": quantized_frames,
            "point_to_point_map": point_to_point_map,
        }


class MinkowskiPretrainDataset(PretrainDataset):
    def __init__(self, scenes, cfg):
        super(MinkowskiPretrainDataset, self).__init__(scenes, cfg)

        color_trans_ratio = 0.05
        color_jitter_std = 0.005
        self.scale_range = (0.9, 1.1)

        ROTATION_AUGMENTATION_BOUND = (
            (-np.pi / 32, np.pi / 32),
            (-np.pi / 32, np.pi / 32),
            (-np.pi, np.pi),
        )

        self.augmentations = transforms.Compose(
            [
                transforms.RandomHorizontalFlip("z", False),
                transforms.ChromaticAutoContrast(),
                transforms.ChromaticTranslation(color_trans_ratio),
                transforms.ChromaticJitter(color_jitter_std),
                transforms.RandomRotateZ(
                    ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND
                ),
            ]
        )

        self.coord_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize(256),
                T.CenterCrop(224),
            ]
        )

        self.image_transforms = T.Compose(
            [
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_coords(self, image):
        height = image.shape[0]
        width = image.shape[1]

        # Create coordinate image
        # dims are (2, W, H)
        # where the first axis holds the (x,y) coordinates
        coord_x = np.linspace(0, width - 1, width).reshape(1, 1, -1)
        coord_x = np.repeat(coord_x, height, axis=1)
        coord_y = np.linspace(0, height - 1, height).reshape(1, -1, 1)
        coord_y = np.repeat(coord_y, width, axis=2)

        coord = np.concatenate([coord_x, coord_y], axis=0)
        coord = np.transpose(coord, (1, 2, 0))

        return coord

    def collate(self, batch):

        # Image stuff
        point_to_pixel_maps1 = [
            datapoint["quantized_frames"][0]["point_to_pixel_map"]
            for datapoint in batch
            if datapoint
        ]

        images = [datapoint["image"] for datapoint in batch if datapoint]
        image_coordinates = [
            datapoint["image_coords"] for datapoint in batch if datapoint
        ]
        images = torch.cat(images, 0)

        # Points stuff
        coords1_list = [
            datapoint["quantized_frames"][0]["discrete_coords"]
            for datapoint in batch
            if datapoint
        ]
        features1_list = [
            datapoint["quantized_frames"][0]["unique_feats"]
            for datapoint in batch
            if datapoint
        ]

        points1, features1 = ME.utils.sparse_collate(coords1_list, features1_list)

        pretrain_input = MinkowskiPretrainInputNew(
            points1=points1,
            features1=features1.float(),
            points2=None,
            features2=None,
            point_to_pixel_maps1=point_to_pixel_maps1,
            point_to_pixel_maps2=None,
            images=images,
            image_coordinates=image_coordinates,
            point_to_point_map=None,
            batch_size=len(batch),
        )

        return pretrain_input

    def __getitem__(self, index):

        scene = self.scenes[index]

        with scene.open("rb") as scene_pickle:
            scene = pickle.load(scene_pickle)

        if scene.points.shape[0] == 0:
            new_ind = random.randint(0, len(self.scenes) - 1)
            return self[new_ind]

        # Get image coordinates
        coords = self.get_coords(scene.color_image)
        coords = self.coord_transforms(coords)

        # Transform image
        image = scene.color_image / 255.0
        image = self.coord_transforms(image)
        image = self.image_transforms(image).unsqueeze(dim=0).to(torch.float32)

        # Make sure points fit in range
        x_0 = int(coords[0, 0, 0]) + 1
        y_0 = int(coords[1, 0, 0]) + 1
        x_n = int(coords[0, 0, -1]) - 1
        y_n = int(coords[1, -1, 0]) - 1

        point_indices = np.arange(scene.scan_points.shape[0])
        point_to_pixel_map = np.array(scene.point_to_pixel_map)
        point_to_pixel_map = np.concatenate(
            [point_indices.reshape(-1, 1), point_to_pixel_map], axis=1
        )
        select = (
            (point_to_pixel_map[:, 1] > x_0)
            & (point_to_pixel_map[:, 1] < x_n)
            & (point_to_pixel_map[:, 2] > y_0)
            & (point_to_pixel_map[:, 2] < y_n)
        )

        point_to_pixel_map = point_to_pixel_map[select]

        quantized_frames = []
        random_scale = np.random.uniform(*self.scale_range)
        for frame in [scene]:

            # Extract data
            xyz = np.ascontiguousarray(frame.scan_points[:, 0:3])
            features = torch.from_numpy(frame.scan_point_colors)

            # apply a random scalling
            xyz *= random_scale

            # Unnormalize features for a sec
            features = (features + 1.0) * 127.5

            # Randomly rotate each frame
            xyz = xyz - xyz.mean(0)
            xyz, features, _ = self.augmentations(xyz, features, None)
            xyz = xyz - xyz.mean(0)

            # Renormalize Colour
            features = features / 255.0 - 0.5

            # Voxelize input
            discrete_coords, mapping, inverse_mapping = ME.utils.sparse_quantize(
                coordinates=xyz,
                quantization_size=self.voxel_size,
                return_index=True,
                return_inverse=True,
            )

            unique_feats = features[mapping]

            # Select random points to be used in training
            max_pos = 2000
            point_indices = np.random.choice(
                point_to_pixel_map.shape[0], max_pos, replace=False
            )
            point_to_pixel_map_frame = point_to_pixel_map[point_indices]

            # Map point indices to voxel indices
            point_to_pixel_map_frame[:, 0] = inverse_mapping[
                point_to_pixel_map_frame[:, 0]
            ]

            # Append to quantized frames
            quantized_frames.append(
                {
                    "discrete_coords": discrete_coords,
                    "unique_feats": unique_feats,
                    "point_to_pixel_map": point_to_pixel_map_frame,
                    "inverse_mapping": inverse_mapping,
                }
            )

        # Randomly pick points of original scene
        max_pos = 1000
        point_indices = np.random.choice(
            frame.scan_points.shape[0], max_pos, replace=False
        )

        return {
            "quantized_frames": quantized_frames,
            "image": image,
            "image_coords": coords.detach().cpu().numpy(),
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
                # transforms.Crop(self.max_npoint, self.ignore_label),
                # transforms.RandomDropout(0.2),
                transforms.RandomDropout(0.1),
                transforms.RandomHorizontalFlip("z", False),
                transforms.ChromaticTranslation(color_trans_ratio),
                transforms.ChromaticJitter(color_jitter_std),
                transforms.RandomScale(scale_range),
                transforms.RandomRotate(),
                # transforms.ElasticDistortion(elastic_distortion_params),
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

        CLIP_BOUND = 6  # [-N, N]
        TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))

        color_jitter_std = 0.005
        color_trans_ratio = 0.05
        scale_range = (0.9, 1.1)
        elastic_distortion_params = ((0.2, 0.4), (0.8, 1.6))
        self.augmentations = transforms.Compose(
            [
                # transforms.Crop(self.max_npoint, self.ignore_label),
                transforms.Clip(
                    CLIP_BOUND, TRANSLATION_AUGMENTATION_RATIO_BOUND, self.ignore_label
                ),
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

        # Unnormalize features for a sec
        features = (features + 1.0) * 127.5

        if not self.is_test:
            xyz, features, labels = self.augmentations(xyz, features, labels)

        # Renormalize Colour
        features = features / 255.0 - 0.5

        # Move coordinates back to origin
        xyz = xyz - xyz.min(0)

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


class MinkowskiS3DISDataset(MinkowskiDataset):
    def __init__(self, scenes, cfg, is_test=False):
        super(MinkowskiS3DISDataset, self).__init__(scenes, cfg, is_test=is_test)

        color_jitter_std = 0.005
        color_trans_ratio = 0.05
        scale_range = (0.9, 1.1)

        CLIP_BOUND = 4  # [-N, N]

        # Augmentation arguments
        ROTATION_AUGMENTATION_BOUND = (
            (-np.pi / 32, np.pi / 32),
            (-np.pi / 32, np.pi / 32),
            (-np.pi, np.pi),
        )
        TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))

        self.augmentations = transforms.Compose(
            [
                transforms.Clip(
                    CLIP_BOUND, TRANSLATION_AUGMENTATION_RATIO_BOUND, self.ignore_label
                ),
                transforms.RandomDropout(0.2),
                transforms.RandomHorizontalFlip("z", False),
                transforms.ChromaticAutoContrast(),
                transforms.ChromaticTranslation(color_trans_ratio),
                transforms.ChromaticJitter(color_jitter_std),
                transforms.RandomScale(scale_range),
                transforms.RandomRotateZ(
                    ROTATION_AUGMENTATION_BOUND=ROTATION_AUGMENTATION_BOUND
                ),
            ]
        )
