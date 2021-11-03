import logging
import random

import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME

from util.types import PointGroupBatch, PretrainInput, MinkowskiInput
from packages.pointgroup_ops.functions import pointgroup_ops
from util.utils import get_random_colour
from dataloaders.transforms import augment_data, elastic_distortion
import dataloaders.transforms as transforms

from dataloaders.crop import crop, crop_single, crop_multiple

from spconv.utils import VoxelGeneratorV2


log = logging.getLogger(__name__)

color_jitter_std = 0.05
color_trans_ratio = 0.1
elastic_distortion_params = ((0.2, 0.4), (0.8, 1.6))
augmentation_transforms = transforms.Compose(
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

pointgroup_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip("z", False),
        transforms.ChromaticTranslation(color_trans_ratio),
        transforms.ChromaticJitter(color_jitter_std),
        transforms.RandomRotate(),
        transforms.ElasticDistortion(elastic_distortion_params),
    ]
)


def minkowski_merge(self, id, scenes, crop=False, is_test=False):

    coords_list = []
    features_list = []
    labels_list = []

    for i, idx in enumerate(id):

        scene = scenes[idx]
        if not self.are_scenes_preloaded:
            scene = scene.load()

            if crop:
                scene = crop_single(scene, self.max_npoint, self.ignore_label)
                if not scene:
                    continue

        # Make sure the scene is not way too big
        scene = crop_single(scene, self.max_pointcloud_size, self.ignore_label)

        xyz = np.ascontiguousarray(scene.points)
        features = torch.from_numpy(scene.features)
        labels = scene.semantic_labels

        # Apply transformations
        if not is_test:
            xyz, features, labels = augmentation_transforms(xyz, features, labels)

        coords, feats, labels = ME.utils.sparse_quantize(
            xyz,
            features=features,
            labels=labels,
            quantization_size=(1 / self.scale),
        )

        coords_list.append(coords)
        features_list.append(feats)
        labels_list.append(labels)

    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        coords_list, features_list, labels_list
    )

    if is_test:
        test_filename = scenes[id[0]].scene_name
    else:
        test_filename = None

    return MinkowskiInput(
        points=coordinates_batch,
        features=features_batch.float(),
        labels=labels_batch.int(),
        test_filename=test_filename,
    )


def minkowski_pretrain_merge(self, id):

    batch_offsets = []
    batch_correspondances = []
    coords_list = []
    features_list = []
    labels_list = []

    for i, idx in enumerate(id):

        scene = self.pretrain_data[idx]
        if not self.are_scenes_preloaded:
            scene = scene.load_measurements()

        if not scene.matching_frames_map:
            continue

        # pick matching scenes at random
        frame1 = random.choice(list(scene.matching_frames_map.keys()))
        frame2 = random.choice(scene.matching_frames_map[frame1])

        correspondances = scene.correspondance_map[frame1][frame2]

        frame1 = scene.get_measurement(frame1)
        frame2 = scene.get_measurement(frame2)

        # select a max number of correspondances
        keys = list(correspondances.keys())
        if len(keys) > 4092:
            keys = random.sample(keys, 4092)
            correspondances = {key: correspondances[key] for key in keys}

        batch_correspondances.append(correspondances)

        for frame in [frame1, frame2]:

            # Randomly rotate each frame
            xyz_middle = frame.points - frame.points.mean(0)
            xyz_middle = augment_data(xyz_middle, rot=True)

            ### scale
            xyz = xyz_middle

            ### offset
            xyz -= xyz.min(0)

            features = torch.from_numpy(scene.features) + torch.randn(3) * 0.1

            discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
                xyz,
                features=features,
                labels=scene.semantic_labels,
                quantization_size=(1 / self.scale),
            )

            coords_list.append(discrete_coords)
            features_list.append(unique_feats)
            labels_list.append(unique_labels)
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            # self.visualize_partitions(xyz)

    bcoords = ME.utils.batched_coordinates(coords_list)
    feats_batch = torch.from_numpy(np.concatenate(features_list, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels_list, 0)).int()

    pretrain_input = MinkowskiInput(
        points=bcoords,
        features=feats_batch,
        labels=labels_batch,
        correspondances=batch_correspondances,
        batch_size=2 * len(id),
        offsets=batch_offsets,
    )

    # Verify that point correspondences are correct
    # for debugging
    # self.visualize_correspondances(pretrain_input)

    return pretrain_input


def pointgroup_merge(self, id, scenes, crop=False, is_test=False, test_split="val"):

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
                scene = crop_single(scene, self.max_npoint, self.ignore_label)
                if not scene:
                    continue

        if is_test:

            # Make sure the scene is not way too big
            scene = crop_single(scene, self.max_pointcloud_size, self.ignore_label)

            xyz_middle = augment_data(scene.points, False, True, True)

            xyz = xyz_middle * self.scale

            xyz -= xyz.min(0)

            batch_features.append(torch.from_numpy(scene.features))

            semantic_labels = scene.semantic_labels
            instance_labels = scene.instance_labels

        else:

            # apply data augmentations
            labels = np.array([scene.semantic_labels, scene.instance_labels]).T
            xyz_middle, features, labels = pointgroup_transforms(
                scene.points, scene.features, labels
            )

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            batch_features.append(torch.from_numpy(features))

            semantic_labels = labels[:, 0]
            instance_labels = labels[:, 1]

        # else:

        #     ### jitter / flip x / rotation
        #     xyz_middle = augment_data(scene.points, True, True, True)

        #     ### scale
        #     xyz = xyz_middle * self.scale

        #     ### elastic
        #     xyz = elastic_distortion(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
        #     xyz = elastic_distortion(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

        #     ### offset
        #     xyz -= xyz.min(0)

        #     rgb = scene.features
        #     # semantic_labels = scene.semantic_labels
        #     # instance_labels = scene.instance_labels

        #     batch_features.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)

        if are_labels_available:
            ### get instance information
            (
                number_of_instances,
                instance_pointnum,
                instance_centers,
            ) = self.get_instance_centers(xyz_middle, instance_labels.astype(np.int32))

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

        # # TODO: Continue from here!!
        # max_num_points_per_voxel = 5
        # voxel_generator = VoxelGeneratorV2(
        #     [0.05, 0.05, 0.05],
        #     [0.0, 0.0, 0.0, 20.0, 20.0, 10.0],
        #     max_num_points_per_voxel,
        # )
        # test = np.array([[20, 0, 0], [3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [2, 3, 1]])
        # res = voxel_generator.generate(xyz)

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


def pointgroup_pretrain_merge(self, id):

    batch_coordinates = []
    batch_points = []
    batch_features = []

    batch_offsets = [0]

    batch_correspondances = []

    for i, idx in enumerate(id):

        scene = self.pretrain_data[idx]
        if not self.are_scenes_preloaded:
            scene = scene.load_measurements()

        if not scene.matching_frames_map:
            continue

        # pick matching scenes at random
        frame1 = random.choice(list(scene.matching_frames_map.keys()))
        frame2 = random.choice(scene.matching_frames_map[frame1])

        correspondances = scene.correspondance_map[frame1][frame2]

        frame1 = scene.get_measurement(frame1)
        frame2 = scene.get_measurement(frame2)

        # select a max number of correspondances
        keys = list(correspondances.keys())
        if len(keys) > 4092:
            keys = random.sample(keys, 4092)
            correspondances = {key: correspondances[key] for key in keys}

        batch_correspondances.append(correspondances)

        for frame in [frame1, frame2]:

            # Randomly rotate each frame
            xyz_middle = frame.points - frame.points.mean(0)
            xyz_middle = augment_data(xyz_middle, rot=True)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

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

            # self.visualize_partitions(xyz)

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


def visualize_partitions(self, xyz):
    """Visualize partitions of shape context."""

    from util.shape_context import ShapeContext

    partitioner = ShapeContext()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    partition = partitioner.compute_partitions(xyz)
    partition = partition[1000]
    colours = np.zeros(xyz.shape)
    for partition_id in range(partitioner.partitions):
        mask_q = partition == partition_id
        # mask_q.fill_diagonal_(True)
        colours[mask_q] = get_random_colour()

    pcd.colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([pcd])


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
