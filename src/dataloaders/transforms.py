# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import math
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
from scipy.spatial import KDTree


def augment_data(xyz, jitter=False, flip=False, rot=False):
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


def elastic_distortion(x, granularity, magnitude):
    blurs = [
        np.ones(shape).astype("float32") / 3
        for shape in [(3, 1, 1), (1, 3, 1), (1, 1, 3)]
    ]

    # Select random noise for each voxel of bounding box
    bounding_box = np.abs(x).max(0).astype(np.int32) // granularity + 3
    noise = [np.random.randn(*list(bounding_box)).astype("float32") for _ in range(3)]

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
        scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
        for n in noise
    ]

    return x + np.hstack([i(x)[:, None] for i in interp]) * magnitude


# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Feature transformations
##############################
class ChromaticTranslation(object):
    """Add random color to the image, input must be an array in [0,255] or a PIL image"""

    def __init__(self, trans_range_ratio=1e-1):
        """
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        """
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 2 * self.trans_range_ratio
            # tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feats[:, :3] = np.clip(feats[:, :3] + tr, -0.5, 0.5)
        return coords, feats, labels


class ChromaticAutoContrast(object):
    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords, feats, labels):
        if random.random() < 0.2:
            # mean = np.mean(feats, 0, keepdims=True)
            # std = np.std(feats, 0, keepdims=True)
            # lo = mean - std
            # hi = mean + std
            lo = feats[:, :3].min(0, keepdims=True)
            hi = feats[:, :3].max(0, keepdims=True)
            assert hi.max() > 1, f"invalid color value. Color is supposed to be [0-255]"

            scale = 255 / (hi - lo)

            contrast_feats = (feats[:, :3] - lo) * scale

            blend_factor = (
                random.random() if self.randomize_blend_factor else self.blend_factor
            )
            feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return coords, feats, labels


class ChromaticJitter(object):
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            noise = np.random.randn(feats.shape[0], 3)
            noise *= self.std
            feats[:, :3] = np.clip(feats[:, :3] + noise, -0.5, 0.5)
        return coords, feats, labels


##############################
# Coordinate transformations
##############################
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < self.dropout_ratio:
            N = len(coords)
            if N < 10:
                return coords, feats, labels
            inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
            return coords[inds], feats[inds], labels[inds]
        return coords, feats, labels


class RandomScale(object):
    def __init__(self, scale_range=(0.8, 1.2)):
        """
        Randomly scale points.
        scale_range: (min, max) of possible scale
        """
        self.scale_range = scale_range

    def __call__(self, coords, feats, labels):
        coords *= np.random.uniform(*self.scale_range)
        return coords, feats, labels


class RandomRotate(object):
    def __init__(self):
        """
        Randomly rotate the scene
        """

    def __call__(self, coords, feats, labels):

        theta = np.random.rand() * 2 * math.pi
        rot = [
            [math.cos(theta), math.sin(theta), 0],
            [-math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ]
        coords = np.matmul(coords, rot)
        return coords, feats, labels


class RandomHorizontalFlip(object):
    def __init__(self, upright_axis, is_temporal):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {"x": 0, "y": 1, "z": 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = np.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return coords, feats, labels


class ElasticDistortion:
    def __init__(self, distortion_params):
        self.distortion_params = distortion_params

    def elastic_distortion(self, coords, feats, labels, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.

        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=0, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords, feats, labels

    def __call__(self, coords, feats, labels):
        if self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    coords, feats, labels = self.elastic_distortion(
                        coords, feats, labels, granularity, magnitude
                    )
        return coords, feats, labels


class Crop(object):
    def __init__(self, max_npoint, ignore_label):
        """
        crop point cloud if beyond max size
        """
        self.max_npoint = max_npoint
        self.ignore_label = ignore_label

    def __call__(self, coords, feats, labels):
        """
        Crop by picking a random point and selecting all
        neighbouring points up to a max number of points
        """

        # Check if already below size
        if coords.shape[0] < self.max_npoint:
            return coords, feats, labels

        # Build KDTree
        kd_tree = KDTree(coords)

        instance_labels = labels[:, 1]

        valid_instance_idx = instance_labels != self.ignore_label
        unique_instance_labels = np.unique(instance_labels[valid_instance_idx])

        if unique_instance_labels.size == 0:
            return False

        # Randomly select a query point
        query_instance = np.random.choice(unique_instance_labels)
        query_points = coords[instance_labels == query_instance]
        query_point_ind = random.randint(0, query_points.shape[0] - 1)
        query_point = query_points[query_point_ind]

        # select subset of neighbouring points from the random center point
        [_, idx] = kd_tree.query(query_point, k=self.max_npoint)

        # Make sure there is at least one instance in the scene
        current_instances = np.unique(instance_labels[idx])
        if current_instances.size == 1 and current_instances[0] == self.ignore_label:
            raise RuntimeError("No instances in scene")

        coords = coords[idx]
        feats = feats[idx]
        semantic_labels = labels[idx, 0]
        instance_labels = labels[idx, 1]

        # Remap instance numbers
        instance_ids = np.unique(instance_labels)
        new_index = 0
        for old_index in instance_ids:
            if old_index != self.ignore_label:
                instance_indices = np.where(instance_labels == old_index)
                instance_labels[instance_indices] = new_index
                new_index += 1

        labels = np.array([semantic_labels, instance_labels]).T

        return coords, feats, labels


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
