import math
import random
import numpy as np
import open3d as o3d

from scipy.spatial import KDTree
from util.types import SceneWithLabels


def crop_multiple(scene: SceneWithLabels, max_npoint: int, ignore_label: int,):

    num_points = scene.points.shape[0]
    if num_points <= max_npoint:
        return [scene]

    num_splits = math.floor(num_points / max_npoint)
    scenes = crop(scene, max_npoint, ignore_label, num_splits=num_splits)
    if not scenes:
        return []
    return scenes


def crop_single(scene: SceneWithLabels, max_npoint: int, ignore_label: int,):
    if scene.points.shape[0] > max_npoint:
        scene = crop(scene, max_npoint, ignore_label)
        scene = scene[0]
    return scene


def crop(
    scene: SceneWithLabels, max_npoint: int, ignore_label: int, num_splits: int = 1
):
    """
    Crop by picking a random point and selecting all
    neighbouring points up to a max number of points
    """

    # Build KDTree
    kd_tree = KDTree(scene.points)

    valid_instance_idx = scene.instance_labels != ignore_label
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
        if current_instances.size == 1 and current_instances[0] == ignore_label:
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
            if old_index != ignore_label:
                instance_indices = np.where(cropped_scene.instance_labels == old_index)
                cropped_scene.instance_labels[instance_indices] = new_index
                new_index += 1

        cropped_scenes.append(cropped_scene)

    return cropped_scenes
