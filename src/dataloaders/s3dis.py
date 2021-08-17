import logging
import json

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import torch
import numpy as np

from util.types import DataInterface, DataPoint, SceneWithLabels

log = logging.getLogger(__name__)


@dataclass
class S3DISDataPoint(DataPoint):

    room: Path
    force_reload: bool
    label_to_index_map: dict
    ignore_label: int
    ignore_classes: list

    def __post_init__(self):

        # Files and names
        self.scene_name = f"{self.room.parent.name}_{self.room.name}"
        self.processed_scene = self.room / (self.room.name + ".pth")
        self.scene_details_file = self.room / (self.room.name + "_details.json")

    @property
    def num_points(self) -> int:
        if not self.is_scene_preprocessed(force_reload=False):
            self.preprocess()

        with self.scene_details_file.open() as fp:
            details = json.loads(fp.read())

        return details["num_points"]

    def is_scene_preprocessed(self, force_reload):
        return (
            self.processed_scene.exists()
            and self.scene_details_file.exists()
            and not force_reload
            and not self.force_reload
        )

    def preprocess(self, force_reload=False, crop_callback=None) -> None:
        if self.is_scene_preprocessed(force_reload):
            return

        log.info(f"Loading scene: {self.scene_name}")

        points = []
        features = []
        semantic_labels = []
        instance_labels = []

        # Load points of each object instance making up the scene
        annotations_dir = self.room / "Annotations"
        instance_counter = 0
        for object in annotations_dir.iterdir():

            # Ignore any hidden files
            if object.name.startswith("."):
                continue

            class_name = object.name.split("_")[0]

            # Ignore certain classes for instance segmentation
            if class_name in self.ignore_classes:
                instance_label = self.ignore_label
            else:
                instance_label = instance_counter
                instance_counter += 1

            semantic_label = self.label_to_index_map[class_name]
            object_points = np.loadtxt(str(object), delimiter=" ")
            object_points = object_points.astype(np.float32)

            num_points = object_points.shape[0]
            points.append(object_points[:, 0:3])
            features.append(object_points[:, 3:6])
            semantic_labels.append(np.ones((num_points), dtype=np.int) * semantic_label)
            instance_labels.append(np.ones((num_points), dtype=np.int) * instance_label)

        # Concatonate to make into full vectors
        points = np.concatenate(points, 0)
        features = np.concatenate(features, 0)
        semantic_labels = np.concatenate(semantic_labels, None)
        instance_labels = np.concatenate(instance_labels, None)

        # Zero and normalize inputs
        points -= points.mean(0)
        features = features / 127.5 - 1

        # Save data to avoid re-computation in the future
        log.info(f"Saving scene: {self.scene_name}")
        torch.save(
            (points, features, semantic_labels, instance_labels),
            self.processed_scene,
        )

        details = {"num_points": points.shape[0]}
        with self.scene_details_file.open(mode="w") as fp:
            json.dump(details, fp)

    def load(self, force_reload=False) -> SceneWithLabels:

        # Load processed scene if already preprocessed
        if not self.is_scene_preprocessed(force_reload):
            return self.preprocess(force_reload=force_reload)

        try:
            (points, features, semantic_labels, instance_labels) = torch.load(
                str(self.processed_scene)
            )

        except:
            log.info(f"Error loading {self.scene_name}. Trying to force reload.")
            return self.load(force_reload=True)

        scene = SceneWithLabels(
            name=self.scene_name,
            points=points.astype(np.float32),
            features=features.astype(np.float32),
            semantic_labels=semantic_labels.astype(np.float32),
            instance_labels=instance_labels.astype(np.float32),
        )

        return scene


@dataclass
class S3DISDataInterface(DataInterface):
    """
    Interface to load required data for a scene
    """

    dataset_dir: Path

    # Split is done using areas
    train_split: list
    val_split: list
    test_split: list

    force_reload: bool = False

    def __post_init__(self):

        # Ignore stuff classes
        # TODO: Move this to config file
        self.ignore_label = -100
        self.ignore_classes = ["wall", "floor", "ceiling"]

        self.instance_categories = [
            label
            for label in self.semantic_categories
            if label not in self.ignore_classes
        ]

        self.label_to_index_map = defaultdict(
            lambda: self.ignore_label,
            {
                label_name: index
                for index, label_name in enumerate(self.semantic_categories)
            },
        )
        self.index_to_label_map = {
            index: label_name for label_name, index in self.label_to_index_map.items()
        }

        self.fix_any_errors()

    def fix_any_errors(self):
        """Fix any errors found in the original files"""

        annotation = self.dataset_dir / "Area_5/office_19/Annotations/ceiling_1.txt"
        lines = annotation.open("r").readlines()
        lines[323473] = (
            lines[323473]
            .encode("unicode-escape")
            .decode()
            .replace("\\x1", "")
            .replace("\\n", "\n")
        )
        annotation.open("w").writelines(lines)

    @property
    def train_data(self) -> list:
        return self._load(self.train_split)

    @property
    def val_data(self) -> list:
        return self._load(self.val_split)

    @property
    def test_data(self) -> list:
        return self._load(self.test_split)

    def _get_rooms(self, areas) -> list:
        return [
            room
            for area in areas
            for room in (self.dataset_dir / (f"Area_{area}")).iterdir()
            if room.is_dir()
        ]

    def _load(self, split, force_reload=False) -> list:
        return [
            S3DISDataPoint(
                room=room,
                force_reload=force_reload,
                label_to_index_map=self.label_to_index_map,
                ignore_label=self.ignore_label,
                ignore_classes=self.ignore_classes,
            )
            for room in self._get_rooms(split)
        ]
