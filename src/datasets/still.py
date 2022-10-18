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
class StillDataPoint(DataPoint):

    name: Path
    preprocessed_path: Path

    def __post_init__(self):

        # Make directories
        self.preprocessed_path = self.name

        # Files and names
        self.scene_name = self.name.stem
        self.processed_scene = self.name

    @property
    def num_points(self) -> int:
        return 10000

    def is_scene_preprocessed(self):
        return True

    def load(self) -> SceneWithLabels:

        # Load processed scene if already preprocessed
        if not self.is_scene_preprocessed():
            raise RuntimeError(f"Scene {self.room}is not preprocessed")

        test = np.load(str(self.processed_scene))
        points = test[:, 0:3]
        features = test[:, 3:6] / 127.5 - 1.0
        semantic_labels = np.zeros((points.shape[0]))
        instance_labels = np.zeros((points.shape[0]))

        scene = SceneWithLabels(
            name=self.scene_name,
            points=points.astype(np.float32),
            features=features.astype(np.float32),
            semantic_labels=semantic_labels.astype(np.float32),
            instance_labels=instance_labels.astype(np.float32),
        )

        return scene


@dataclass
class StillDataInterface(DataInterface):
    """
    Interface to load required data for a scene
    """

    dataset_dir: Path
    preprocessed_path: Path

    ignore_label: int
    instance_ignore_classes: list

    def __post_init__(self):

        self.instance_categories = [
            label
            for label in self.semantic_categories
            if label not in self.instance_ignore_classes
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

    @property
    def train_data(self) -> list:
        return []

    @property
    def val_data(self) -> list:
        return self.load([])

    @property
    def test_data(self) -> list:
        return self.load([])

    @property
    def pretrain_data(self) -> list:
        return []

    @property
    def pretrain_val_data(self) -> list:
        return []

    def load(self, split, force_reload=False) -> list:
        return [
            StillDataPoint(name=scan, preprocessed_path=self.preprocessed_path)
            for scan in self.preprocessed_path.iterdir()
        ]

    def preprocess(self, datapoint: StillDataPoint) -> None:
        return
