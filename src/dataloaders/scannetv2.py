import logging
import csv
import json

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

from plyfile import PlyData

import torch
import numpy as np

from util.types import DataInterface, SceneWithLabels

log = logging.getLogger(__name__)


@dataclass
class ScannetDataInterface(DataInterface):
    """
    Interface to load required data for a scene
    """

    scans_dir: Path
    train_split: list
    val_split: list
    test_split: list

    # Constants
    raw_labels_filename: str = "../scannetv2-labels.combined.tsv"
    mesh_file_extension: str = "_vh_clean_2.ply"
    labels_file_extension: str = "_vh_clean_2.labels.ply"
    segment_file_extension: str = "_vh_clean_2.0.010000.segs.json"
    instances_file_extension: str = ".aggregation.json"
    ignore_label: int = -100
    ignore_classes: list = field(default_factory=lambda: ["wall", "floor"])
    force_reload: bool = False

    # Default categories
    semantic_categories: list = field(
        default_factory=lambda: scannet_semantic_categories
    )

    @property
    def train_data(self) -> list:
        return [self._load(scene) for scene in self.train_split]

    @property
    def val_data(self) -> list:
        return [self._load(scene) for scene in self.val_split]

    @property
    def test_data(self) -> list:
        return [self._load(scene) for scene in self.test_split]

    @property
    def _required_extensions(self):
        return [
            self.mesh_file_extension,
            self.labels_file_extension,
            self.segment_file_extension,
            self.instances_file_extension,
        ]

    @property
    def _required_extensions_test(self):
        return [
            self.mesh_file_extension,
        ]

    @property
    def _scannet_labels_filename(self):
        return self.scans_dir / self.raw_labels_filename

    @property
    def _nyu_id_remap(self):
        return defaultdict(
            lambda: self.ignore_label,
            {nyu_id: i for i, nyu_id in enumerate(self._nyu_ids)},
        )

    @property
    def _nyu_ids(self):
        reader = csv.DictReader(self._scannet_labels_filename.open(), delimiter="\t")
        return sorted(
            set(
                [
                    int(line["nyu40id"])
                    for line in reader
                    if line["nyu40class"] in self.semantic_categories
                ]
            )
        )

    def _load(self, scene, force_reload=False):
        scene = self.scans_dir / scene
        processed_scene = scene / (scene.name + ".pth")

        # If already preprocessed, then load previous
        if processed_scene.exists() and not force_reload and not self.force_reload:
            (points, features, semantic_labels, instance_labels) = torch.load(
                str(processed_scene)
            )

        else:

            # Make sure all files exist
            self._check_all_files_exist_in_scene(scene)

            # Load the required data
            points, features = self._extract_inputs(scene)
            semantic_labels = self._extract_semantic_labels(scene)
            instance_labels = self._extract_instance_labels(scene)

            # Save data to avoid re-computation in the future
            torch.save(
                (points, features, semantic_labels, instance_labels),
                processed_scene,
            )

        return SceneWithLabels(
            name=scene.name,
            points=points,
            features=features,
            semantic_labels=semantic_labels,
            instance_labels=instance_labels,
        )

    def _extract_inputs(self, scene):

        # Define raw points file
        mesh_file = scene / (scene.name + self.mesh_file_extension)

        # Read the raw data and extract the inputs
        raw = PlyData.read(mesh_file.open(mode="rb"))["vertex"]
        points = np.array([raw["x"], raw["y"], raw["z"]]).T
        colors = np.array([raw["red"], raw["green"], raw["blue"]]).T

        points = points.astype(np.float32)
        colors = colors.astype(np.float32)

        # Zero and normalize inputs
        points -= points.mean(0)
        colors = colors / 127.5 - 1

        return points, colors

    def _extract_semantic_labels(self, scene):

        # Define label files
        label_file = scene / (scene.name + self.labels_file_extension)

        # Read semantic labels
        semantic_labels = PlyData.read(label_file.open(mode="rb"))
        semantic_labels = semantic_labels["vertex"]["label"]

        # Remap them to use nyu id's
        nyu_id_remap = self._nyu_id_remap
        semantic_labels = [nyu_id_remap[label] for label in semantic_labels]

        return np.array(semantic_labels)

    def _extract_instance_labels(self, scene):

        # Define instance label and segmentation files
        segment_file = scene / (scene.name + self.segment_file_extension)
        instances_file = scene / (scene.name + self.instances_file_extension)

        # Load all segment and instance info
        segments = np.array(json.load(segment_file.open())["segIndices"])
        segments_to_instances = json.load(instances_file.open())["segGroups"]

        # Eliminate duplicate instances in scene0217_00
        if scene.name == "scene0217_00":
            segments_to_instances = segments_to_instances[
                : int(len(segments_to_instances) / 2)
            ]

        # Map segments to instances
        instance_index = 0
        instance_labels = np.ones(segments.size) * self.ignore_label
        for instance in segments_to_instances:

            # Ignore classes
            if instance["label"] in self.ignore_classes:
                continue

            for segment in instance["segments"]:
                instance_labels[np.where(segments == segment)] = instance_index

            instance_index += 1

        return instance_labels

    def _check_all_files_exist_in_scene(self, scene):
        error = False
        for ext in self._required_extensions:
            filename = scene / (scene.stem + ext)
            if not filename.exists():
                log.error(f"scene {scene.name} is missing file {filename.name}")
                error = True

        if error:
            raise RuntimeError(
                f"Missing files in scene: {scene.name}. See error log for details."
            )


scannet_semantic_categories = [
    "unannotated",
    "wall",
    "floor",
    "chair",
    "table",
    "desk",
    "bed",
    "bookshelf",
    "sofa",
    "sink",
    "bathtub",
    "toilet",
    "curtain",
    "counter",
    "door",
    "window",
    "shower curtain",
    "refridgerator",
    "picture",
    "cabinet",
    "otherfurniture",
]
