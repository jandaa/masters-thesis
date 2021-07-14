import logging
import csv
import json
import concurrent.futures
from math import floor

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
    semantic_categories: list

    # Constants
    raw_labels_filename: str = "../scannetv2-labels.combined.tsv"
    mesh_file_extension: str = "_vh_clean_2.ply"
    labels_file_extension: str = "_vh_clean_2.labels.ply"
    segment_file_extension: str = "_vh_clean_2.0.010000.segs.json"
    instances_file_extension: str = ".aggregation.json"
    ignore_label: int = -100
    ignore_classes: list = field(
        default_factory=lambda: ["wall", "floor", "unannotated"]
    )
    force_reload: bool = False
    num_threads: int = 8
    _nyu_id_remap: map = field(init=False)

    def __post_init__(self):

        # Get map from nyu ids to our ids
        reader = csv.DictReader(self._scannet_labels_filename.open(), delimiter="\t")
        self.raw_to_nyu_label_map = {
            line["raw_category"]: line["nyu40class"] for line in reader
        }

        reader = csv.DictReader(self._scannet_labels_filename.open(), delimiter="\t")
        nyu_label_to_id_map = {
            line["nyu40class"]: int(line["nyu40id"])
            for line in reader
            if line["nyu40class"] in self.semantic_categories
        }
        nyu_ids = sorted(set(nyu_label_to_id_map.values()))

        self._nyu_id_remap = defaultdict(
            lambda: self.ignore_label,
            {nyu_id: i for i, nyu_id in enumerate(nyu_ids)},
        )
        self.label_to_index_map = {
            label: self._nyu_id_remap[id] for label, id in nyu_label_to_id_map.items()
        }
        self.index_to_label_map = {
            index: label_name for label_name, index in self.label_to_index_map.items()
        }

    @property
    def train_data(self) -> list:
        return self._load_multithread(self.train_split)

    @property
    def val_data(self) -> list:
        return self._load_multithread(self.val_split)

    @property
    def test_data(self) -> list:
        return self._load_multithread(self.test_split)

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

    def _get_scenes_per_thread(self, scenes):
        num_scenes = len(scenes)
        if num_scenes == 0:
            return []

        num_threads = min(num_scenes, self.num_threads)
        num_rooms_per_thread = floor(num_scenes / num_threads)

        def start_index(thread_ind):
            return thread_ind * num_rooms_per_thread

        def end_index(thread_ind):
            if thread_ind == num_threads - 1:
                return None
            return start_index(thread_ind) + num_rooms_per_thread

        return [
            scenes[start_index(thread_ind) : end_index(thread_ind)]
            for thread_ind in range(num_threads)
        ]

    def _load_multithread(self, scenes) -> list:

        scenes = [
            scene
            for scene in scenes
            if self._do_all_files_exist_in_scene(self.scans_dir / scene)
        ]

        scenes_per_thread = self._get_scenes_per_thread(scenes)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            threads = [
                executor.submit(self._load_scenes, scenes)
                for scenes in scenes_per_thread
            ]

        output = []
        for thread in threads:
            output += thread.result()

        return output

    def _load_scenes(self, scenes):
        return [self._load(self.scans_dir / scene) for scene in scenes]

    def _load(self, scene: Path, force_reload=False):
        processed_scene = self._get_processed_scene_name(scene)

        # If already preprocessed, then load previous
        if processed_scene.exists() and not force_reload and not self.force_reload:
            # log.info(f"Trying to load preprocessed scene: {scene.name}")
            try:
                (points, features, semantic_labels, instance_labels) = torch.load(
                    str(processed_scene)
                )
            except:
                log.info(f"Error trying to force reload: {scene.name}")
                return self._load(scene, force_reload=True)

        else:

            log.info(f"Loading scene: {scene.name}")

            # Load the required data
            points, features = self._extract_inputs(scene)
            semantic_labels = self._extract_semantic_labels(scene)
            instance_labels = self._extract_instance_labels(scene)

            log.info(f"Saving scene: {scene.name}")

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
        semantic_labels = [self._nyu_id_remap[label] for label in semantic_labels]

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
            if self.raw_to_nyu_label_map[instance["label"]] in self.ignore_classes:
                continue

            for segment in instance["segments"]:
                instance_labels[np.where(segments == segment)] = instance_index

            instance_index += 1

        return instance_labels

    def _get_processed_scene_name(self, scene: Path):
        return scene / (scene.name + ".pth")

    def _do_all_files_exist_in_scene(self, scene):
        processed_scene_name = self._get_processed_scene_name(scene)
        if processed_scene_name.exists():
            return True

        all_files_exist = True
        for ext in self._required_extensions:
            filename = scene / (scene.stem + ext)
            if not filename.exists():
                log.error(f"scene {scene.name} is missing file {filename.name}")
                all_files_exist = False

        if not all_files_exist:
            log.error(f"skipping scene: {scene.name} due to missing files")
        return all_files_exist
