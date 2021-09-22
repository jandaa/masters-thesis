import logging
import csv
import json
import zipfile
import shutil

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
from subprocess import Popen, PIPE, DEVNULL

import torch
import numpy as np
from hydra.utils import get_original_cwd

from plyfile import PlyData

from util.types import DataInterface, DataPoint, SceneWithLabels
from util.scene import SceneMeasurements, measurements_dir_name

log = logging.getLogger(__name__)


@dataclass
class ScannetDataPoint(DataPoint):

    scene_path: Path
    force_reload: bool
    preprocess_callback: Callable

    def __post_init__(self):

        # Files and names
        self.scene_name = self.scene_path.name
        self.processed_scene = self.scene_path / (self.scene_path.name + ".pth")
        self.measurements_file = self.scene_path / (
            self.scene_path.name + "_" + measurements_dir_name + ".pkl"
        )
        self.scene_details_file = self.scene_path / (
            self.scene_path.name + "_details.json"
        )

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

    def does_scene_contain_measurements(self):
        return self.measurements_file.exists()

    def preprocess(self, force_reload=False):
        return self.preprocess_callback(self, force_reload)

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

    def load_measurements(self, force_reload=False) -> SceneMeasurements:
        if not self.does_scene_contain_measurements():
            return self.preprocess(force_reload=True)

        return SceneMeasurements.load_scene(self.scene_path)


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

    ignore_label: int
    instance_ignore_classes: list

    # Defaults
    force_reload: bool = False

    def __post_init__(self):

        # Default values
        self.mesh_file_extension = "_vh_clean_2.ply"
        self.labels_file_extension = "_vh_clean_2.labels.ply"
        self.segment_file_extension = "_vh_clean_2.0.010000.segs.json"
        self.instances_file_extension = ".aggregation.json"
        self.required_extensions = [
            self.mesh_file_extension,
            self.labels_file_extension,
            self.segment_file_extension,
            self.instances_file_extension,
        ]
        self.required_extensions_test = [self.mesh_file_extension]
        self.sensor_measurments_extension = ".sens"
        self.zipfiles_to_extract = [
            "_2d-instance-filt.zip",
            "_2d-label-filt.zip",
        ]
        self.scannet_labels_filename = (
            self.scans_dir / "../scannetv2-labels.combined.tsv"
        )

        # Specify which categories to ignore for instance segmentation
        self.instance_categories = [
            label
            for label in self.semantic_categories
            if label not in self.instance_ignore_classes
        ]

        # Get map from nyu ids to our ids
        reader = csv.DictReader(self.scannet_labels_filename.open(), delimiter="\t")
        self.raw_to_nyu_label_map = {
            line["raw_category"]: line["nyu40class"] for line in reader
        }

        reader = csv.DictReader(self.scannet_labels_filename.open(), delimiter="\t")
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
    def pretrain_data(self) -> list:
        return self.load_pretrain(self.train_split)

    @property
    def train_data(self) -> list:
        return self.load(self.train_split)

    @property
    def val_data(self) -> list:
        return self.load(self.val_split)

    @property
    def test_data(self) -> list:
        return self.load(self.test_split)

    def load(self, scenes: list, force_reload=False):
        force_reload = force_reload or self.force_reload
        datapoints = self.get_datapoints(scenes, force_reload=force_reload)
        if force_reload:
            return [
                datapoint
                for datapoint in datapoints
                if self.do_all_files_exist_in_scene(datapoint.scene_path)
            ]
        return datapoints

    def load_pretrain(self, scenes: list, force_reload=False):
        force_reload = force_reload or self.force_reload
        datapoints = self.get_datapoints(scenes, force_reload=force_reload)
        return [
            datapoint
            for datapoint in datapoints
            if (
                force_reload
                and self.do_sensor_files_exist_in_scene(datapoint.scene_path)
            )
            or (
                datapoint.does_scene_contain_measurements()
                or self.do_sensor_files_exist_in_scene(datapoint.scene_path)
            )
        ]

    def get_datapoints(self, scenes: list, force_reload=False):
        force_reload = force_reload or self.force_reload
        return [
            ScannetDataPoint(
                scene_path=self.scans_dir / scene,
                force_reload=force_reload,
                preprocess_callback=self.preprocess,
            )
            for scene in scenes
        ]

    def do_all_files_exist_in_scene(self, scene):
        all_files_exist = True
        for ext in self.required_extensions:
            filename = scene / (scene.stem + ext)
            if not filename.exists():
                log.error(f"scene {scene.name} is missing file {filename.name}")
                all_files_exist = False

        if not all_files_exist:
            log.error(f"skipping scene: {scene.name} due to missing files")
        return all_files_exist

    def do_sensor_files_exist_in_scene(self, scene):
        all_files_exist = True
        extensions = self.zipfiles_to_extract + [self.sensor_measurments_extension]
        for ext in self.required_extensions:
            filename = scene / (scene.stem + ext)
            if not filename.exists():
                log.error(f"scene {scene.name} is missing file {filename.name}")
                all_files_exist = False

        if not all_files_exist:
            log.error(
                f"skipping measurments in scene: {scene.name} due to missing files"
            )
        return all_files_exist

    def preprocess(self, datapoint: ScannetDataPoint, force_reload: bool) -> None:
        if datapoint.is_scene_preprocessed(force_reload):
            return

        if not self.do_all_files_exist_in_scene(datapoint.scene_path):
            raise RuntimeError(
                f"Trying to preprocess scene {datapoint.scene_name} but missing original files"
            )

        log.info(f"Loading scene: {datapoint.scene_name}")

        # Load the required data
        points, features = self.extract_inputs(datapoint.scene_path)
        semantic_labels = self.extract_semantic_labels(datapoint.scene_path)
        instance_labels = self.extract_instance_labels(datapoint.scene_path)

        # Save data to avoid re-computation in the future
        log.info(f"Saving scene: {datapoint.scene_name}")
        torch.save(
            (points, features, semantic_labels, instance_labels),
            datapoint.processed_scene,
        )

        details = {"num_points": points.shape[0]}
        with datapoint.scene_details_file.open(mode="w") as fp:
            json.dump(details, fp)

        # Extract sensor measurements if available
        if self.do_sensor_files_exist_in_scene(datapoint.scene_path):
            self.preprocess_measurements(datapoint)

    def preprocess_measurements(self, datapoint: ScannetDataPoint):
        """Preprocess raw sensor measurements of a scene including it's semantic and
        instance labels."""

        log.info(f"Loading sensor measurements for scene: {datapoint.scene_name}")

        # Extract all data from compressed raw data
        self.extract_sens_file(datapoint.scene_path)
        self.extract_zip_files(datapoint.scene_path)

        # Preprocess measurements
        measurements = SceneMeasurements(datapoint.scene_path)

        # Save measurements
        log.info(f"Saving sensor measurements for scene: {datapoint.scene_name}")
        measurements.save_to_file()

        # Clean up all extracted raw data
        for item in datapoint.scene_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)

        return measurements

    def extract_sens_file(self, scene):
        """Extract all data out of the .sens file."""
        output_folder = scene / measurements_dir_name
        if not output_folder.exists():
            output_folder.mkdir()

        p = Popen(
            [
                get_original_cwd() + "/src/packages/SensReader/sens",
                scene / (scene.name + self.sensor_measurments_extension),
                output_folder,
            ],
            stdin=PIPE,
            stdout=DEVNULL,
        )
        while True:
            if not p.poll() is None:
                break

    def extract_zip_files(self, scene):
        """Unzip all zip files containing label and instance data."""
        for ext in self.zipfiles_to_extract:
            file_to_extract = scene / (scene.name + ext)
            with zipfile.ZipFile(file_to_extract) as zp:
                zp.extractall(scene)

    def extract_inputs(self, scene):

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

    def extract_semantic_labels(self, scene):

        # Define label files
        label_file = scene / (scene.name + self.labels_file_extension)

        # Read semantic labels
        semantic_labels = PlyData.read(label_file.open(mode="rb"))
        semantic_labels = semantic_labels["vertex"]["label"]

        # Remap them to use nyu id's
        semantic_labels = [self._nyu_id_remap[label] for label in semantic_labels]

        return np.array(semantic_labels)

    def extract_instance_labels(self, scene):

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
            if (
                self.raw_to_nyu_label_map[instance["label"]]
                in self.instance_ignore_classes
            ):
                continue

            for segment in instance["segments"]:
                instance_labels[np.where(segments == segment)] = instance_index

            instance_index += 1

        return instance_labels
