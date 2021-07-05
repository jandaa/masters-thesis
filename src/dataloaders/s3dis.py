import logging
import concurrent.futures
from math import floor

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import numpy as np

from util.types import DataInterface, SceneWithLabels

log = logging.getLogger(__name__)


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

    ignore_label: int = -100
    force_reload: bool = False
    num_threads: int = 8

    def __post_init__(self):
        self.label_to_index_map = defaultdict(
            lambda: self.ignore_label,
            {
                label_name: index
                for index, label_name in enumerate(self.semantic_categories)
            },
        )
        self.index_to_label_map = {
            index: label_name for index, label_name in self.label_to_index_map.items()
        }

    @property
    def train_data(self) -> list:
        return self._load_split(self.train_split)

    @property
    def val_data(self) -> list:
        return self._load_split(self.val_split)

    @property
    def test_data(self) -> list:
        return self._load_split(self.test_split)

    def _get_rooms(self, areas) -> list:
        return [
            room
            for area in areas
            for room in (self.dataset_dir / (f"Area_{area}")).iterdir()
            if room.is_dir()
        ]

    def _get_rooms_per_thread(self, split):
        rooms = self._get_rooms(split)
        num_threads = min(len(rooms), self.num_threads)
        num_rooms_per_thread = floor(len(rooms) / num_threads)

        def start_index(thread_ind):
            return thread_ind * num_rooms_per_thread

        def end_index(thread_ind):
            if thread_ind == num_threads - 1:
                return None
            return start_index(thread_ind) + num_rooms_per_thread

        return [
            rooms[start_index(thread_ind) : end_index(thread_ind)]
            for thread_ind in range(num_threads)
        ]

    def _load_split(self, split) -> list:

        rooms_per_thread = self._get_rooms_per_thread(split)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            threads = [
                executor.submit(self._load_rooms, rooms) for rooms in rooms_per_thread
            ]

        output = []
        for thread in threads:
            output += thread.result()

        return output

    def _load_rooms(self, rooms: list) -> list:
        return [self._load_room(room) for room in rooms]

    def _load_room(self, room: Path, force_reload=False) -> SceneWithLabels:

        # Files and names
        scene_name = f"{room.parent.name}_{room.name}"
        processed_scene = room / (room.name + ".pth")

        # Load processed scene if already preprocessed
        if processed_scene.exists() and not force_reload and not self.force_reload:
            try:
                (points, features, semantic_labels, instance_labels) = torch.load(
                    str(processed_scene)
                )
            except:
                log.info(f"Error loading {room.name}. Trying to force reload.")
                return self._load_room(room, force_reload=True)
            
        # Process scene if not already done so
        else:

            log.info(f"Loading scene: {scene_name}")

            points = []
            features = []
            semantic_labels = []
            instance_labels = []

            # Load points of each object instance making up the scene
            annotations_dir = room / "Annotations"
            for instance_label, object in enumerate(annotations_dir.iterdir()):

                # Ignore any hidden files
                if object.name.startswith("."):
                    continue

                semantic_label = self.label_to_index_map[object.name.split("_")[0]]
                object_points = np.loadtxt(str(object), delimiter=" ")
                object_points = object_points.astype(np.float32)

                num_points = object_points.shape[0]
                points.append(object_points[:, 0:3])
                features.append(object_points[:, 3:6])
                semantic_labels.append(
                    np.ones((num_points), dtype=np.int) * semantic_label
                )
                instance_labels.append(
                    np.ones((num_points), dtype=np.int) * instance_label
                )

            # Concatonate to make into full vectors
            points = np.concatenate(points, 0)
            features = np.concatenate(features, 0)
            semantic_labels = np.concatenate(semantic_labels, None)
            instance_labels = np.concatenate(instance_labels, None)

            # Zero and normalize inputs
            points -= points.mean(0)
            features = features / 127.5 - 1

            # Save data to avoid re-computation in the future
            log.info(f"Saving scene: {scene_name}")
            torch.save(
                (points, features, semantic_labels, instance_labels),
                processed_scene,
            )

        return SceneWithLabels(
            name=f"{room.parent.name}_{room.name}",
            points=points,
            features=features,
            semantic_labels=semantic_labels,
            instance_labels=instance_labels,
        )
