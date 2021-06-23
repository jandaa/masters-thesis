"""
(Modified from PointGroup dataloader)
"""
import glob, os, math, logging, csv, json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from plyfile import PlyData
from omegaconf import DictConfig

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import scipy
import scipy.ndimage
import scipy.interpolate
from packages.pointgroup_ops.functions import pointgroup_ops

from util.types import PointGroupBatch, DataInterface, SceneWithLabels

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


class ScannetDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, data_interface: DataInterface):
        super().__init__()

        # Dataloader specific parameters
        self.batch_size = cfg.dataset.batch_size
        self.full_scale = cfg.dataset.full_scale
        self.scale = cfg.dataset.scale
        self.max_npoint = cfg.dataset.max_npoint
        self.mode = cfg.dataset.mode

        # What kind of test?
        # val == with labels
        # test == without labels
        self.test_split = cfg.model.test.split  # val or test

        # Number of workers
        self.train_workers = cfg.model.train.train_workers
        self.val_workers = cfg.model.train.train_workers
        self.test_workers = cfg.model.test.test_workers

        # Load data from interface
        self.train_data = data_interface.train_data
        self.val_data = data_interface.val_data
        self.test_data = data_interface.val_data

        log.info(f"Training samples: {len(self.train_data)}")
        log.info(f"Validation samples: {len(self.train_data)}")
        log.info(f"Testing samples: {len(self.train_data)}")

    def train_dataloader(self):
        return DataLoader(
            list(range(len(self.train_data))),
            batch_size=self.batch_size,
            collate_fn=self.train_merge,
            num_workers=self.train_workers,
            shuffle=True,
            sampler=None,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            list(range(len(self.val_data))),
            batch_size=self.batch_size,
            collate_fn=self.val_merge,
            num_workers=self.val_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            list(range(len(self.test_data))),
            batch_size=1,
            collate_fn=self.test_merge,
            num_workers=self.test_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    # Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [
            np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0)
            for n in noise
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0)
            for n in noise
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0)
            for n in noise
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0)
            for n in noise
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0)
            for n in noise
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0)
            for n in noise
        ]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0
            )
            for n in noise
        ]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label):
        """
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        """
        instance_info = (
            np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0
        )  # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []  # (nInst), int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {
            "instance_info": instance_info,
            "instance_pointnum": instance_pointnum,
        }

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
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

    def crop(self, xyz):
        """
        :param xyz: (n, 3) >= 0
        """
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.max_npoint:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(
                3
            )
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * (
                (xyz_offset < full_scale).sum(1) == 3
            )
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def train_merge(self, id):
        return self.merge(id, self.train_data)

    def val_merge(self, id):
        return self.merge(id, self.val_data)

    def test_merge(self, id):
        return self.merge(id, self.test_data, is_test=True)

    def merge(self, id, scenes, is_test=False):

        # Make sure valid test split option is specified
        if is_test and self.test_split not in ["val", "test"]:
            raise RuntimeError(f"Wrong test split: {self.test_split}")

        # Whether semantics and instance labels are available
        are_labels_available = is_test and self.test_split == "val" or not is_test

        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):

            scene = scenes[idx]

            if is_test:

                xyz_middle = self.dataAugment(scene.points, False, True, True)

                xyz = xyz_middle * self.scale

                xyz -= xyz.min(0)

                feats.append(torch.from_numpy(scene.features))

                label = scene.semantic_labels
                instance_label = scene.instance_labels

            else:
                ### jitter / flip x / rotation
                xyz_middle = self.dataAugment(scene.points, True, True, True)

                ### scale
                xyz = xyz_middle * self.scale

                ### elastic
                xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
                xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

                ### offset
                xyz -= xyz.min(0)

                ### crop
                xyz, valid_idxs = self.crop(xyz)

                xyz_middle = xyz_middle[valid_idxs]
                xyz = xyz[valid_idxs]
                rgb = scene.features[valid_idxs]
                label = scene.semantic_labels[valid_idxs]
                instance_label = self.getCroppedInstLabel(
                    scene.instance_labels, valid_idxs
                )

                feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)

            if are_labels_available:
                ### get instance information
                inst_num, inst_infos = self.getInstanceInfo(
                    xyz_middle, instance_label.astype(np.int32)
                )
                inst_info = inst_infos[
                    "instance_info"
                ]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
                inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

                instance_label[np.where(instance_label != -100)] += total_inst_num
                total_inst_num += inst_num

                # Add training and validation info
                instance_infos.append(torch.from_numpy(inst_info))
                instance_pointnum.extend(inst_pointnum)

                labels.append(torch.from_numpy(label))
                instance_labels.append(torch.from_numpy(instance_label))

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(
                torch.cat(
                    [
                        torch.LongTensor(xyz.shape[0], 1).fill_(i),
                        torch.from_numpy(xyz).long(),
                    ],
                    1,
                )
            )
            locs_float.append(torch.from_numpy(xyz_middle))

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(
            locs, 0
        )  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)

        spatial_shape = np.clip(
            (locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None
        )  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(
            locs, self.batch_size, self.mode
        )

        if are_labels_available:
            labels = torch.cat(labels, 0).long()  # long (N)
            instance_labels = torch.cat(instance_labels, 0).long()  # long (N)

            instance_infos = torch.cat(instance_infos, 0).to(
                torch.float32
            )  # float (N, 9) (meanxyz, minxyz, maxxyz)
            instance_pointnum = torch.tensor(
                instance_pointnum, dtype=torch.int
            )  # int (total_nInst)

        if is_test:
            test_filename = scenes[id[0]].name
        else:
            test_filename = None

        return PointGroupBatch(
            coordinates=locs,
            voxel_coordinates=voxel_locs,
            point_to_voxel_map=p2v_map,
            voxel_to_point_map=v2p_map,
            point_coordinates=locs_float,
            features=feats,
            labels=labels,
            instance_labels=instance_labels,
            instance_info=instance_infos,
            instance_pointnum=instance_pointnum,
            offsets=batch_offsets,
            id=id,
            spatial_shape=spatial_shape,
            test_filename=test_filename,
        )
