"""
(Modified from PointGroup dataloader)
"""
import glob, os, math, logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader

import scipy
import scipy.ndimage
import scipy.interpolate
from packages.pointgroup_ops.functions import pointgroup_ops

from util.types import PointGroupBatch

log = logging.getLogger(__name__)


class ScannetDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.data_dir = cfg.dataset_dir
        self.filename_suffix = cfg.dataset.filename_suffix

        self.batch_size = cfg.dataset.batch_size
        self.train_workers = cfg.model.train.train_workers
        self.val_workers = cfg.model.train.train_workers

        self.full_scale = cfg.dataset.full_scale
        self.scale = cfg.dataset.scale
        self.max_npoint = cfg.dataset.max_npoint
        self.mode = cfg.dataset.mode

        self.test_split = cfg.model.test.split  # val or test
        self.test_workers = cfg.model.test.test_workers

    def setup(self, stage=None):
        _, self.train_files = self.load_data_files("train")
        _, self.val_files = self.load_data_files("val")
        self.test_filenames, self.test_files = self.load_data_files("val")

        log.info(f"Training samples: {len(self.train_files)}")
        log.info(f"Validation samples: {len(self.val_files)}")
        log.info(f"Testing samples: {len(self.test_files)}")

    def train_dataloader(self):
        return DataLoader(
            list(range(len(self.train_files))),
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
            list(range(len(self.val_files))),
            batch_size=self.batch_size,
            collate_fn=self.val_merge,
            num_workers=self.val_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            list(range(len(self.test_files))),
            batch_size=1,
            collate_fn=self.test_merge,
            num_workers=self.test_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def load_data_files(self, split_type):
        filenames = sorted(
            glob.glob(
                os.path.join(self.data_dir, split_type, "*" + self.filename_suffix)
            )
        )
        return filenames, [torch.load(i) for i in filenames]

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
        return self.merge(id, self.train_files)

    def val_merge(self, id):
        return self.merge(id, self.val_files)

    def test_merge(self, id):
        return self.merge(id, self.test_files, is_test=True)

    def merge(self, id, files, is_test=False):

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

            if are_labels_available:
                xyz_origin, rgb, label, instance_label = files[idx]
            else:
                xyz_origin, rgb = self.test_files[idx]

            if is_test:

                xyz_middle = self.dataAugment(xyz_origin, False, True, True)

                xyz = xyz_middle * self.scale

                xyz -= xyz.min(0)

                feats.append(torch.from_numpy(rgb))

            else:
                ### jitter / flip x / rotation
                xyz_middle = self.dataAugment(xyz_origin, True, True, True)

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
                rgb = rgb[valid_idxs]
                label = label[valid_idxs]
                instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

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
            test_filename = Path(self.test_filenames[id[0]]).stem.replace(
                "_inst_nostuff", ""
            )
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


from dataclasses import dataclass, field
import csv
import json
from plyfile import PlyData
from collections import defaultdict


@dataclass
class Scene:
    """
    A single scene with points, features, semantic
    and instance information.
    """

    points: np.array
    features: np.array
    semantic_label: np.array
    instance_label: np.array


@dataclass
class ScannetDataInterface:
    """
    Interface to load required data for a scene

    properties
    ----------

    scans_dir: directory of all scans and their files

    train_split: list of all scans belonging to training set

    val_split: list of all scans belonging to validations set

    test_split: list of all scans belonging to test set

    semantic_categories: list of all valid semantic categories
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

    @property
    def train_data(self):
        return [self._load(scene) for scene in self.train_split]

    @property
    def val_data(self):
        return [self._load(scene) for scene in self.val_split]

    @property
    def test_data(self):
        return [self._load(scene) for scene in self.test_split]

    @property
    def _scannet_labels_filename(self):
        return self.scans_dir / self.raw_labels_filename

    @property
    def _required_extensions(self):
        return [
            self.mesh_file_extension,
            self.labels_file_extension,
            self.segment_file_extension,
            self.instances_file_extension,
        ]

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

    def _load(self, scene):
        scene = self.scans_dir / scene

        # Make sure all files exist
        self._check_all_files_exist_in_scene(scene)

        # Read raw points file
        mesh_file = scene / (scene.name + self.mesh_file_extension)
        label_file = scene / (scene.name + self.labels_file_extension)
        segment_file = scene / (scene.name + self.segment_file_extension)
        instances_file = scene / (scene.name + self.instances_file_extension)

        raw = PlyData.read(mesh_file.open(mode="rb"))["vertex"]
        labels = PlyData.read(label_file.open(mode="rb"))["vertex"]["label"]

        points = np.array([raw["x"], raw["y"], raw["z"]]).T
        colors = np.array([raw["red"], raw["green"], raw["blue"]]).T

        # Zero out points
        points -= points.mean(0)

        # Normalize colours
        colors = colors / 127.5 - 1

        # Remap all semantic labels
        nyu_id_remap = self._nyu_id_remap
        labels = [nyu_id_remap[label] for label in labels]

    def _check_all_files_exist_in_scene(self, scene):
        for ext in self._required_extensions:
            filename = scene / (scene.stem + ext)
            if not filename.exists():
                log.error(f"scene {scene.name} is missing file {filename.name}")

    # TODO: Is this necessary? Can I just check if the label is
    # in the semantic categories or not?
    def _get_raw_to_label_map(self):
        """
        Get map that converts raw ground truth labels
        to those designated in semantic_categories.

        returns
        -------
        dictionary with keys of raw names mapping to semantic labels
        """
        raw_to_scannet_map = {}
        with self._scannet_labels_filename.open() as f:
            reader = csv.DictReader(f, delimiter="\t")

        for line in reader:
            if line["nyu40class"] in self.semantic_categories:
                raw_to_scannet_map[line["raw_category"]] = line["nyu40class"]
            else:
                raw_to_scannet_map[line["raw_category"]] = "unannotated"

        return raw_to_scannet_map


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
