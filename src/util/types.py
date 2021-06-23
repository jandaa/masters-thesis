from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import numpy as np


@dataclass
class Scene:
    """
    A single scene with points and features to
    be used as inputs for segmentation
    """

    name: str
    points: np.array
    features: np.array


@dataclass
class SceneWithLabels(Scene):
    """
    A single scene with additional semantic and instance labels
    for training and evaluation.
    """

    semantic_labels: np.array
    instance_labels: np.array


@dataclass
class DataInterface(ABC):
    """
    General data interface that is able to load
    each data split type.
    """

    @property
    @abstractmethod
    def train_data(self) -> list:
        raise NotImplementedError()

    @property
    @abstractmethod
    def val_data(self) -> list:
        raise NotImplementedError()

    @property
    @abstractmethod
    def test_data(self) -> list:
        raise NotImplementedError()


@dataclass
class PointGroupInput:
    """Input type of Point Group forward function."""

    # features of inputs (e.g. color channels)
    features: torch.Tensor = torch.tensor([])

    # mapping from points to voxels
    point_to_voxel_map: torch.Tensor = torch.tensor([])

    # mapping from voxels to points
    voxel_to_point_map: torch.Tensor = torch.tensor([])

    # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    # TODO: Not sure what coordinates these are
    coordinates: torch.Tensor = torch.tensor([])

    # input points, float (N, 3)
    point_coordinates: torch.Tensor = torch.tensor([])

    # Coordinates of voxels
    voxel_coordinates: torch.Tensor = torch.tensor([])

    spatial_shape: int = 3  # TODO: not sure

    @property
    def batch_indices(self):
        return self.coordinates[:, 0].int()


@dataclass
class PointGroupOutput:
    """Output type of Point Group forward function."""

    # scores across all classes for each point (# Points, # Classes)
    semantic_scores: torch.Tensor

    # Point offsets of each cluster
    point_offsets: torch.Tensor

    # scores of specific instances (TODO: rename this variable)
    proposal_scores: torch.Tensor
    proposal_offsets: torch.Tensor
    proposal_indices: torch.Tensor

    @property
    def semantic_pred(self):
        if self.semantic_scores.numel():
            return self.semantic_scores.max(1)[1]
        else:
            raise RuntimeError("No semantic scores are set")


@dataclass
class PointGroupBatch(PointGroupInput):
    """Batch type containing all fields required for training."""

    labels: torch.Tensor = torch.tensor([])
    instance_labels: torch.Tensor = torch.tensor([])
    instance_info: torch.Tensor = torch.tensor([])
    instance_pointnum: torch.Tensor = torch.tensor([])
    id: torch.Tensor = torch.tensor([])
    offsets: torch.Tensor = torch.tensor([])

    id: list = field(default_factory=list)
    test_filename: Path = None

    device: str = "cpu"

    def to(self, device):
        """Cast all tensor-type attributes to device"""
        self.device = device
        for fieldname, data in self.__dict__.items():
            if type(data) == torch.Tensor:
                setattr(self, fieldname, data.to(device))

        return self

    @property
    def batch_size(self):
        return len(self)

    def __len__(self):
        """Return the size of the batch."""
        return len(self.id)


@dataclass
class LossType:
    """Loss type containing all different types of losses"""

    # Semantic Segmentation Loss
    semantic_loss: torch.Tensor
    offset_norm_loss: torch.Tensor
    offset_dir_loss: torch.Tensor
    number_of_points: int
    number_of_valid_labels: int

    # Instance Segmentation Loss
    score_loss: torch.Tensor
    number_of_instances: int

    # Total loss
    total_loss: torch.Tensor
