from pathlib import Path
from abc import ABC, abstractmethod
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
class DataPoint(ABC):
    """
    General datapoint class that will load the data
    when called during training
    """

    @property
    @abstractmethod
    def num_points(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def is_scene_preprocessed(self, force_reload) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, force_reload=False) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load(self, force_reload=False) -> SceneWithLabels:
        raise NotImplementedError()


@dataclass
class DataInterface(ABC):
    """
    General data interface that is able to load
    each data split type.
    """

    semantic_categories: list
    instance_categories: list = field(init=False)
    index_to_label_map: map = field(init=False)
    label_to_index_map: map = field(init=False)

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

    # input points, float (N, 3)
    points: torch.Tensor = torch.tensor([])

    # features of inputs (e.g. color channels)
    features: torch.Tensor = torch.tensor([])

    # Coordinates of voxels
    voxel_coordinates: torch.Tensor = torch.tensor([])

    # mapping from points to voxels
    point_to_voxel_map: torch.Tensor = torch.tensor([])

    # mapping from voxels to points
    voxel_to_point_map: torch.Tensor = torch.tensor([])

    # element for each point with index of which batch
    # the point came from
    batch_indices: torch.Tensor = torch.tensor([])

    spatial_shape: int = 3  # Shape in terms of voxels


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
    instance_centers: torch.Tensor = torch.tensor([])
    instance_pointnum: torch.Tensor = torch.tensor([])
    offsets: torch.Tensor = torch.tensor([])

    batch_size: int = 0
    test_filename: Path = None
    device: str = "cpu"

    def to(self, device):
        """Cast all tensor-type attributes to device"""
        self.device = device
        for fieldname, data in self.__dict__.items():
            if type(data) == torch.Tensor:
                setattr(self, fieldname, data.to(device))

        return self

    def __len__(self):
        """Return the size of the batch."""
        return self.batch_size


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
