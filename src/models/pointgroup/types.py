from pathlib import Path
from dataclasses import dataclass, field

import torch
from util.types import SemanticOutput


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
class PretrainInput(PointGroupInput):
    """Input type of pretraining objective."""

    correspondances: dict = field(default_factory=dict)
    batch_size: int = 0


@dataclass
class PointGroupOutput(SemanticOutput):
    """Output type of Point Group forward function."""

    # Point offsets of each cluster
    point_offsets: torch.Tensor

    # scores of specific instances (TODO: rename this variable)
    proposal_scores: torch.Tensor
    proposal_offsets: torch.Tensor
    proposal_indices: torch.Tensor


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