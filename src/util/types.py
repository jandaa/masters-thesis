from dataclasses import dataclass, field
import torch
from pathlib import Path


@dataclass
class PointGroupInput:
    """Input type of Point Group forward function."""

    # features of inputs (e.g. color channels)
    features: torch.tensor = torch.tensor([])

    # mapping from points to voxels
    point_to_voxel_map: torch.tensor = torch.tensor([])

    # mapping from voxels to points
    voxel_to_point_map: torch.tensor = torch.tensor([])

    # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
    # TODO: Not sure what coordinates these are
    coordinates: torch.tensor = torch.tensor([])

    # input points, float (N, 3)
    point_coordinates: torch.tensor = torch.tensor([])

    # Coordinates of voxels
    voxel_coordinates: torch.tensor = torch.tensor([])

    spatial_shape: int = 3  # TODO: not sure

    @property
    def batch_indices(self):
        return self.coordinates[:, 0].int()


@dataclass
class PointGroupOutput:
    """Output type of Point Group forward function."""

    # scores across all classes for each point (# Points, # Classes)
    semantic_scores: torch.tensor

    # Point offsets of each cluster
    point_offsets: torch.tensor

    # scores of specific instances (TODO: rename this variable)
    proposal_scores: torch.tensor  # = None
    proposal_offsets: torch.tensor  # = None
    proposal_indices: torch.tensor  # = None

    @property
    def semantic_pred(self):
        if self.semantic_scores.numel():
            return self.semantic_scores.max(1)[1]  # (N) long, cuda
        else:
            raise RuntimeError("No semantic scores are set")


@dataclass
class PointGroupBatch(PointGroupInput):
    """Batch type containing all fields required for training."""

    labels: torch.tensor = torch.tensor([])
    instance_labels: torch.tensor = torch.tensor([])
    instance_info: torch.tensor = torch.tensor([])
    instance_pointnum: torch.tensor = torch.tensor([])
    id: torch.tensor = torch.tensor([])
    offsets: torch.tensor = torch.tensor([])

    id: list = field(default_factory=list)
    test_filename: Path = None

    def to(self, device):
        """Cast all tensor-type attributes to device"""
        for fieldname, data in self.__dict__.items():
            if type(data) == torch.Tensor:
                setattr(self, fieldname, data.to(device))

        return self

    def __len__(self):
        """Return the size of the batch."""
        return len(self.id)
