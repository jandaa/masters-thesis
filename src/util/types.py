from dataclasses import dataclass
import torch
from pathlib import Path


@dataclass
class PointGroupInput:
    """Input type of Point Group forward function."""

    features: torch.tensor  # features of inputs (e.g. color channels)
    point_coordinates: torch.tensor  # input points
    point_to_voxel_map: torch.tensor  # mapping from points to voxels
    voxel_to_point_map: torch.tensor  # mapping from voxels to points
    coordinates: torch.tensor  # TODO: Not sure what coordinates these are
    voxel_coordinates: torch.tensor  # Coordinates of voxels

    spatial_shape: int  # TODO: not sure

    @property
    def batch_indices(self):
        return self.coordinates[:, 0].int()

    @staticmethod
    def from_batch(batch):
        return PointGroupInput(
            features=batch["feats"],
            point_coordinates=batch["locs_float"],
            point_to_voxel_map=batch["p2v_map"],
            voxel_to_point_map=batch["v2p_map"],
            coordinates=batch["locs"],
            voxel_coordinates=batch["voxel_locs"],
            spatial_shape=batch["spatial_shape"],
        )


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
        if self.semantic_scores:
            return self.semantic_scores.max(1)[1]  # (N) long, cuda
        else:
            raise RuntimeError("No semantic scores are set")


@dataclass
class PointGroupBatch(PointGroupInput):
    """Batch type containing all fields required for training."""

    labels: torch.tensor
    instance_labels: torch.tensor
    instance_info: torch.tensor
    instance_pointnum: torch.tensor
    id: torch.tensor
    batch_offsets: torch.tensor

    test_filename: Path
