from dataclasses import dataclass
import torch

import MinkowskiEngine as ME

from util.types import SemanticOutput


@dataclass
class MinkowskiInput:
    """Input type of Minkovski Network."""

    points: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor
    batch_size: int
    test_filename: str

    def to(self, device):
        """Cast all tensor-type attributes to device"""
        self.device = device
        for fieldname, data in self.__dict__.items():
            if type(data) == torch.Tensor:
                setattr(self, fieldname, data.to(device))

        return self

    def __len__(self):
        return self.batch_size


@dataclass
class ImagePretrainInput:
    """Pretrain input type of Minkowki Networks."""

    # 2D Input
    images1: torch.Tensor
    images2: torch.Tensor
    coords1: torch.Tensor
    coords2: torch.Tensor
    labels: torch.Tensor
    correspondences: list

    batch_size: int

    # # 3D input
    # points1: torch.Tensor
    # features1: torch.Tensor
    # points2: torch.Tensor
    # features2: torch.Tensor

    # # 2D to 3D
    # point_to_pixel_maps1: torch.Tensor
    # point_to_pixel_maps2: torch.Tensor
    # image_coordinates: list

    def to(self, device):
        """Cast all tensor-type attributes to device"""
        self.device = device
        for fieldname, data in self.__dict__.items():
            if type(data) == torch.Tensor:
                setattr(self, fieldname, data.to(device))

        return self

    def __len__(self):
        return self.batch_size


@dataclass
class MinkowskiPretrainInputNew:
    """Pretrain input type of Minkowki Networks."""

    # 3D input
    points1: torch.Tensor
    features1: torch.Tensor
    points2: torch.Tensor
    features2: torch.Tensor

    # 2D
    images: torch.Tensor
    image_coordinates: list

    # 3D to 3D
    point_to_point_map: list

    # 2D to 3D
    point_to_pixel_maps1: torch.Tensor
    point_to_pixel_maps2: torch.Tensor

    batch_size: int

    def to(self, device):
        """Cast all tensor-type attributes to device"""
        self.device = device
        for fieldname, data in self.__dict__.items():
            if type(data) == torch.Tensor:
                setattr(self, fieldname, data.to(device))

        return self

    def __len__(self):
        return self.batch_size


@dataclass
class MinkowskiPretrainInput:
    """Pretrain input type of Minkowki Networks."""

    points: torch.Tensor
    features: torch.Tensor
    images: torch.Tensor
    correspondences: list
    batch_size: int

    image_coordinates: torch.Tensor = None

    def to(self, device):
        """Cast all tensor-type attributes to device"""
        self.device = device
        for fieldname, data in self.__dict__.items():
            if type(data) == torch.Tensor:
                setattr(self, fieldname, data.to(device))

        return self

    def __len__(self):
        return self.batch_size


@dataclass
class MinkowskiOutput(SemanticOutput):
    """Minkowski output type."""

    output: ME.SparseTensor
