from dataclasses import dataclass, field
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
class MinkowskiPretrainInput:
    """Pretrain input type of Minkowki Networks."""

    points: torch.Tensor
    features: torch.Tensor
    correspondences: list
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
class MinkowskiOutput(SemanticOutput):
    """Minkowski output type."""

    output: ME.SparseTensor
