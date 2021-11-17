from pathlib import Path
from dataclasses import dataclass, field

import torch
from util.types import SemanticOutput


@dataclass
class MinkowskiInput:
    """Input type of Minkovski Network."""

    points: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor

    test_filename: str

    def to(self, device):
        """Cast all tensor-type attributes to device"""
        self.device = device
        for fieldname, data in self.__dict__.items():
            if type(data) == torch.Tensor:
                setattr(self, fieldname, data.to(device))

        return self


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


@dataclass
class PretrainInput(MinkowskiInput):
    """Input type of pretraining objective."""

    correspondances: dict = field(default_factory=dict)
    batch_size: int = 0
    offsets: torch.Tensor = torch.tensor([])
