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


@dataclass
class MinkowskiPretrainInput(MinkowskiInput):
    """Input type of pretraining objective."""

    correspondances: dict = field(default_factory=dict)
    batch_size: int = 0
    offsets: torch.Tensor = torch.tensor([])


@dataclass
class MinkowskiOutput(SemanticOutput):
    """Minkowski output type."""


@dataclass
class PretrainInput(MinkowskiInput):
    """Input type of pretraining objective."""

    correspondances: dict = field(default_factory=dict)
    batch_size: int = 0
    offsets: torch.Tensor = torch.tensor([])
