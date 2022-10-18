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
    def is_scene_preprocessed(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def load(self) -> SceneWithLabels:
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
class SemanticOutput:
    """General semantic outputs"""

    # scores across all classes for each point (# Points, # Classes)
    semantic_scores: torch.Tensor

    @property
    def semantic_pred(self):
        if self.semantic_scores.numel():
            return self.semantic_scores.max(1)[1]
        else:
            raise RuntimeError("No semantic scores are set")
