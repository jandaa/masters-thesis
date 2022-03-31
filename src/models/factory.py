from pathlib import Path
from omegaconf import DictConfig

from models.minkowski.trainer import (
    MinkowskiTrainer,
    MinkowskiBackboneTrainer,
    MinkowskiMocoBackboneTrainer,
    MinkowskiBOYLBackboneTrainer,
    CMEBackboneTrainer,
    CMEBackboneTrainerFull,
    ImageTrainer,
)
from models.minkowski.dataset import (
    ImagePretrainDataset,
    MinkowskiDataset,
    MinkowskiPretrainDataset,
    MinkowskiFrameDataset,
    MinkowskiS3DISDataset,
    PointContrastPretrainDataset,
)
from util.types import DataInterface

minkowski_name = "minkowski"
pointcontrast_name = "pointcontrast"
moco_name = "minkowski_moco"
byol_name = "minkowski_byol"
cme_name = "minkowski_cme"
image_name = "image_pretrain"
supported_models = [
    minkowski_name,
    moco_name,
    byol_name,
    cme_name,
    image_name,
    pointcontrast_name,
]


class ModelFactory:
    def __init__(self, cfg: DictConfig, data_interface: DataInterface, backbone=None):
        self.model_name = cfg.model.name
        self.dataset_name = cfg.dataset.name
        self.cfg = cfg
        self.data_interface = data_interface
        self.backbone = backbone

        # Ensure that the model is supported
        self.error_msg = f"model {self.model_name} is not supported"
        if self.model_name not in supported_models:
            raise RuntimeError(self.error_msg)

    def get_model(self):
        return MinkowskiTrainer(self.cfg, self.data_interface, backbone=self.backbone)
        # if minkowski_name in self.model_name:
        #     return MinkowskiTrainer(
        #         self.cfg, self.data_interface, backbone=self.backbone
        #     )
        # else:
        #     raise RuntimeError(self.error_msg)

    def get_backbone_wrapper_type(self):
        if self.model_name == minkowski_name:
            return MinkowskiBackboneTrainer
        elif self.model_name == pointcontrast_name:
            return MinkowskiBackboneTrainer
        elif self.model_name == moco_name:
            return MinkowskiMocoBackboneTrainer
        elif self.model_name == byol_name:
            return MinkowskiBOYLBackboneTrainer
        elif self.model_name == cme_name:
            return CMEBackboneTrainerFull
        elif self.model_name == image_name:
            return ImageTrainer
        else:
            raise RuntimeError(self.error_msg)

    def load_from_checkpoint(self, checkpoint_path: Path):
        if minkowski_name in self.model_name:
            return MinkowskiTrainer.load_from_checkpoint(
                cfg=self.cfg,
                data_interface=self.data_interface,
                checkpoint_path=checkpoint_path,
            )

        else:
            raise RuntimeError(self.error_msg)

    def get_dataset_type(self):
        if self.dataset_name == "S3DIS":
            return MinkowskiS3DISDataset
        else:
            return MinkowskiDataset

    def get_backbone_dataset_type(self):
        if image_name == self.model_name:
            return ImagePretrainDataset
        if self.model_name == pointcontrast_name:
            return PointContrastPretrainDataset
        if minkowski_name in self.model_name:
            return MinkowskiPretrainDataset

        else:
            raise RuntimeError(self.error_msg)
