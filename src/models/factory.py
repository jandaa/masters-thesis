from pathlib import Path
from omegaconf import DictConfig

# import torch

# from models.pointgroup.trainer import PointgroupTrainer
# from models.pointgroup.dataset import SpconvDataset
from models.minkowski.trainer import (
    MinkowskiTrainer,
    MinkowskiBackboneTrainer,
    MinkowskiMocoBackboneTrainer,
    MinkowskiBOYLBackboneTrainer,
)
from models.minkowski.dataset import (
    MinkowskiDataset,
    MinkowskiPretrainDataset,
    MinkowskiEntropyPretrainDataset,
)
from util.types import DataInterface

pointgroup_name = "pointgroup"
minkowski_name = "minkowski"
moco_name = "minkowski_moco"
byol_name = "minkowski_byol"
supported_models = [pointgroup_name, minkowski_name, moco_name, byol_name]


class ModelFactory:
    def __init__(self, cfg: DictConfig, data_interface: DataInterface, backbone=None):
        self.model_name = cfg.model.name
        self.cfg = cfg
        self.data_interface = data_interface
        self.backbone = backbone

        # Ensure that the model is supported
        self.error_msg = f"model {self.model_name} is not supported"
        if self.model_name not in supported_models:
            raise RuntimeError(self.error_msg)

    def get_model(self):
        if self.model_name == pointgroup_name:
            raise NotImplementedError("No pointgroup")
            # return PointgroupTrainer(
            #     self.cfg, data_interface=self.data_interface, backbone=self.backbone
            # )
        elif minkowski_name in self.model_name:
            return MinkowskiTrainer(
                self.cfg, self.data_interface, backbone=self.backbone
            )
        else:
            raise RuntimeError(self.error_msg)

    def get_backbone_wrapper_type(self):
        if self.model_name == minkowski_name:
            return MinkowskiBackboneTrainer
        elif self.model_name == moco_name:
            return MinkowskiMocoBackboneTrainer
        elif self.model_name == byol_name:
            return MinkowskiBOYLBackboneTrainer
        else:
            raise RuntimeError(self.error_msg)

    def load_from_checkpoint(self, checkpoint_path: Path):

        if self.model_name == pointgroup_name:
            raise NotImplementedError("No pointgroup")
            # # Set the epoch to that loaded in the module
            # loaded_checkpoint = torch.load(checkpoint_path)
            # do_instance_segmentation = False
            # if loaded_checkpoint["epoch"] >= self.cfg.model.train.prepare_epochs:
            #     do_instance_segmentation = True

            # return PointgroupTrainer.load_from_checkpoint(
            #     cfg=self.cfg,
            #     data_interface=self.data_interface,
            #     checkpoint_path=checkpoint_path,
            #     do_instance_segmentation=do_instance_segmentation,
            # )

        elif minkowski_name in self.model_name:
            return MinkowskiTrainer.load_from_checkpoint(
                cfg=self.cfg,
                data_interface=self.data_interface,
                checkpoint_path=checkpoint_path,
            )

        else:
            raise RuntimeError(self.error_msg)

    def get_dataset_type(self):
        if self.model_name == pointgroup_name:
            raise NotImplementedError("No pointgroup")
            # return SpconvDataset
        elif minkowski_name in self.model_name:
            return MinkowskiDataset
        else:
            raise RuntimeError(self.error_msg)

    def get_backbone_dataset_type(self):
        if self.model_name == pointgroup_name:
            raise RuntimeError(self.error_msg)
        elif minkowski_name in self.model_name:
            return MinkowskiEntropyPretrainDataset
        else:
            raise RuntimeError(self.error_msg)
