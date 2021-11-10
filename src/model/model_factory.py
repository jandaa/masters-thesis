from pathlib import Path
from omegaconf import DictConfig

import torch
from model.pointgroup import PointGroupWrapper
from model.minkowski.main import MinkovskiWrapper
from dataloaders.datasets import MinkowskiDataset, SpconvDataset
from util.types import DataInterface

pointgroup_name = "pointgroup"
minkowski_name = "minkowski"
supported_models = [pointgroup_name, minkowski_name]


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
            model_type = PointGroupWrapper
            return model_type(
                self.cfg, data_interface=self.data_interface, backbone=self.backbone
            )
        elif self.model_name == minkowski_name:
            model_type = MinkovskiWrapper
            return model_type(self.cfg)
        else:
            raise RuntimeError(self.error_msg)

    def load_from_checkpoint(self, checkpoint_path: Path):

        if self.model_name == pointgroup_name:

            # Set the epoch to that loaded in the module
            loaded_checkpoint = torch.load(checkpoint_path)
            do_instance_segmentation = False
            if loaded_checkpoint["epoch"] >= self.cfg.model.train.prepare_epochs:
                do_instance_segmentation = True

            return PointGroupWrapper.load_from_checkpoint(
                cfg=self.cfg,
                data_interface=self.data_interface,
                checkpoint_path=checkpoint_path,
                do_instance_segmentation=do_instance_segmentation,
            )

        elif self.model_name == minkowski_name:
            return MinkovskiWrapper.load_from_checkpoint(
                cfg=self.cfg,
                checkpoint_path=checkpoint_path,
            )

        else:
            raise RuntimeError(self.error_msg)

    def get_dataset_type(self):
        if self.model_name == pointgroup_name:
            return SpconvDataset
        elif self.model_name == minkowski_name:
            return MinkowskiDataset
        else:
            raise RuntimeError(self.error_msg)
