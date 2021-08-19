import logging
from pathlib import Path

# Load data Interfaces
from util.types import DataInterface
from dataloaders.scannetv2 import ScannetDataInterface
from dataloaders.s3dis import S3DISDataInterface

log = logging.getLogger(__name__)


class DataInterfaceFactory:
    """Factory that returns the interface specified by the dataset type chosen."""

    def __init__(self, dataset_dir, dataset_cfg):
        self.dataset_cfg = dataset_cfg
        self.dataset_dir = Path(dataset_dir)

    def get_interface(self) -> DataInterface:
        if self.dataset_cfg.name == "scannetv2":
            return self._get_interface_scannet()
        elif self.dataset_cfg.name == "S3DIS":
            return self._get_interface_s3dis()
        else:
            log.error(f"Unsupported dataset: {self.dataset_cfg.name}")
            raise ValueError(self.dataset_cfg.name)

    def _get_interface_scannet(self) -> DataInterface:

        train_split_file = self.dataset_dir / self.dataset_cfg.train_split_file
        val_split_file = self.dataset_dir / self.dataset_cfg.val_split_file
        test_split_file = self.dataset_dir / self.dataset_cfg.test_split_file

        train_split = train_split_file.open().read().splitlines()
        val_split = val_split_file.open().read().splitlines()
        test_split = test_split_file.open().read().splitlines()

        return ScannetDataInterface(
            scans_dir=self.dataset_dir / "scans",
            semantic_categories=self.dataset_cfg.categories,
            ignore_label=self.dataset_cfg.ignore_label,
            instance_ignore_classes=self.dataset_cfg.instance_ignore_categories,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            # force_reload=True,
        )

    def _get_interface_s3dis(self) -> DataInterface:
        return S3DISDataInterface(
            dataset_dir=Path(self.dataset_dir),
            semantic_categories=self.dataset_cfg.categories,
            ignore_label=self.dataset_cfg.ignore_label,
            instance_ignore_classes=self.dataset_cfg.instance_ignore_categories,
            train_split=self.dataset_cfg.train_split,
            val_split=self.dataset_cfg.val_split,
            test_split=self.dataset_cfg.test_split,
            # force_reload=True,
        )
