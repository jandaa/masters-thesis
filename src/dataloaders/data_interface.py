import logging
from pathlib import Path
import shutil

# Load data Interfaces
from util.types import DataInterface
from dataloaders.scannetv2 import ScannetDataInterface
from dataloaders.s3dis import S3DISDataInterface

log = logging.getLogger(__name__)


class DataInterfaceFactory:
    """Factory that returns the interface specified by the dataset type chosen."""

    def __init__(self, cfg):
        self.force_reload = cfg.force_reload
        self.dataset_cfg = cfg.dataset
        self.dataset_dir = Path(cfg.dataset_dir)

        self.output_dir = self.dataset_dir
        if cfg.output_dir:
            self.output_dir = Path(cfg.output_dir)
            if not self.output_dir.exists():
                self.output_dir.mkdir()

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

        # Copy over files to output directory
        if self.dataset_dir != self.output_dir:
            shutil.copy(
                train_split_file, self.output_dir / self.dataset_cfg.train_split_file
            )
            shutil.copy(
                val_split_file, self.output_dir / self.dataset_cfg.val_split_file
            )
            shutil.copy(
                test_split_file, self.output_dir / self.dataset_cfg.test_split_file
            )

        train_split = train_split_file.open().read().splitlines()
        val_split = val_split_file.open().read().splitlines()
        test_split = test_split_file.open().read().splitlines()

        return ScannetDataInterface(
            scans_dir=self.dataset_dir / "scans",
            output_path=self.output_dir,
            semantic_categories=self.dataset_cfg.categories,
            ignore_label=self.dataset_cfg.ignore_label,
            instance_ignore_classes=self.dataset_cfg.instance_ignore_categories,
            dataset_cfg=self.dataset_cfg,
            force_reload=self.force_reload,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
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
            force_reload=self.force_reload,
        )
