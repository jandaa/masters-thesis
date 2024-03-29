import logging
from pathlib import Path
import shutil

# Load data Interfaces
from util.types import DataInterface
from datasets.scannetv2 import ScannetDataInterface
from datasets.scannetv2_small import ScannetPretrainDataInterface
from datasets.scannetv2_new import NewScannetPretrainDataInterface
from datasets.s3dis import S3DISDataInterface
from datasets.still import StillDataInterface

log = logging.getLogger(__name__)


class DataInterfaceFactory:
    """Factory that returns the interface specified by the dataset type chosen."""

    def __init__(self, cfg):
        self.dataset_cfg = cfg.dataset
        self.dataset_dir = Path(cfg.dataset_dir)

        self.output_dir = self.dataset_dir
        if cfg.output_dir:
            self.output_dir = Path(cfg.output_dir)
            if not self.output_dir.exists():
                self.output_dir.mkdir(exist_ok=True)

    def get_interface(self) -> DataInterface:
        if self.dataset_cfg.name == "scannetv2":
            return self._get_interface_scannet()
        elif self.dataset_cfg.name == "scannetv2_pretrain":
            return self._get_interface_scannet_pretrain()
        elif self.dataset_cfg.name == "scannetv2_pretrain_new":
            return self._get_interface_scannet_pretrain_new()
        elif self.dataset_cfg.name == "S3DIS":
            return self._get_interface_s3dis()
        elif self.dataset_cfg.name == "still":
            return self._get_interface_still()
        else:
            log.error(f"Unsupported dataset: {self.dataset_cfg.name}")
            raise ValueError(self.dataset_cfg.name)

    def _get_interface_scannet(self) -> DataInterface:

        tsv_file = self.dataset_dir / "scannetv2-labels.combined.tsv"
        train_split_file = self.dataset_dir / self.dataset_cfg.train_split_file
        val_split_file = self.dataset_dir / self.dataset_cfg.val_split_file
        test_split_file = self.dataset_dir / self.dataset_cfg.test_split_file

        # Copy over files to output directory
        if self.dataset_dir != self.output_dir:
            shutil.copy(tsv_file, self.output_dir / "scannetv2-labels.combined.tsv")
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
            preprocessed_path=self.output_dir,
            semantic_categories=self.dataset_cfg.categories,
            ignore_label=self.dataset_cfg.ignore_label,
            instance_ignore_classes=self.dataset_cfg.instance_ignore_categories,
            dataset_cfg=self.dataset_cfg,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )

    def _get_interface_scannet_pretrain(self) -> DataInterface:

        tsv_file = self.dataset_dir / "scannetv2-labels.combined.tsv"
        train_split_file = self.dataset_dir / self.dataset_cfg.train_split_file
        val_split_file = self.dataset_dir / self.dataset_cfg.val_split_file
        test_split_file = self.dataset_dir / self.dataset_cfg.test_split_file

        # Copy over files to output directory
        if self.dataset_dir != self.output_dir:
            shutil.copy(tsv_file, self.output_dir / "scannetv2-labels.combined.tsv")
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

        return NewScannetPretrainDataInterface(
            scans_dir=self.dataset_dir / "scans",
            preprocessed_path=self.output_dir,
            semantic_categories=self.dataset_cfg.categories,
            ignore_label=self.dataset_cfg.ignore_label,
            instance_ignore_classes=self.dataset_cfg.instance_ignore_categories,
            dataset_cfg=self.dataset_cfg,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )

    def _get_interface_scannet_pretrain_new(self) -> DataInterface:

        tsv_file = self.dataset_dir / "scannetv2-labels.combined.tsv"
        train_split_file = self.dataset_dir / self.dataset_cfg.train_split_file
        val_split_file = self.dataset_dir / self.dataset_cfg.val_split_file
        test_split_file = self.dataset_dir / self.dataset_cfg.test_split_file

        # Copy over files to output directory
        if self.dataset_dir != self.output_dir:
            shutil.copy(tsv_file, self.output_dir / "scannetv2-labels.combined.tsv")
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

        return NewScannetPretrainDataInterface(
            scans_dir=self.dataset_dir / "scans",
            preprocessed_path=self.output_dir,
            semantic_categories=self.dataset_cfg.categories,
            ignore_label=self.dataset_cfg.ignore_label,
            instance_ignore_classes=self.dataset_cfg.instance_ignore_categories,
            dataset_cfg=self.dataset_cfg,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )

    def _get_interface_s3dis(self) -> DataInterface:
        return S3DISDataInterface(
            dataset_dir=Path(self.dataset_dir),
            preprocessed_path=Path(self.output_dir),
            semantic_categories=self.dataset_cfg.categories,
            ignore_label=self.dataset_cfg.ignore_label,
            instance_ignore_classes=self.dataset_cfg.instance_ignore_categories,
            train_split=self.dataset_cfg.train_split,
            val_split=self.dataset_cfg.val_split,
            test_split=self.dataset_cfg.test_split,
        )

    def _get_interface_still(self) -> DataInterface:
        return StillDataInterface(
            dataset_dir=Path(self.dataset_dir),
            preprocessed_path=Path(self.output_dir),
            semantic_categories=self.dataset_cfg.categories,
            ignore_label=self.dataset_cfg.ignore_label,
            instance_ignore_classes=self.dataset_cfg.instance_ignore_categories,
        )
