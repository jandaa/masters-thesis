"""
Base modules for models to inherit from.
"""

import logging
from pathlib import Path
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
import numpy as np
import open3d as o3d

import util.utils as utils
from util.types import DataInterface
import util.eval as eval
import util.eval_semantic as eval_semantic

log = logging.getLogger(__name__)


class SegmentationModule(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        data_interface: DataInterface,
    ):
        super().__init__()

        self.optimizer_cfg = cfg.model.optimizer
        self.scheduler_cfg = cfg.model.scheduler

        # Dataset configuration
        self.dataset_dir = cfg.dataset_dir
        self.dataset_cfg = cfg.dataset

        # Model configuration
        self.train_cfg = cfg.model.train
        self.test_cfg = cfg.model.test

        self.semantic_categories = data_interface.semantic_categories
        self.instance_categories = data_interface.instance_categories
        self.instance_index_to_label_map = {
            k: v
            for k, v in data_interface.index_to_label_map.items()
            if v in self.instance_categories
        }

        self.semantic_colours = [
            np.random.choice(range(256), size=3) / 255.0
            for i in range(cfg.dataset.classes + 1)
        ]

        self.ignore_label_colour = utils.get_random_colour()

    @property
    def return_instances(self):
        raise NotImplementedError()

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx) -> None:
        torch.cuda.empty_cache()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx) -> None:
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        if self.optimizer_cfg.type == "Adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.optimizer_cfg.lr,
            )
        elif self.optimizer_cfg.type == "SGD":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.optimizer_cfg.lr,
                momentum=self.optimizer_cfg.momentum,
                weight_decay=self.optimizer_cfg.weight_decay,
            )
        else:
            # TODO: Put error logging at high level try catch block
            log.error(f"Invalid optimizer type: {self.optimizer_type}")
            raise ValueError(f"Invalid optimizer type: {self.optimizer_type}")

        # Get scheduler if any
        if not self.scheduler_cfg.type:
            log.info("No learning rate schedular specified")
            return optimizer
        if self.scheduler_cfg.type == "ExpLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, self.scheduler_cfg.exp_gamma
            )
        else:
            log.error(f"Invalid scheduler type: {self.scheduler_cfg.type}")
            raise ValueError(f"Invalid scheduler type: {self.scheduler_cfg.type}")

        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs):
        semantic_matches = {}
        for i, output in enumerate(outputs):
            scene_name = f"scene{i}"
            semantic_matches[scene_name] = output

        mean_iou = eval_semantic.evaluate(semantic_matches, verbose=True)
        self.log("val_semantic_mIOU", mean_iou, sync_dist=True)

    def get_matches_val(self, batch, output):
        """Get all gt and predictions for validation"""
        semantic_pred = output.semantic_pred.detach().cpu().numpy()
        semantic_gt = batch.labels.detach().cpu().numpy()
        semantic_matches = {"gt": semantic_gt, "pred": semantic_pred}

        return semantic_matches

    def test_epoch_end(self, outputs) -> None:

        # Semantic eval
        semantic_matches = {}
        for output in outputs:
            scene_name = output["test_scene_name"]
            semantic_matches[scene_name] = {}
            semantic_matches[scene_name]["gt"] = output["semantic"]["gt"]
            semantic_matches[scene_name]["pred"] = output["semantic"]["pred"]

        eval_semantic.evaluate(semantic_matches)

        # Instance eval
        if self.return_instances:
            instance_matches = {}
            for output in outputs:
                scene_name = output["test_scene_name"]
                instance_matches[scene_name] = {}
                instance_matches[scene_name]["gt"] = output["instance"]["gt"]
                instance_matches[scene_name]["pred"] = output["instance"]["pred"]

            ap_scores = eval.evaluate_matches(
                instance_matches, self.instance_categories
            )
            avgs = eval.compute_averages(ap_scores, self.instance_categories)
            eval.print_results(avgs, self.instance_categories)

    def get_matches_test(self, batch, preds, pred_info=None):
        """Generate test-time prediction to gt matches"""

        matches = {}
        matches["test_scene_name"] = batch.test_filename

        # Semantic eval & ground truth
        semantic_pred = preds.semantic_pred.detach().cpu().numpy()
        semantic_gt = batch.labels.detach().cpu().numpy()

        matches["semantic"] = {"gt": semantic_gt, "pred": semantic_pred}

        # instance eval
        if self.return_instances:
            if not pred_info:
                raise RuntimeError("Missing pred_info")

            gt2pred, pred2gt = eval.assign_instances_for_scan(
                batch.test_filename,
                pred_info,
                batch.instance_labels.detach().cpu().numpy(),
                batch.labels.detach().cpu().numpy(),
                self.instance_index_to_label_map,
            )

            matches["instance"] = {"gt": gt2pred, "pred": pred2gt}

        return matches

    def save_pointcloud(self, batch, preds, pred_info=None):
        """Save point cloud predictions and ground truth"""

        # Semantic eval & ground truth
        semantic_pred = preds.semantic_pred.detach().cpu().numpy()
        semantic_gt = batch.labels.detach().cpu().numpy()

        point_cloud_folder = Path.cwd() / "predictions"
        if not point_cloud_folder.exists():
            point_cloud_folder.mkdir()
        point_cloud_folder /= batch.test_filename
        if not point_cloud_folder.exists():
            point_cloud_folder.mkdir()

        # Set 3D points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(batch.points.detach().cpu().numpy())

        # Save original colour inputs
        pcd.colors = o3d.utility.Vector3dVector(batch.features.detach().cpu().numpy())
        o3d.io.write_point_cloud(str(point_cloud_folder / "input.pcd"), pcd)

        # Save semantic predictions
        self.color_point_cloud_semantic(pcd, semantic_pred)
        o3d.io.write_point_cloud(str(point_cloud_folder / "semantic_pred.pcd"), pcd)

        self.color_point_cloud_semantic(pcd, semantic_gt)
        o3d.io.write_point_cloud(str(point_cloud_folder / "semantic_gt.pcd"), pcd)

        # Save instance predictions
        if self.return_instances and pred_info:

            gt_ids = batch.instance_labels.detach().cpu().numpy()
            instance_ids = set(gt_ids)

            gt_instance_colours = self.color_point_cloud_instance_ground_truth(
                pcd, instance_ids, gt_ids
            )
            o3d.io.write_point_cloud(str(point_cloud_folder / "instance_gt.pcd"), pcd)

            self.color_point_cloud_instance(pcd, pred_info["mask"], gt_instance_colours)
            o3d.io.write_point_cloud(str(point_cloud_folder / "instance_pred.pcd"), pcd)

    def color_point_cloud_semantic(self, pcd, predictions):
        semantic_colours = (
            np.ones((len(pcd.points), 3)).astype(np.float) * self.semantic_colours[-1]
        )
        for class_ind in range(self.dataset_cfg.classes):
            semantic_colours[predictions == class_ind] = self.semantic_colours[
                class_ind
            ]
        pcd.colors = o3d.utility.Vector3dVector(semantic_colours)

    def color_point_cloud_instance_ground_truth(
        self, pcd, instance_ids, instance_predictions
    ):
        instance_colours = (
            np.ones((len(pcd.points), 3)).astype(np.float) * utils.get_random_colour()
        )
        for instance_id in instance_ids:
            if instance_id == self.dataset_cfg.ignore_label:
                colour = self.ignore_label_colour
            else:
                colour = utils.get_random_colour()

            instance_colours[instance_predictions == instance_id] = colour
        pcd.colors = o3d.utility.Vector3dVector(instance_colours)
        return instance_colours

    def color_point_cloud_instance(self, pcd, instance_masks, gt_instance_colours):
        instance_colours = (
            np.ones((len(pcd.points), 3)).astype(np.float) * self.ignore_label_colour
        )
        colours_used = []
        for mask in instance_masks:
            mask = mask == 1
            colour = self.get_most_common_colour(gt_instance_colours[mask])

            # If colour already used then pick another random colour that
            # has not already been used.
            while any(all(colour == colour_used) for colour_used in colours_used):
                colour = utils.get_random_colour()
            colours_used.append(colour)

            instance_colours[mask] = colour
        pcd.colors = o3d.utility.Vector3dVector(instance_colours)

    def get_most_common_colour(self, colours):
        unique, counts = np.unique(colours, return_counts=True, axis=0)

        max_count = max(counts)
        return unique[counts == max_count][0]
