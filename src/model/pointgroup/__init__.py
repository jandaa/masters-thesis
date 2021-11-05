import logging
import functools
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl

from model.modules import SegmentationModule
from util.types import DataInterface, PointGroupBatch, LossType, PretrainInput
from model.pointgroup.modules import PointGroup, PointGroupBackbone
from model.pointgroup.util import NCESoftmaxLoss, get_segmented_scores

from packages.pointgroup_ops.functions import pointgroup_ops

log = logging.getLogger(__name__)


class PointgroupModule(SegmentationModule):
    def __init__(
        self,
        cfg: DictConfig,
        data_interface: DataInterface,
        backbone: PointGroupBackbone = None,
        do_instance_segmentation: bool = False,
    ):
        super(PointgroupModule, self).__init__(cfg, data_interface)

        self.model = PointGroup(cfg, backbone=backbone)
        self.do_instance_segmentation = do_instance_segmentation
        # self.do_instance_segmentation = True

        self.semantic_criterion = nn.CrossEntropyLoss(
            ignore_index=cfg.dataset.ignore_label
        )
        self.score_criterion = nn.BCELoss(reduction="none")

    @property
    def return_instances(self):
        """
        Return whether should be using instance segmentation based on the learning curriculum
        """
        return (
            self.current_epoch > self.train_cfg.prepare_epochs
            or self.do_instance_segmentation
        )

    def training_step(self, batch: PointGroupBatch, batch_idx: int):
        output = self.model(batch, batch.device, return_instances=self.return_instances)
        loss = self.loss_fn(batch, output)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss.total_loss)
        log("semantic_loss", loss.semantic_loss)
        log("offset_norm_loss", loss.offset_norm_loss)
        log("offset_dir_loss", loss.offset_dir_loss)
        if self.return_instances:
            log("score_loss", loss.score_loss)

        return loss.total_loss

    def validation_step(self, batch: PointGroupBatch, batch_idx: int):
        output = self.model(batch, batch.device, return_instances=self.return_instances)
        loss = self.loss_fn(batch, output)
        self.log("val_loss", loss.total_loss, sync_dist=True)

        return self.get_matches_val(batch, output)

    def test_step(self, batch: PointGroupBatch, batch_idx: int):
        preds = self.model(
            batch,
            batch.device,
            return_instances=self.return_instances,
        )

        pred_info = None
        if self.return_instances:
            pred_info = self.model.get_clusters(batch, preds)

        # Save point cloud
        if self.test_cfg.save_point_cloud:
            self.save_pointcloud(batch, preds, pred_info=pred_info)

        return self.get_matches_test(batch, preds, pred_info=pred_info)

    def loss_fn(self, batch, output):

        """semantic loss"""
        semantic_scores = output.semantic_scores
        semantic_labels = batch.labels

        semantic_loss = self.semantic_criterion(semantic_scores, semantic_labels)

        # return LossType(
        #     semantic_loss=semantic_loss,
        #     offset_norm_loss=0,
        #     offset_dir_loss=0,
        #     score_loss=0,
        #     number_of_instances=0,
        #     number_of_points=semantic_scores.shape[0],
        #     number_of_valid_labels=0,
        #     total_loss=semantic_loss,
        # )

        """offset loss"""
        gt_offsets = batch.instance_centers - batch.points  # (N, 3)
        pt_diff = output.point_offsets - gt_offsets  # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
        valid = (batch.instance_labels != self.dataset_cfg.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(output.point_offsets, p=2, dim=1)
        pt_offsets_ = output.point_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = -(gt_offsets_ * pt_offsets_).sum(-1)  # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        """score loss"""
        score_loss = 0
        number_of_instances = 0
        if self.return_instances:

            scores = output.proposal_scores
            proposals_idx = output.proposal_indices
            proposals_offset = output.proposal_offsets
            instance_pointnum = batch.instance_pointnum

            ious = pointgroup_ops.get_iou(
                proposals_idx[:, 1].to(self.device),
                proposals_offset.to(self.device),
                batch.instance_labels,
                instance_pointnum,
            )  # (nProposal, nInstance), float
            gt_ious, _ = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(
                gt_ious, self.train_cfg.fg_thresh, self.train_cfg.bg_thresh
            )

            score_loss = self.score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            number_of_instances = gt_ious.shape[0]

        """total loss"""
        loss = (
            self.train_cfg.loss_weight[0] * semantic_loss
            + self.train_cfg.loss_weight[1] * offset_norm_loss
            + self.train_cfg.loss_weight[2] * offset_dir_loss
        )

        if self.return_instances:
            loss += self.train_cfg.loss_weight[3] * score_loss

        return LossType(
            semantic_loss=semantic_loss,
            offset_norm_loss=offset_norm_loss,
            offset_dir_loss=offset_dir_loss,
            score_loss=score_loss,
            number_of_instances=number_of_instances,
            number_of_points=semantic_scores.shape[0],
            number_of_valid_labels=int(valid.sum()),
            total_loss=loss,
        )


class PointGroupBackboneModule(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        self.model = PointGroupBackbone(cfg)
        self.optimizer_cfg = cfg.model.optimizer

        # Dataset configuration
        self.dataset_dir = cfg.dataset_dir
        self.dataset_cfg = cfg.dataset

        # Model configuration
        self.train_cfg = cfg.model.train
        self.use_coords = cfg.model.structure.use_coords
        self.test_cfg = cfg.model.test

        self.criterion = NCESoftmaxLoss()

    def training_step(self, batch: PretrainInput, batch_idx: int):
        output = self.model(batch)
        loss = self.loss_fn(batch, output)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def validation_step(self, batch: PretrainInput, batch_idx: int):
        output = self.model(batch)
        loss = self.loss_fn(batch, output)
        self.log("val_loss", loss, sync_dist=True)

    def loss_fn(self, batch: PretrainInput, output):
        tau = 0.07

        loss = 0
        for i, matches in enumerate(batch.correspondances):
            points1_start = batch.offsets[2 * i]
            points2_start = batch.offsets[2 * i + 1]

            point_indices_1 = [points1_start + points1 for points1 in matches.keys()]
            point_indices_2 = [
                points2_start + matches[points1] for points1 in matches.keys()
            ]
            q = output[point_indices_1]
            k = output[point_indices_2]

            # Labels
            npos = len(matches.keys())
            labels = torch.arange(npos).cuda().long()

            logits = torch.mm(q, k.transpose(1, 0))  # npos by npos
            out = torch.div(logits, tau)
            out = out.squeeze().contiguous()

            loss += self.criterion(out, labels)

        return loss / len(batch.correspondances)

    def configure_optimizers(self):
        if self.optimizer_cfg.type == "Adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.dataset_cfg.pretrain.learning_rate,
            )
        elif self.optimizer_cfg.type == "SGD":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.dataset_cfg.pretrain.learning_rate,
                momentum=self.optimizer_cfg.momentum,
                weight_decay=self.optimizer_cfg.weight_decay,
            )
        else:
            # TODO: Put error logging at high level try catch block
            log.error(f"Invalid optimizer type: {self.optimizer_type}")
            raise ValueError(f"Invalid optimizer type: {self.optimizer_type}")

        scheduler = ExponentialLR(optimizer, 0.99)
        return [optimizer], [scheduler]
