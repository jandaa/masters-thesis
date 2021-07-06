import logging
from omegaconf import DictConfig
from collections import OrderedDict
import functools

from pathlib import Path
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn
import pytorch_lightning as pl

import spconv
from spconv.modules import SparseModule

from packages.pointgroup_ops.functions import pointgroup_ops
import util.utils as utils
import util.eval as eval
import util.eval_semantic as eval_semantic
from util.types import (
    DataInterface,
    PointGroupBatch,
    PointGroupInput,
    PointGroupOutput,
    LossType,
)

log = logging.getLogger(__name__)


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(
            input.features, input.indices, input.spatial_shape, input.batch_size
        )

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {
            "block{}".format(i): block(
                nPlanes[0],
                nPlanes[0],
                norm_fn,
                indice_key="subm{}".format(indice_key_id),
            )
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                ),
            )

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1
            )

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                ),
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail["block{}".format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key="subm{}".format(indice_key_id),
                )
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(
            output.features, output.indices, output.spatial_shape, output.batch_size
        )

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat(
                (identity.features, output_decoder.features), dim=1
            )

            output = self.blocks_tail(output)

        return output


class PointGroup(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(PointGroup, self).__init__()

        # Dataset specific parameters
        input_c = cfg.dataset.input_channel
        self.dataset_cfg = cfg.dataset

        # model parameters
        self.training_params = cfg.model.train
        self.cluster = cfg.model.cluster
        self.structure = cfg.model.structure
        self.test_cfg = cfg.model.test

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if self.structure.block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if self.structure.use_coords:
            input_c += 3

        # Redefine for convenience
        m = self.structure.m

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_c, m, kernel_size=3, padding=1, bias=False, indice_key="subm1"
            )
        )

        self.unet = UBlock(
            [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
            norm_fn,
            cfg.model.structure.block_reps,
            block,
            indice_key_id=1,
        )

        self.output_layer = spconv.SparseSequential(norm_fn(m), nn.ReLU())

        # semantic segmentation
        self.linear = nn.Linear(m, self.dataset_cfg.classes)  # bias(default): True

        # offset
        self.offset = nn.Sequential(nn.Linear(m, m, bias=True), norm_fn(m), nn.ReLU())
        self.offset_linear = nn.Linear(m, 3, bias=True)

        # score branch
        self.score_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(norm_fn(m), nn.ReLU())
        self.score_linear = nn.Linear(m, 1)

        self.apply(self.set_bn_init)

        self.module_map = {
            "input_conv": self.input_conv,
            "unet": self.unet,
            "output_layer": self.output_layer,
            "linear": self.linear,
            "offset": self.offset,
            "offset_linear": self.offset_linear,
            "score_unet": self.score_unet,
            "score_outputlayer": self.score_outputlayer,
            "score_linear": self.score_linear,
        }

        # Don't train any layers specified in fix modules
        self.fix_modules(cfg.model.train.fix_module)

    def forward(
        self,
        input: PointGroupInput,
        device: str,
        return_instances: bool = False,
    ):
        if self.structure.use_coords:
            features = torch.cat((input.features, input.point_coordinates), 1)
        else:
            features = input.features

        voxel_feats = pointgroup_ops.voxelization(
            features, input.voxel_to_point_map, self.dataset_cfg.mode
        )

        input_ = spconv.SparseConvTensor(
            voxel_feats,
            input.voxel_coordinates.int(),
            input.spatial_shape,
            self.dataset_cfg.batch_size,
        )

        output = self.input_conv(input_)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input.point_to_voxel_map.long()]

        #### semantic segmentation
        semantic_scores = self.linear(output_feats)  # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]  # (N), long

        #### offset
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats)  # (N, 3), float32

        scores = None
        proposals_idx = None
        proposals_offset = None
        if return_instances:

            #### get proposal clusters

            # Get indices of points that are predicted to be objects
            object_idxs = torch.nonzero(semantic_preds > 1).view(-1)

            # Points are grouped together into one vector regardless of scene
            # so need to keep track of which scene it game from
            # with batch offsets being what you need to add to the index to get correct batch
            batch_idxs_ = input.batch_indices[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
            coords_ = input.point_coordinates[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]

            semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()

            proposals_idx_shift, proposals_offset_shift = self.get_proposal_offsets(
                coords_ + pt_offsets_,
                semantic_preds_cpu,
                batch_idxs_,
                batch_offsets_,
                object_idxs,
                self.cluster.shift_meanActive,
            )

            proposals_idx, proposals_offset = self.get_proposal_offsets(
                coords_,
                semantic_preds_cpu,
                batch_idxs_,
                batch_offsets_,
                object_idxs,
                self.cluster.meanActive,
            )

            # Concatonate clustering from both P(coordinates) & Q(shifted coordinates)
            proposals_idx_shift[:, 0] += proposals_offset.size(0) - 1
            proposals_offset_shift += proposals_offset[-1]
            proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
            proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))

            #### proposals voxelization again
            input_feats, inp_map = clusters_voxelization(
                proposals_idx,
                proposals_offset,
                output_feats,
                input.point_coordinates,
                self.training_params.score_fullscale,
                self.training_params.score_scale,
                self.training_params.score_mode,
                device,
            )

            #### score
            score = self.score_unet(input_feats)
            score = self.score_outputlayer(score)
            score_feats = score.features[inp_map.long()]  # (sumNPoint, C)
            score_feats = pointgroup_ops.roipool(
                score_feats, proposals_offset.to(device)
            )  # (nProposal, C)
            scores = self.score_linear(score_feats)  # (nProposal, 1)

        return PointGroupOutput(
            semantic_scores=semantic_scores,
            point_offsets=pt_offsets,
            proposal_scores=scores,
            proposal_indices=proposals_idx,
            proposal_offsets=proposals_offset,
        )

    def get_proposal_offsets(
        self,
        points,
        semantic_predictions_cpu,
        batch_indices,
        batch_offsets,
        object_idxs,
        mean_active,
    ):
        """Get indices and offsets of proposals"""
        idx, start_len = pointgroup_ops.ballquery_batch_p(
            points,
            batch_indices,
            batch_offsets,
            self.cluster.radius,
            mean_active,
        )
        proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(
            semantic_predictions_cpu,
            idx.cpu(),
            start_len.cpu(),
            self.cluster.npoint_threshold,
        )
        proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
        # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int

        return proposals_idx, proposals_offset

    def get_clusters(self, input: PointGroupInput, output: PointGroupOutput):
        """Process the proposed clusters to get a final output of instances."""
        scores_pred = torch.sigmoid(output.proposal_scores.view(-1))

        N = input.features.shape[0]

        proposals_idx = output.proposal_indices
        proposals_offset = output.proposal_offsets

        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, N),
            dtype=torch.int,
            device=scores_pred.device,
        )
        proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

        # Takes the first point in each proposal as the semantic label
        semantic_id = output.semantic_pred[
            proposals_idx[:, 1][proposals_offset[:-1].long()].long()
        ]

        ##### score threshold
        score_mask = scores_pred > self.test_cfg.TEST_SCORE_THRESH
        scores_pred = scores_pred[score_mask]
        proposals_pred = proposals_pred[score_mask]
        semantic_id = semantic_id[score_mask]

        ##### npoint threshold
        proposals_pointnum = proposals_pred.sum(1)
        npoint_mask = proposals_pointnum > self.test_cfg.TEST_NPOINT_THRESH
        scores_pred = scores_pred[npoint_mask]
        proposals_pred = proposals_pred[npoint_mask]
        semantic_id = semantic_id[npoint_mask]

        ##### nms
        if semantic_id.shape[0] == 0:
            pick_idxs = np.empty(0)
        else:
            proposals_pred_f = proposals_pred.float()
            intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())
            proposals_pointnum = proposals_pred_f.sum(1)
            proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(
                1, proposals_pointnum.shape[0]
            )
            proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(
                proposals_pointnum.shape[0], 1
            )
            cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
            pick_idxs = non_max_suppression(
                cross_ious.cpu().numpy(),
                scores_pred.cpu().numpy(),
                self.test_cfg.TEST_NMS_THRESH,
            )

        clusters = proposals_pred[pick_idxs]
        cluster_scores = scores_pred[pick_idxs]
        cluster_semantic_id = semantic_id[pick_idxs]

        pred_info = {}
        with torch.no_grad():
            pred_info["conf"] = cluster_scores.cpu().numpy()
            pred_info["label_id"] = cluster_semantic_id.cpu().numpy()
            pred_info["mask"] = clusters.cpu().numpy()

        return pred_info

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def fix_modules(self, modules):
        for module_name in modules:
            mod = self.module_map[module_name]
            for param in mod.parameters():
                param.requires_grad = False


class PointGroupWrapper(pl.LightningModule):
    def __init__(self, cfg: DictConfig, data_interface: DataInterface):
        super().__init__()

        self.model = PointGroup(cfg)
        self.optimizer_cfg = cfg.model.optimizer

        # Dataset configuration
        self.dataset_dir = cfg.dataset_dir
        self.dataset_cfg = cfg.dataset

        # Model configuration
        self.train_cfg = cfg.model.train
        self.use_coords = cfg.model.structure.use_coords
        self.test_cfg = cfg.model.test

        self.save_point_cloud = True

        self.semantic_categories = data_interface.semantic_categories
        self.index_to_label_map = data_interface.index_to_label_map
        self.label_to_index_map = data_interface.label_to_index_map

        self.semantic_colours = [
            np.random.choice(range(256), size=3) / 255.0
            for i in range(cfg.dataset.classes + 1)
        ]

        self.semantic_criterion = nn.CrossEntropyLoss(
            ignore_index=cfg.dataset.ignore_label
        )
        self.score_criterion = nn.BCELoss(reduction="none")

    @property
    def return_instances(self):
        """
        Return whether should be using instance segmentation based on the learning curriculum
        """
        return self.current_epoch > self.train_cfg.prepare_epochs

    def step_learning_rate(
        self, optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6
    ):
        lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
        lr = 1.0
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def on_train_epoch_start(self):
        # Set learning rate for epoch
        self.step_learning_rate(
            self.trainer.optimizers[0],
            self.optimizer_cfg.lr,
            self.current_epoch - 1,
            self.train_cfg.epochs,
            multiplier=self.train_cfg.multiplier,
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

    def test_step(self, batch: PointGroupBatch, batch_idx: int):
        preds = self.model(
            batch,
            batch.device,
            return_instances=self.return_instances,
        )

        matches = {}
        matches["test_scene_name"] = batch.test_filename

        # Semantic eval & ground truth
        semantic_pred = preds.semantic_pred.detach().cpu().numpy()
        semantic_gt = batch.labels.detach().cpu().numpy()

        matches["semantic"] = {"gt": semantic_gt, "pred": semantic_pred}

        # instance eval
        if self.return_instances:
            pred_info = self.model.get_clusters(batch, preds)

            gt2pred, pred2gt = eval.assign_instances_for_scan(
                batch.test_filename,
                pred_info,
                batch.instance_labels.detach().cpu().numpy(),
                self.index_to_label_map,
            )

            matches["instance"] = {"gt": gt2pred, "pred": pred2gt}

        # Save to file
        if self.save_point_cloud:

            point_cloud_folder = Path.cwd() / "predictions"
            if not point_cloud_folder.exists():
                point_cloud_folder.mkdir()
            point_cloud_folder /= batch.test_filename
            if not point_cloud_folder.exists():
                point_cloud_folder.mkdir()

            # Set 3D points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                batch.point_coordinates.cpu().numpy()
            )

            # Save original colour inputs
            pcd.colors = o3d.utility.Vector3dVector(
                batch.features.detach().cpu().numpy()
            )
            o3d.io.write_point_cloud(str(point_cloud_folder / "input.pcd"), pcd)

            # Save semantic predictions
            self.color_point_cloud_semantic(pcd, semantic_pred)
            o3d.io.write_point_cloud(str(point_cloud_folder / "semantic_pred.pcd"), pcd)

            self.color_point_cloud_semantic(pcd, semantic_gt)
            o3d.io.write_point_cloud(str(point_cloud_folder / "semantic_gt.pcd"), pcd)

            # Save instance predictions
            if self.return_instances:
                self.color_point_cloud_instance(pcd, pred_info["mask"])
                o3d.io.write_point_cloud(
                    str(point_cloud_folder / "instance_pred.pcd"), pcd
                )

            gt_ids = batch.instance_labels.detach().cpu().numpy()
            instance_ids = set(gt_ids)

            self.color_point_cloud_instance_ground_truth(pcd, instance_ids, gt_ids)
            o3d.io.write_point_cloud(str(point_cloud_folder / "instance_gt.pcd"), pcd)

        return matches

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
        instance_colours = np.ones((len(pcd.points), 3)).astype(np.float)
        for instance_id in instance_ids:
            instance_colours[instance_predictions == instance_id] = (
                np.random.choice(range(256), size=3).astype(np.float) / 255.0
            )
        pcd.colors = o3d.utility.Vector3dVector(instance_colours)

    def color_point_cloud_instance(self, pcd, instance_masks):
        instance_colours = np.ones((len(pcd.points), 3)).astype(np.float)
        for mask in instance_masks:
            instance_colours[mask] = (
                np.random.choice(range(256), size=3).astype(np.float) / 255.0
            )
        pcd.colors = o3d.utility.Vector3dVector(instance_colours)

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
                instance_matches, self.semantic_categories
            )
            avgs = eval.compute_averages(ap_scores, self.semantic_categories)
            eval.print_results(avgs, self.semantic_categories)

    def loss_fn(self, batch, output):

        """semantic loss"""
        semantic_scores = output.semantic_scores
        semantic_labels = batch.labels

        semantic_loss = self.semantic_criterion(semantic_scores, semantic_labels)

        """offset loss"""
        gt_offsets = batch.instance_centers - batch.point_coordinates  # (N, 3)
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

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx) -> None:
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        if self.optimizer_cfg.type == "Adam":
            return torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.optimizer_cfg.lr,
            )
        elif self.optimizer_cfg.type == "SGD":
            return torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.optimizer_cfg.lr,
                momentum=self.optimizer_cfg.momentum,
                weight_decay=self.optimizer_cfg.weight_decay,
            )
        else:
            # TODO: Put error logging at high level try catch block
            log.error(f"Invalid optimizer type: {self.optimizer_type}")
            raise ValueError(f"Invalid optimizer type: {self.optimizer_type}")


def clusters_voxelization(
    clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode, output_device
):
    """
    :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
    :param clusters_offset: (nCluster + 1), int, cpu
    :param feats: (N, C), float, cuda
    :param coords: (N, 3), float, cuda
    :return:
    """
    c_idxs = clusters_idx[:, 1].to(output_device)
    clusters_feats = feats[c_idxs.long()]
    clusters_coords = coords[c_idxs.long()]

    clusters_coords_mean = pointgroup_ops.sec_mean(
        clusters_coords, clusters_offset.to(output_device)
    )  # (nCluster, 3), float
    clusters_coords_mean = torch.index_select(
        clusters_coords_mean, 0, clusters_idx[:, 0].to(output_device).long()
    )  # (sumNPoint, 3), float
    clusters_coords -= clusters_coords_mean

    clusters_coords_min = pointgroup_ops.sec_min(
        clusters_coords, clusters_offset.to(output_device)
    )  # (nCluster, 3), float
    clusters_coords_max = pointgroup_ops.sec_max(
        clusters_coords, clusters_offset.to(output_device)
    )  # (nCluster, 3), float

    clusters_scale = (
        1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01
    )  # (nCluster), float
    clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

    min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
    max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

    clusters_scale = torch.index_select(
        clusters_scale, 0, clusters_idx[:, 0].to(output_device).long()
    )

    clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

    range = max_xyz - min_xyz
    offset = (
        -min_xyz
        + torch.clamp(fullscale - range - 0.001, min=0)
        * torch.rand(3).to(output_device)
        + torch.clamp(fullscale - range + 0.001, max=0)
        * torch.rand(3).to(output_device)
    )
    offset = torch.index_select(offset, 0, clusters_idx[:, 0].to(output_device).long())
    clusters_coords += offset
    assert (
        clusters_coords.shape.numel()
        == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()
    )

    clusters_coords = clusters_coords.long()
    clusters_coords = torch.cat(
        [clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1
    )  # (sumNPoint, 1 + 3)

    out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(
        clusters_coords, int(clusters_idx[-1, 0]) + 1, mode
    )
    # output_coords: M * (1 + 3) long
    # input_map: sumNPoint int
    # output_map: M * (maxActive + 1) int

    out_feats = pointgroup_ops.voxelization(
        clusters_feats, out_map.to(output_device), mode
    )  # (M, C), float, cuda

    spatial_shape = [fullscale] * 3
    voxelization_feats = spconv.SparseConvTensor(
        out_feats,
        out_coords.int().to(output_device),
        spatial_shape,
        int(clusters_idx[-1, 0]) + 1,
    )

    return voxelization_feats, inp_map


def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
    """
    :param scores: (N), float, 0~1
    :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
    """
    fg_mask = scores > fg_thresh
    bg_mask = scores < bg_thresh
    interval_mask = (fg_mask == 0) & (bg_mask == 0)

    segmented_scores = (fg_mask > 0).float()
    k = 1 / (fg_thresh - bg_thresh)
    b = bg_thresh / (bg_thresh - fg_thresh)
    segmented_scores[interval_mask] = scores[interval_mask] * k + b

    return segmented_scores


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)
