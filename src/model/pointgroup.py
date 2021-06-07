import logging
from omegaconf import DictConfig
from collections import OrderedDict
import functools
from dataclasses import dataclass

import torch
import torch.nn as nn
import pytorch_lightning as pl

import spconv
from spconv.modules import SparseModule

from lib.pointgroup_ops.functions import pointgroup_ops
import util.utils as utils

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


# @dataclass
# class PointGroupOutput:
#     """Output type of Point Group foreward function."""


class PointGroup(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.optimizer_cfg = cfg.optimizer

        # TODO: Clean up these inputs to be under single heading
        # e.g. self.cfg = cfg.pointgroup
        input_c = cfg.input_channel
        m = cfg.structure.m
        classes = cfg.classes
        block_reps = cfg.structure.block_reps
        block_residual = cfg.structure.block_residual

        # these modules should be under cfg.cluster.radius
        self.cluster_radius = cfg.group.cluster_radius
        self.cluster_meanActive = cfg.group.cluster_meanActive
        self.cluster_shift_meanActive = cfg.group.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.group.cluster_npoint_thre

        self.score_scale = cfg.train.score_scale
        self.score_fullscale = cfg.train.score_fullscale
        self.mode = cfg.train.score_mode
        self.loss_weight = cfg.train.loss_weight
        self.use_coords = cfg.structure.use_coords
        self.ignore_label = cfg.ignore_label
        self.batch_size = cfg.batch_size

        self.prepare_epochs = cfg.group.prepare_epochs

        self.pretrain_path = cfg.train.pretrain_path
        self.pretrain_module = cfg.train.pretrain_module
        self.fix_module = cfg.train.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if self.use_coords:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_c, m, kernel_size=3, padding=1, bias=False, indice_key="subm1"
            )
        )

        self.unet = UBlock(
            [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
        )

        self.output_layer = spconv.SparseSequential(norm_fn(m), nn.ReLU())

        #### semantic segmentation
        self.linear = nn.Linear(m, classes)  # bias(default): True

        #### offset
        self.offset = nn.Sequential(nn.Linear(m, m, bias=True), norm_fn(m), nn.ReLU())
        self.offset_linear = nn.Linear(m, 3, bias=True)

        #### score branch
        self.score_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(norm_fn(m), nn.ReLU())
        self.score_linear = nn.Linear(m, 1)

        self.apply(self.set_bn_init)

        module_map = {
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

        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)
        self.score_criterion = nn.BCELoss(reduction="none")

        #### fix parameter
        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        loss, loss_out, infos = self.shared_step(batch, batch_idx)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_out, infos = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss)

    def shared_step(self, batch, batch_idx):

        # Unravel batch input
        coords = batch["locs"]  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch["voxel_locs"]  # (M, 1 + 3), long, cuda
        p2v_map = batch["p2v_map"]  # (N), int, cuda
        v2p_map = batch["v2p_map"]  # (M, 1 + maxActive), int, cuda

        coords_float = batch["locs_float"]  # (N, 3), float32, cuda
        feats = batch["feats"]  # (N, C), float32, cuda
        labels = batch["labels"]  # (N), long, cuda
        instance_labels = batch[
            "instance_labels"
        ]  # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch[
            "instance_info"
        ]  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch["instance_pointnum"]  # (total_nInst), int, cuda

        batch_offsets = batch["offsets"]  # (B + 1), int, cuda

        spatial_shape = batch["spatial_shape"]

        if self.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(
            feats, v2p_map, self.mode
        )  # (M, C), float

        input_ = spconv.SparseConvTensor(
            voxel_feats, voxel_coords.int(), spatial_shape, self.batch_size
        )

        ret = self.forward(
            input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets
        )
        semantic_scores = ret["semantic_scores"]  # (N, nClass) float32, cuda
        pt_offsets = ret["pt_offsets"]  # (N, 3), float32, cuda
        # if(epoch > cfg.prepare_epochs):
        #     scores, proposals_idx, proposals_offset = ret['proposal_scores']
        #     # scores: (nProposal, 1) float, cuda
        #     # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        #     # proposals_offset: (nProposal + 1), int, cpu

        loss_inp = {}
        loss_inp["semantic_scores"] = (semantic_scores, labels)
        loss_inp["pt_offsets"] = (
            pt_offsets,
            coords_float,
            instance_info,
            instance_labels,
        )
        # if(epoch > cfg.prepare_epochs):
        #     loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)

        return self.loss_fn(loss_inp)

    def loss_fn(self, loss_inp):

        loss_out = {}
        infos = {}

        """semantic loss"""
        semantic_scores, semantic_labels = loss_inp["semantic_scores"]
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = self.semantic_criterion(semantic_scores, semantic_labels)
        loss_out["semantic_loss"] = (semantic_loss, semantic_scores.shape[0])

        """offset loss"""
        pt_offsets, coords, instance_info, instance_labels = loss_inp["pt_offsets"]
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords  # (N, 3)
        pt_diff = pt_offsets - gt_offsets  # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
        valid = (instance_labels != self.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = -(gt_offsets_ * pt_offsets_).sum(-1)  # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out["offset_norm_loss"] = (offset_norm_loss, valid.sum())
        loss_out["offset_dir_loss"] = (offset_dir_loss, valid.sum())

        """total loss"""
        loss = (
            self.loss_weight[0] * semantic_loss
            + self.loss_weight[1] * offset_norm_loss
            + self.loss_weight[2] * offset_dir_loss
        )

        return loss, loss_out, infos

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

    def forward(self, input, input_map, coords, batch_idxs, epoch):
        """
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        """
        ret = {}

        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        #### semantic segmentation
        semantic_scores = self.linear(output_feats)  # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]  # (N), long

        ret["semantic_scores"] = semantic_scores

        #### offset
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats)  # (N, 3), float32

        ret["pt_offsets"] = pt_offsets

        return ret

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def clusters_voxelization(
        self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode
    ):
        """
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        """
        c_idxs = clusters_idx[:, 1]
        clusters_feats = feats[c_idxs]
        clusters_coords = coords[c_idxs]

        clusters_coords_mean = pointgroup_ops.sec_mean(
            clusters_coords, clusters_offset
        )  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(
            clusters_coords_mean, 0, clusters_idx[:, 0].long()
        )  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(
            clusters_coords, clusters_offset
        )  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(
            clusters_coords, clusters_offset
        )  # (nCluster, 3), float

        clusters_scale = (
            1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0]
            - 0.01
        )  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(
            -1
        )  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(
            clusters_scale, 0, clusters_idx[:, 0].cuda().long()
        )

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = (
            -min_xyz
            + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda()
            + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        )
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].long())
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
            clusters_feats, out_map, mode
        )  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(
            out_feats,
            out_coords.int(),
            spatial_shape,
            int(clusters_idx[-1, 0]) + 1,
        )

        return voxelization_feats, inp_map
