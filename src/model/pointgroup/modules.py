import logging
from omegaconf import DictConfig
from collections import OrderedDict
import functools

import numpy as np

import torch
import torch.nn as nn

import spconv
from spconv.modules import SparseModule

from model.pointgroup.util import clusters_voxelization, non_max_suppression
from packages.pointgroup_ops.functions import pointgroup_ops
import util.utils as utils
from model.pointgroup.types import (
    PointGroupInput,
    PointGroupOutput,
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


class PointGroupBackbone(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(PointGroupBackbone, self).__init__()

        # Dataset specific parameters
        input_c = cfg.dataset.input_channel
        self.dataset_cfg = cfg.dataset

        # model parameters
        self.structure = cfg.model.structure

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if self.structure.block_residual:
            self.block = ResidualBlock
        else:
            self.block = VGGBlock

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
            self.block,
            indice_key_id=1,
        )

        self.output_layer = spconv.SparseSequential(norm_fn(m), nn.ReLU())

        self.module_map = {
            "input_conv": self.input_conv,
            "unet": self.unet,
            "output_layer": self.output_layer,
        }

        self.apply(self.set_bn_init)

        # Don't train any layers specified in fix modules
        self.fix_modules(cfg.model.train.fix_module)

    def forward(
        self,
        input: PointGroupInput,
    ):
        if self.structure.use_coords:
            features = torch.cat((input.features, input.points), 1)
        else:
            features = input.features

        voxel_feats = pointgroup_ops.voxelization(
            features, input.voxel_to_point_map, self.dataset_cfg.mode
        )

        input_ = spconv.SparseConvTensor(
            voxel_feats,
            input.voxel_coordinates.int(),
            input.spatial_shape,
            input.batch_size,
        )

        output = self.input_conv(input_)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input.point_to_voxel_map.long()]

        return output_feats

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


class PointGroup(nn.Module):
    def __init__(self, cfg: DictConfig, backbone: PointGroupBackbone = None):
        super(PointGroup, self).__init__()

        # Dataset specific parameters
        input_c = cfg.dataset.input_channel
        self.dataset_cfg = cfg.dataset

        # model parameters
        self.training_params = cfg.model.train
        self.cluster = cfg.model.cluster
        self.structure = cfg.model.structure
        self.test_cfg = cfg.model.test

        # Redefine for convenience
        m = self.structure.m

        # if a pretrained backbone exists use it otherwise start from scratch
        if backbone:
            self.backbone = backbone
        else:
            self.backbone = PointGroupBackbone(cfg)

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # semantic segmentation
        self.linear = nn.Linear(m, self.dataset_cfg.classes)  # bias(default): True

        # offset
        self.offset = nn.Sequential(nn.Linear(m, m, bias=True), norm_fn(m), nn.ReLU())
        self.offset_linear = nn.Linear(m, 3, bias=True)

        # score branch
        self.score_unet = UBlock(
            [m, 2 * m], norm_fn, 2, self.backbone.block, indice_key_id=1
        )
        self.score_outputlayer = spconv.SparseSequential(norm_fn(m), nn.ReLU())
        self.score_linear = nn.Linear(m, 1)

        self.apply(self.set_bn_init)

        self.module_map = {
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

        # Get features of the points
        output_feats = self.backbone(input)

        #### semantic segmentation
        semantic_scores = self.linear(output_feats)  # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]  # (N), long

        # return PointGroupOutput(
        #     semantic_scores=semantic_scores,
        #     point_offsets=None,
        #     proposal_scores=None,
        #     proposal_indices=None,
        #     proposal_offsets=None,
        # )

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
            coords_ = input.points[object_idxs]
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
                input.points,
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

    @staticmethod
    def init_weights(m):
        """This is for debugging purposes"""
        classname = m.__class__.__name__
        if classname.find("SubMConv3d") != -1:
            m.weight.data.fill_(1.0)

    def fix_modules(self, modules):
        for module_name in modules:
            mod = self.module_map[module_name]
            for param in mod.parameters():
                param.requires_grad = False
