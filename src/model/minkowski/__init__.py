# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import logging
from omegaconf import DictConfig
import functools

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np

from model.modules import SegmentationModule, BackboneModule
from model.minkowski.res16unet import Res16UNet34C

from util.types import DataInterface
from model.minkowski.types import (
    MinkowskiInput,
    MinkowskiOutput,
    MinkowskiPretrainInput,
)
from util.utils import NCESoftmaxLoss

log = logging.getLogger(__name__)


class MinkovskiSemantic(nn.Module):
    def __init__(self, cfg: DictConfig, backbone=None):
        nn.Module.__init__(self)

        self.dataset_cfg = cfg.dataset
        self.feature_dim = cfg.model.net.model_n_out

        if backbone:
            self.backbone = backbone
        else:
            self.backbone = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)
        self.linear = ME.MinkowskiLinear(
            self.feature_dim, self.dataset_cfg.classes, bias=False
        )

    def forward(self, input):
        """Extract features and predict semantic class."""
        output = self.backbone(input)
        output = self.linear(output)
        return MinkowskiOutput(semantic_scores=output.F)


class MinkowskiBackboneModule(BackboneModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(MinkowskiBackboneModule, self).__init__(cfg)

        self.feature_dim = cfg.model.net.model_n_out
        self.model = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)

    def forward(self, batch: MinkowskiInput):
        model_input = ME.SparseTensor(batch.features.float(), batch.points)
        output = self.model(model_input)
        return output

    def training_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features.float(), batch.points)
        output = self.model(model_input)
        loss = self.loss_fn(batch, output)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def validation_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features, batch.points)
        output = self.model(model_input)
        loss = self.loss_fn(batch, output)
        self.log("val_loss", loss, sync_dist=True)

    def loss_fn(self, batch, output):
        tau = 0.4
        max_pos = 4092

        # model_input = ME.SparseTensor(batch.features, batch.points)

        # Get all positive and negative pairs
        qs, ks = [], []
        for i, matches in enumerate(batch.correspondences):
            voxel_indices_1 = [v for v, _ in matches]
            voxel_indices_2 = [v for _, v in matches]
            output_batch_1 = output.features_at(2 * i)
            output_batch_2 = output.features_at(2 * i + 1)
            q = output_batch_1[voxel_indices_1]
            k = output_batch_2[voxel_indices_2]

            # points1 = model_input.coordinates_at(2 * i).detach().cpu().numpy()

            # visualize_correspondances(model_input, i, matches)

            qs.append(q)
            ks.append(k)

        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)

        if q.shape[0] > max_pos:
            inds = np.random.choice(q.shape[0], max_pos, replace=False)
            q = q[inds]
            k = k[inds]

        # Labels
        npos = q.shape[0]
        labels = torch.arange(npos).to(batch.device).long()

        logits = torch.mm(q, k.transpose(1, 0))  # npos by npos
        out = torch.div(logits, tau)
        out = out.squeeze().contiguous()

        return self.criterion(out, labels)


import open3d as o3d
import random
from util.utils import get_random_colour


def visualize_correspondances(model_input, i, correspondances):
    """Visualize the point correspondances between the matched scans in
    the pretrain input"""

    # for i, matches in enumerate(pretrain_input.correspondances):
    points1 = model_input.coordinates_at(2 * i).detach().cpu().numpy()
    colors1 = model_input.features_at(2 * i).detach().cpu().numpy()

    points2 = model_input.coordinates_at(2 * i + 1).detach().cpu().numpy()
    colors2 = model_input.features_at(2 * i + 1).detach().cpu().numpy()

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)
    pcd2 = pcd2.translate([100.0, 0, 0])

    correspondences = random.choices(correspondances, k=100)
    lineset = o3d.geometry.LineSet()
    lineset = lineset.create_from_point_cloud_correspondences(
        pcd1, pcd2, correspondences
    )
    colors = [get_random_colour() for i in range(len(correspondences))]
    lineset.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd1, pcd2, lineset])


class MinkowskiModule(SegmentationModule):
    def __init__(self, cfg: DictConfig, data_interface: DataInterface, backbone=None):
        super(MinkowskiModule, self).__init__(cfg, data_interface)

        self.model = MinkovskiSemantic(cfg, backbone=backbone)
        self.criterion = NCESoftmaxLoss()
        self.semantic_criterion = nn.CrossEntropyLoss(
            ignore_index=cfg.dataset.ignore_label
        )

    @property
    def return_instances(self):
        return False

    def training_step(self, batch: MinkowskiInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features, batch.points)
        output = self.model(model_input)
        loss = self.loss_fn(batch, output)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def validation_step(self, batch: MinkowskiInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features, batch.points)
        output = self.model(model_input)
        loss = self.loss_fn(batch, output)
        self.log("val_loss", loss, sync_dist=True)

        return self.get_matches_val(batch, output)

    def forward(self, batch: MinkowskiInput):
        model_input = ME.SparseTensor(batch.features, batch.points)
        output = self.model(model_input)
        return output

    def loss_fn(self, batch, output):
        """Just return the semantic loss"""
        semantic_scores = output.semantic_scores
        semantic_labels = batch.labels.long()
        loss = self.semantic_criterion(semantic_scores, semantic_labels)
        return loss

    def test_step(self, batch: MinkowskiInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features, batch.points)
        preds = self.model(model_input)

        # Remove batch index from points
        batch.points = batch.points[:, 1:4]

        # Save point cloud
        if self.test_cfg.save_point_cloud:
            self.save_pointcloud(batch, preds)

        return self.get_matches_test(batch, preds)
