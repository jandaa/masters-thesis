import logging
from omegaconf import DictConfig
import functools

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
from scipy.stats import wasserstein_distance

from models.trainers import SegmentationTrainer, BackboneTrainer
from models.minkowski.modules.res16unet import Res16UNet34C
from models.minkowski.modules.resnet import get_norm
from models.minkowski.modules.common import NormType

from util.types import DataInterface
from models.minkowski.types import (
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
        self.bn_momentum = cfg.model.net.bn_momentum
        self.norm_type = NormType.BATCH_NORM

        # Backbone
        if backbone:
            self.backbone = backbone
        else:
            self.backbone = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)

        # # Projection head
        self.linear1 = ME.MinkowskiLinear(
            self.feature_dim,
            self.dataset_cfg.classes,
        )
        # self.bn1 = get_norm(
        #     self.norm_type, self.dataset_cfg.classes, 3, bn_momentum=self.bn_momentum
        # )

        # self.linear2 = ME.MinkowskiLinear(
        #     self.dataset_cfg.classes,
        #     self.dataset_cfg.classes,
        # )
        # self.bn2 = get_norm(
        #     self.norm_type, self.dataset_cfg.classes, 3, bn_momentum=self.bn_momentum
        # )
        # self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, input):
        """Extract features and predict semantic class."""

        # Get backbone features
        output = self.backbone(input)

        # # Run features through 2-layer non-linear projection head
        output = self.linear1(output)
        # output = self.bn1(output)
        # output = self.relu(output)
        # output = self.linear2(output)
        # output = self.bn2(output)
        # output = self.relu(output)

        return MinkowskiOutput(output=output, semantic_scores=output.F)


class MinkowskiBackboneTrainer(BackboneTrainer):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(MinkowskiBackboneTrainer, self).__init__(cfg)

        self.feature_dim = cfg.model.net.model_n_out
        self.model = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)
        # self.model = MinkovskiSemantic(cfg)

        # which training loss to use
        if cfg.model.net.loss == "new":
            self.loss_fn = self.loss_fn_new
        else:
            self.loss_fn = self.loss_fn_original

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

    def loss_fn_new(self, batch, output):
        tau = 0.4
        max_pos = 4092
        n = 4092

        # Get all positive and negative pairs
        qs, ks = [], []
        for i, matches in enumerate(batch.correspondences):
            voxel_indices_1 = [match["frame1"]["voxel_inds"] for match in matches]
            voxel_indices_2 = [match["frame2"]["voxel_inds"] for match in matches]

            output_batch_1 = output.features_at(2 * i)
            output_batch_2 = output.features_at(2 * i + 1)
            q = output_batch_1[voxel_indices_1]
            k = output_batch_2[voxel_indices_2]

            qs.append(q)
            ks.append(k)

        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)

        # normalize to unit vectors
        q = q / torch.norm(q, p=2, dim=1, keepdim=True)
        k = k / torch.norm(k, p=2, dim=1, keepdim=True)

        if q.shape[0] > max_pos:
            inds = np.random.choice(q.shape[0], max_pos, replace=False)
            q = q[inds]
            k = k[inds]

        pos = torch.exp(torch.sum(q * k, dim=-1) / tau)
        combined = torch.exp(torch.mm(q, k.t().contiguous()) / tau)

        Ng = torch.zeros(q.shape[0], device=q.device)
        for ind in range(q.shape[0]):

            # select the negative values
            neg = combined.index_select(0, torch.tensor([ind], device=q.device))
            Ng[ind] = neg.mean(dim=-1) * n

        loss = (-torch.log(pos / (pos + Ng))).mean()

        return loss

    def loss_fn_original(self, batch, output):
        tau = 0.4
        max_pos = 4092

        # Get all positive and negative pairs
        qs, ks = [], []
        for i, matches in enumerate(batch.correspondences):
            voxel_indices_1 = [match["frame1"]["voxel_inds"] for match in matches]
            voxel_indices_2 = [match["frame2"]["voxel_inds"] for match in matches]

            output_batch_1 = output.features_at(2 * i)
            output_batch_2 = output.features_at(2 * i + 1)
            q = output_batch_1[voxel_indices_1]
            k = output_batch_2[voxel_indices_2]

            # visualize_mapping(points1, points2, voxel_indices_1)

            qs.append(q)
            ks.append(k)

        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)

        if q.shape[0] > max_pos:
            inds = np.random.choice(q.shape[0], max_pos, replace=False)
            q = q[inds]
            k = k[inds]

        # normalize to unit vectors
        q = q / torch.norm(q, p=2, dim=1, keepdim=True)
        k = k / torch.norm(k, p=2, dim=1, keepdim=True)

        # Labels
        npos = q.shape[0]
        labels = torch.arange(npos).to(batch.device).long()

        logits = torch.mm(q, k.transpose(1, 0))  # npos by npos
        out = torch.div(logits, tau)
        out = out.squeeze().contiguous()

        return self.criterion(out, labels)

    def loss_fn_delta(self, batch, output):
        tau = 0.4
        max_pos = 50
        n = 4092

        output = output.output

        # Get all positive and negative pairs
        qs, ks = [], []
        qs_fpfh, ks_fpfh = [], []
        for i, matches in enumerate(batch.correspondences):
            voxel_indices_1 = [match["frame1"]["voxel_inds"] for match in matches]
            voxel_indices_2 = [match["frame2"]["voxel_inds"] for match in matches]

            fpfh_1 = [match["frame1"]["fpfh"] for match in matches]
            fpfh_2 = [match["frame2"]["fpfh"] for match in matches]

            output_batch_1 = output.features_at(2 * i)
            output_batch_2 = output.features_at(2 * i + 1)
            q = output_batch_1[voxel_indices_1]
            k = output_batch_2[voxel_indices_2]

            qs.append(q)
            ks.append(k)

            qs_fpfh.append(np.array(fpfh_1))
            ks_fpfh.append(np.array(fpfh_2))

        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)

        q_fpfh = np.concatenate(qs_fpfh)
        k_fpfh = np.concatenate(ks_fpfh)

        # normalize to unit vectors
        q = q / torch.norm(q, p=2, dim=1, keepdim=True)
        k = k / torch.norm(k, p=2, dim=1, keepdim=True)

        batch_indices = np.zeros(q.shape[0])
        old_pos = 0
        for i in range(len(qs)):
            batch_indices[old_pos : old_pos + qs[i].shape[0]] = i
            old_pos = qs[i].shape[0]

        if q.shape[0] > max_pos:
            inds = np.random.choice(q.shape[0], max_pos, replace=False)
            q = q[inds]
            k = k[inds]
            q_fpfh = q_fpfh[inds]
            k_fpfh = k_fpfh[inds]
            batch_indices = batch_indices[inds]

        pos = torch.exp(torch.sum(q * k, dim=-1) / tau)
        combined = torch.exp(torch.mm(q, k.t().contiguous()) / tau)

        import itertools

        distances = np.zeros((q.shape[0], q.shape[0]))
        for i, j in itertools.combinations(range(q.shape[0]), 2):
            distances[i, j] = wasserstein_distance(q_fpfh[i], k_fpfh[j])
            distances[j, i] = wasserstein_distance(q_fpfh[j], k_fpfh[i])

        Ng = torch.zeros(q.shape[0], device=q.device)
        for ind in range(q.shape[0]):

            # select the negative values
            neg = combined.index_select(0, torch.tensor([ind], device=q.device))
            # diff_scene_indices = torch.tensor(
            #     np.where(batch_indices != batch_indices[ind])[0], device=q.device
            # )
            select_indices = torch.tensor(
                np.where(distances[ind] > 5.0)[0], device=q.device
            )
            neg = neg.index_select(1, select_indices)
            Ng[ind] = neg.mean(dim=-1) * n

        loss = (-torch.log(pos / (pos + Ng))).mean()

        return loss

    def loss_fn_entropy(self, batch, output):
        tau = 0.4
        max_pos = 4092

        # Get all positive and negative pairs
        qs, ks = [], []
        es = []
        for i, matches in enumerate(batch.correspondences):
            voxel_indices_1 = [v for v, _ in matches]
            voxel_indices_2 = [v for _, v in matches]
            # entropies = [v for _, _, v in matches]

            output_batch_1 = output.features_at(2 * i)
            output_batch_2 = output.features_at(2 * i + 1)
            q = output_batch_1[voxel_indices_1]
            k = output_batch_2[voxel_indices_2]

            # model_input = ME.SparseTensor(batch.features, batch.points)
            # points1 = model_input.coordinates_at(2 * i)
            # points2 = model_input.coordinates_at(2 * i + 1)

            # entropies = np.array(entropies)
            # entropies = np.array(entropies) / np.array(entropies).sum()
            # voxel_indices_1 = np.random.choice(
            #     voxel_indices_1, 500, p=entropies, replace=False
            # )
            # visualize_mapping(points1, points2, voxel_indices_1)

            qs.append(q)
            ks.append(k)
            # es.append(np.array(entropies))

        # TODO: Iterate through each scene and make sure that the negative
        # points don't come from the same scene

        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)
        # es = np.concatenate(es, axis=0)
        # es = es / es.sum()

        if q.shape[0] > max_pos:
            # inds = np.random.choice(q.shape[0], max_pos, p=es, replace=False)
            inds = np.random.choice(q.shape[0], max_pos, replace=False)

            # max_pos_batch = min(max_pos, np.count_nonzero(es))
            # inds = np.random.choice(q.shape[0], max_pos_batch, p=es, replace=False)

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


def visualize_mapping(points1, points2, voxel_indices_1):
    points1 = points1.detach().cpu().numpy()

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)

    colors = np.ones(points1.shape) * np.array([0, 0.4, 0.4])
    colors[voxel_indices_1] = np.array([0, 0, 0])
    pcd1.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd1])


from util.utils import get_random_colour


def visualize_correspondances(quantized_frames, correspondances):
    """Visualize the point correspondances between the matched scans in
    the pretrain input"""

    # for i, matches in enumerate(pretrain_input.correspondances):
    points1 = quantized_frames[0]["discrete_coords"]
    colors1 = quantized_frames[0]["unique_feats"]

    points2 = quantized_frames[1]["discrete_coords"]
    colors2 = quantized_frames[1]["unique_feats"]

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


class MinkowskiTrainer(SegmentationTrainer):
    def __init__(self, cfg: DictConfig, data_interface: DataInterface, backbone=None):
        super(MinkowskiTrainer, self).__init__(cfg, data_interface)

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
