import logging
from omegaconf import DictConfig
import functools
import itertools
import pickle

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
from scipy.stats import wasserstein_distance

from models.trainers import SegmentationTrainer, BackboneTrainer
from models.minkowski.modules.res16unet import Res16UNet34C
from models.minkowski.modules.resnet import get_norm
from models.minkowski.modules.common import NormType

from util.utils import NCELossMoco
from util.types import DataInterface
from models.minkowski.types import (
    MinkowskiInput,
    MinkowskiOutput,
    MinkowskiPretrainInput,
)
from util.utils import NCESoftmaxLoss

log = logging.getLogger(__name__)


class MinkovskiSemantic(nn.Module):
    def __init__(self, cfg: DictConfig, backbone=None, freeze_backbone=False):
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

        if cfg.model.net.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = MinkowskiMLP(
            cfg, self.feature_dim, self.feature_dim, self.dataset_cfg.classes
        )

    def forward(self, input):
        """Extract features and predict semantic class."""

        # Get backbone features
        output = self.backbone(input)
        output = self.head(output)

        return MinkowskiOutput(output=output, semantic_scores=output.F)


class MinkowskiMLP(nn.Module):
    def __init__(self, cfg: DictConfig, input_size, hidden_size, output_size):
        nn.Module.__init__(self)

        self.dataset_cfg = cfg.dataset
        self.input_size = input_size
        self.hidden_dim = hidden_size
        self.output_size = output_size
        self.bn_momentum = cfg.model.net.bn_momentum
        self.norm_type = NormType.BATCH_NORM

        self.linear1 = ME.MinkowskiLinear(input_size, hidden_size, bias=True)
        self.bn1 = ME.MinkowskiBatchNorm(
            hidden_size, eps=1e-5, momentum=self.bn_momentum
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.linear2 = ME.MinkowskiLinear(hidden_size, output_size, bias=True)

    def forward(self, input):
        output = self.linear1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.linear2(output)

        return output


class MinkowskiBYOL(nn.Module):
    def __init__(self, cfg: DictConfig, use_predictor=False):
        nn.Module.__init__(self)

        self.dataset_cfg = cfg.dataset
        self.use_predictor = use_predictor
        self.feature_dim = cfg.model.net.model_n_out
        self.bn_momentum = cfg.model.net.bn_momentum
        self.norm_type = NormType.BATCH_NORM

        # Backbone
        self.backbone = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)

        # Projection head
        self.projector = MinkowskiMLP(
            cfg,
            self.feature_dim,
            cfg.model.pretrain.loss.projector_hidden_size,
            cfg.model.pretrain.loss.projector_output_size,
        )

        if use_predictor:
            self.predictor = MinkowskiMLP(
                cfg,
                cfg.model.pretrain.loss.projector_output_size,
                cfg.model.pretrain.loss.predictor_hidden_size,
                cfg.model.pretrain.loss.projector_output_size,
            )

    def forward(self, input):
        """Extract features and predict semantic class."""

        # Get embedding
        output = self.backbone(input)

        # Get projected features
        output = self.projector(output)

        # Get predicted features if requested
        if self.use_predictor:
            output = self.predictor(output)

        return output


class MinkowskiBOYLBackboneTrainer(BackboneTrainer):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(MinkowskiBOYLBackboneTrainer, self).__init__(cfg)

        # config
        self.feature_dim = cfg.model.net.model_n_out
        self.tau = cfg.model.pretrain.loss.byol_tau

        # Loss type
        self.criterion = nn.MSELoss()

        # Encoders
        self.model_online = MinkowskiBYOL(cfg, use_predictor=True)
        self.model_target = MinkowskiBYOL(cfg, use_predictor=False)

    @property
    def model(self):
        return self.model_online.backbone

    def training_step(self, batch: MinkowskiPretrainInput, batch_idx: int):

        # Get outputs
        model_input = ME.SparseTensor(batch.features.float(), batch.points)
        output_online = self.model_online(model_input)
        output_target = self.model_target(model_input)

        # Compute loss function
        loss = self.loss_fn(batch, output_online, output_target)

        # update target parameters
        self._update_target_parameters()

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    @torch.no_grad()
    def _update_target_parameters(self):
        for param_online, param_target in zip(
            self.model_online.parameters(), self.model_target.parameters()
        ):
            param_target.data = (
                self.tau * param_target.data + (1.0 - self.tau) * param_online.data
            )

    def validation_step(self, batch: MinkowskiPretrainInput, batch_idx: int):

        # Get outputs
        model_input = ME.SparseTensor(batch.features, batch.points)
        output_online = self.model_online(model_input)
        output_target = self.model_target(model_input)

        # Compute loss function
        loss = self.loss_fn(batch, output_online, output_target)
        self.log("val_loss", loss, sync_dist=True)

    def loss_fn(self, batch, output_online, output_target):
        max_pos = 4092

        # Get all samples
        online_view_1 = []
        online_view_2 = []
        target_view_1 = []
        target_view_2 = []
        for i, matches in enumerate(batch.correspondences):
            voxel_indices_1 = [match["frame1"]["voxel_inds"] for match in matches]
            voxel_indices_2 = [match["frame2"]["voxel_inds"] for match in matches]

            output_online_1 = output_online.features_at(2 * i)
            output_online_2 = output_online.features_at(2 * i + 1)
            online_view_1.append(output_online_1[voxel_indices_1])
            online_view_2.append(output_online_2[voxel_indices_2])

            output_target_1 = output_target.features_at(2 * i)
            output_target_2 = output_target.features_at(2 * i + 1)
            target_view_1.append(output_target_1[voxel_indices_1])
            target_view_2.append(output_target_2[voxel_indices_2])

        # Create tensors
        online_view_1 = torch.cat(online_view_1, 0)
        online_view_2 = torch.cat(online_view_2, 0)
        target_view_1 = torch.cat(target_view_1, 0)
        target_view_2 = torch.cat(target_view_2, 0)

        # limit max number of query points
        if online_view_1.shape[0] > max_pos:
            inds = np.random.choice(online_view_1.shape[0], max_pos, replace=False)
            online_view_1 = online_view_1[inds]
            online_view_2 = online_view_2[inds]
            target_view_1 = target_view_1[inds]
            target_view_2 = target_view_2[inds]

        # normalize to unit vectors
        online_view_1 = nn.functional.normalize(online_view_1, dim=1, p=2)
        online_view_2 = nn.functional.normalize(online_view_2, dim=1, p=2)
        target_view_1 = nn.functional.normalize(target_view_1, dim=1, p=2)
        target_view_2 = nn.functional.normalize(target_view_2, dim=1, p=2)

        # Apply stop gradient to target network outputs
        target_view_1 = target_view_1.detach()
        target_view_2 = target_view_2.detach()

        # Compute regression loss (symmetrically)
        loss = self.criterion(online_view_1, target_view_2)
        loss += self.criterion(online_view_2, target_view_1)

        return loss * 100


class MinkowskiMocoBackboneTrainer(BackboneTrainer):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(MinkowskiMocoBackboneTrainer, self).__init__(cfg)

        # config
        self.feature_dim = cfg.model.net.model_n_out
        self.difficulty = cfg.model.pretrain.loss.difficulty
        self.m = cfg.model.pretrain.loss.momentum
        self.K = (
            cfg.model.pretrain.loss.num_neg_points
            * cfg.model.pretrain.loss.queue_multiple
        )

        # queue
        self.register_buffer("queue", torch.randn(self.feature_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()

        # Encoders
        self.model = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)
        self.model2 = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)

        # initalize encoder 2 with parameters of encoder 1
        for param_q, param_k in zip(self.model.parameters(), self.model2.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def training_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features.float(), batch.points)
        output1 = self.model(model_input)

        # no gradient to keys
        with torch.no_grad():

            # update the key encoder
            self._momentum_update_key_encoder()

            # get output of second encoder
            output2 = self.model2(model_input)

        loss = self.loss_fn(batch, output1, output2)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def validation_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features, batch.points)
        output1 = self.model(model_input)
        output2 = self.model2(model_input)
        loss = self.loss_fn(batch, output1, output2, is_val=True)
        self.log("val_loss", loss, sync_dist=True)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.model2.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = torch.transpose(keys, 0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def loss_fn(self, batch, output1, output2, is_val=False):
        max_pos = 4092
        tau = 0.4

        # Get all positive and negative pairs
        qs, ks = [], []
        for i, matches in enumerate(batch.correspondences):
            voxel_indices_1 = [match["frame1"]["voxel_inds"] for match in matches]
            voxel_indices_2 = [match["frame2"]["voxel_inds"] for match in matches]

            output_batch_1 = output1.features_at(2 * i)
            output_batch_2 = output2.features_at(2 * i + 1)
            q = output_batch_1[voxel_indices_1]
            k = output_batch_2[voxel_indices_2]

            qs.append(q)
            ks.append(k)

        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)

        # normalize to unit vectors
        q = nn.functional.normalize(q, dim=1, p=2)
        k = nn.functional.normalize(k, dim=1, p=2)
        k = k.detach()

        # limit max number of query points
        if q.shape[0] > max_pos:
            inds = np.random.choice(q.shape[0], max_pos, replace=False)
            q = q[inds]
            k = k[inds]

        # return self.criterion(logits, labels)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # negative logits: NxK
        neg_features = self.queue.clone().detach()
        l_neg = torch.einsum("nc,ck->nk", [q, neg_features])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= tau

        if not is_val:
            self._dequeue_and_enqueue(k)

        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.int64)

        return self.criterion(torch.squeeze(logits), labels)


class MinkowskiBackboneTrainer(BackboneTrainer):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(MinkowskiBackboneTrainer, self).__init__(cfg)

        # config
        self.feature_dim = cfg.model.net.model_n_out

        # self.model = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)
        self._model = MinkovskiSemantic(cfg)

    @property
    def model(self):
        return self._model.backbone

    def training_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features.float(), batch.points)
        output = self._model(model_input).output
        loss = self.loss_fn(batch, output)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()

    def validation_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features, batch.points)
        output = self._model(model_input).output
        loss = self.loss_fn(batch, output)
        self.log("val_loss", loss, sync_dist=True)

    def loss_fn_new(self, batch, output):
        tau = 0.4
        max_pos = 3072
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

        loss = (-torch.log(pos / (Ng))).mean()

        return loss

    def loss_fn(self, batch, output):
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
