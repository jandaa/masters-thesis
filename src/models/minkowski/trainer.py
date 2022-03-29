import logging
from omegaconf import DictConfig
import functools

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np

# 2D
from torchvision import models

from models.trainers import SegmentationTrainer, BackboneTrainer
from models.minkowski.modules.res16unet import Res16UNet34C
from models.minkowski.modules.common import NormType

from models.minkowski.decoder import FeatureDecoder, MLP2d
from util.types import DataInterface
from models.minkowski.types import (
    MinkowskiInput,
    MinkowskiOutput,
    MinkowskiPretrainInput,
    MinkowskiPretrainInputNew,
    ImagePretrainInput,
)
from util.utils import NCESoftmaxLoss, set_seed
import matplotlib.pyplot as plt

# Visualization
from sklearn.manifold import TSNE

log = logging.getLogger(__name__)


class MinkovskiSemantic(nn.Module):
    def __init__(
        self, cfg: DictConfig, encoding_only=False, backbone=None, freeze_backbone=False
    ):
        nn.Module.__init__(self)

        self.encoding_only = encoding_only
        self.dataset_cfg = cfg.dataset
        self.feature_dim = cfg.model.net.model_n_out
        self.bn_momentum = cfg.model.net.bn_momentum
        self.norm_type = NormType.BATCH_NORM

        # Backbone
        if backbone:
            self.backbone = backbone
        else:
            self.backbone = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)

        # Freeze backbone if required
        if cfg.model.net.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = MinkowskiMLP(
            cfg, self.feature_dim, self.feature_dim, self.dataset_cfg.classes
        )

    def forward(self, input):
        """Extract features and predict semantic class."""

        if self.encoding_only:
            return self.backbone.encoder(input)
        else:
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


# Cross modal expert trainer
class CMEBackboneTrainer(BackboneTrainer):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(CMEBackboneTrainer, self).__init__(cfg)

        # config
        self.feature_dim = cfg.model.net.model_n_out

        # self.model = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)
        self._model = MinkovskiSemantic(cfg, encoding_only=True)

        # 2D feature extraction
        self.image_feature_extractor = models.resnet50(pretrained=True).eval()

    @property
    def model(self):
        return self._model.backbone

    def training_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features.float(), batch.points)
        output = self._model(model_input).F

        # Get 2D output & apply stop gradient
        with torch.no_grad():
            features_2d = self.image_feature_extractor(batch.images)
            features_2d.detach()

        loss = self.loss_fn(output, features_2d)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()

    def validation_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features.float(), batch.points)
        output = self._model(model_input).F

        # Get 2D output & apply stop gradient
        with torch.no_grad():
            features_2d = self.image_feature_extractor(batch.images)
            features_2d.detach()

        loss = self.loss_fn(output, features_2d)
        self.log("val_loss", loss, sync_dist=True)

    def _loss_fn(self, output_online, output_target):
        output_online = nn.functional.normalize(output_online, dim=-1, p=2)
        output_target = nn.functional.normalize(output_target, dim=-1, p=2)
        return 2 - 2 * (output_online * output_target).sum(dim=-1)

    def loss_fn(self, output_online, output_target):
        # For now don't make it symmetric
        return self._loss_fn(output_online, output_target).mean()


def embed_tsne(data):
    """
    N x D np.array data
    """
    tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(data)
    tsne_results = np.squeeze(tsne_results)
    tsne_min = np.min(tsne_results)
    tsne_max = np.max(tsne_results)
    return (tsne_results - tsne_min) / (tsne_max - tsne_min)


# Cross modal expert trainer
class CMEBackboneTrainerFull(BackboneTrainer):
    def __init__(self, cfg: DictConfig, checkpoint_2d_path=None):
        super(CMEBackboneTrainerFull, self).__init__(cfg)

        # config
        self.feature_dim = cfg.model.net.model_n_out

        # 3D feature extractor
        self.model = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)

        self.head = MinkowskiMLP(
            cfg, self.feature_dim, self.feature_dim, self.dataset_cfg.classes
        )

        self.criterion = nn.CrossEntropyLoss()

        # 2D feature extraction
        if checkpoint_2d_path:
            image_trainer = ImageTrainer.load_from_checkpoint(
                cfg=cfg, checkpoint_path=checkpoint_2d_path
            )
        else:
            image_trainer = ImageTrainer(cfg)
        self.image_feature_extractor = image_trainer.model

    def training_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        input_3d_1 = ME.SparseTensor(batch.features1, batch.points1)
        # input_3d_2 = ME.SparseTensor(batch.features2, batch.points2)
        output_3d_1 = self.model(input_3d_1)
        # output_3d_2 = self.model(input_3d_2)

        projection_1 = self.head(output_3d_1)
        # projection_2 = self.head(output_3d_2)

        # Get 2D output & apply stop gradient
        with torch.no_grad():
            features_2d = self.image_feature_extractor(batch.images)
            features_2d.detach()

        loss = self.loss_fn_2d_3d(
            projection_1, features_2d, batch.point_to_pixel_maps1, batch
        )

        # loss = 0.5 * self.loss_fn_2d_3d(
        #     projection_1, features_2d, batch.point_to_pixel_maps1, batch
        # )
        # loss += 0.5 * self.loss_fn_2d_3d(
        #     projection_2, features_2d, batch.point_to_pixel_maps2, batch
        # )
        # loss /= 2  # Average loss out

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()

    def validation_step(self, batch: MinkowskiPretrainInputNew, batch_idx: int):
        input_3d_1 = ME.SparseTensor(batch.features1, batch.points1)
        # input_3d_2 = ME.SparseTensor(batch.features2, batch.points2)
        output_3d_1 = self.model(input_3d_1)
        # output_3d_2 = self.model(input_3d_2)

        projection_1 = self.head(output_3d_1)
        # projection_2 = self.head(output_3d_2)

        # Get 2D output & apply stop gradient
        with torch.no_grad():
            features_2d = self.image_feature_extractor(batch.images)
            features_2d.detach()

        loss = self.loss_fn_2d_3d(
            projection_1, features_2d, batch.point_to_pixel_maps1, batch
        )

        # loss = 0.5 * self.loss_fn_2d_3d(
        #     projection_1, features_2d, batch.point_to_pixel_maps1, batch
        # )
        # loss += 0.5 * self.loss_fn_2d_3d(
        #     projection_2, features_2d, batch.point_to_pixel_maps2, batch
        # )
        # loss /= 2  # Average loss out

        # loss = self.loss_fn(output, features_2d, batch)
        self.log("val_loss", loss, sync_dist=True)

    def find_closest_feature(self, coords, features, pixels):

        interpolated_features = torch.zeros((pixels.shape[0], features.shape[0]))

        x_0 = int(coords[0, 0, 0])
        y_0 = int(coords[1, 0, 0])
        dx = coords[0, 0, 1] - coords[0, 0, 0]
        dy = coords[1, 1, 0] - coords[1, 0, 0]

        for feature_ind, (p_x, p_y) in enumerate(pixels):

            # Get indices in the transformed image
            i = int((p_x - x_0) / dx)
            j = int((p_y - y_0) / dy)

            interpolated_features[feature_ind] = features[:, j, i]

        return interpolated_features.to(features.device)

    def find_closest_feature_verify(self, coords, features, pixels):

        # interpolated_features = torch.zeros((pixels.shape[0], features.shape[0]))
        interpolated_features = torch.zeros((418, 418, features.shape[0]))

        x_0 = int(coords[0, 0, 0])
        y_0 = int(coords[1, 0, 0])
        x_n = int(coords[0, 0, -1])
        y_n = int(coords[1, -1, 0])
        dx = coords[0, 0, 1] - coords[0, 0, 0]
        dy = coords[1, 1, 0] - coords[1, 0, 0]

        for p_x in range(x_0, x_n):
            for p_y in range(y_0, y_n):

                # Get indices in the transformed image
                i = int((p_x - x_0) / dx)
                j = int((p_y - y_0) / dy)

                interpolated_features[p_y - y_0, p_x - x_0] = features[:, j, i]

        return interpolated_features.to(features.device)

    def bilinear_interpret(self, coords, features, pixels):

        interpolated_features = torch.zeros((pixels.shape[0], features.shape[0]))

        x_0 = int(coords[0, 0, 0])
        y_0 = int(coords[1, 0, 0])
        dx = coords[0, 0, 1] - coords[0, 0, 0]
        dy = coords[1, 1, 0] - coords[1, 0, 0]

        for feature_ind, (p_x, p_y) in enumerate(pixels):

            # Get indices in the transformed image
            i = int((p_x - x_0) / dx)
            j = int((p_y - y_0) / dy)

            dx1 = p_x - coords[0, 0, i]
            dx2 = coords[0, 0, i + 1] - p_x
            dy1 = p_y - coords[1, j, 0]
            dy2 = coords[1, j + 1, 0] - p_y

            # perform bi-linear interpolation
            interpolated_features[feature_ind] = (
                dx2 * dy2 * features[:, j, i]
                + dx1 * dy2 * features[:, j, i + 1]
                + dx2 * dy1 * features[:, j + 1, i]
                + dx1 * dy1 * features[:, j + 1, i + 1]
            )

        interpolated_features = interpolated_features / (dx * dy)
        return interpolated_features.to(features.device)

    def bilinear_interpret_verify(self, coords, features, pixels):

        interpolated_features = torch.zeros((pixels.shape[0], features.shape[0]))
        interpolated_features = torch.zeros((418, 418, features.shape[0]))

        x_0 = int(coords[0, 0, 0])
        y_0 = int(coords[1, 0, 0])
        x_n = int(coords[0, 0, -1])
        y_n = int(coords[1, -1, 0])
        dx = coords[0, 0, 1] - coords[0, 0, 0]
        dy = coords[1, 1, 0] - coords[1, 0, 0]

        for p_x in range(x_0, x_n):
            for p_y in range(y_0, y_n):

                # Get indices in the transformed image
                i = int((p_x - x_0) / dx)
                j = int((p_y - y_0) / dy)

                dx1 = p_x - coords[0, 0, i]
                dx2 = coords[0, 0, i + 1] - p_x
                dy1 = p_y - coords[1, j, 0]
                dy2 = coords[1, j + 1, 0] - p_y

                # perform bi-linear interpolation
                interpolated_features[p_y - y_0, p_x - x_0] = (
                    dx2 * dy2 * features[:, j, i]
                    + dx1 * dy2 * features[:, j, i + 1]
                    + dx2 * dy1 * features[:, j + 1, i]
                    + dx1 * dy1 * features[:, j + 1, i + 1]
                )

        interpolated_features = interpolated_features / (dx * dy)
        return interpolated_features.to(features.device)

    def _loss_fn(self, output_online, output_target):
        q = nn.functional.normalize(output_online, dim=-1, p=2)
        k = nn.functional.normalize(output_target, dim=-1, p=2)

        tau = 0.4

        # Labels
        npos = q.shape[0]
        labels = torch.arange(npos).to(q.device).long()

        logits = torch.mm(q, k.transpose(1, 0))  # npos by npos
        out = torch.div(logits, tau)
        out = out.squeeze().contiguous()

        return self.criterion(out, labels)

    def loss_fn_2d_3d(self, output_online, output_target, point_to_pixel_map, batch):
        max_pos = 4092

        # Find corresponding 2D feature vectors to 3D points
        online_features = []
        target_features = []
        for i in range(batch.batch_size):

            # # Visualize feature interpolation
            # features = self.bilinear_interpret_verify(
            #     batch.image_coordinates[i],
            #     output_target[i],
            #     point_to_pixel_map[i][:, 1:3],
            # )

            # # features = output_target[i]
            # # features = features[:, 150:300, 150:300]
            # # features = features.reshape(16, -1).T.detach().cpu().numpy()
            # features = features.reshape(-1, 16).detach().cpu().numpy()
            # embedding = embed_tsne(features)
            # embedding = embedding.reshape(418, 418)

            # # Plot TSNE results
            # plt.imshow(embedding, cmap="autumn", interpolation="nearest")
            # plt.title("2-D Heat Map")
            # plt.savefig("vis.png")

            # Get 2D features
            target_features_i = self.bilinear_interpret(
                batch.image_coordinates[i],
                output_target[i],
                point_to_pixel_map[i][:, 1:3],
            )
            target_features.append(target_features_i)

            # Get 3D features
            online_features_i = output_online.features_at(i)
            online_features_i = online_features_i[point_to_pixel_map[i][:, 0]]
            online_features.append(online_features_i)

        online_features = torch.vstack(online_features)
        target_features = torch.vstack(target_features)

        # # limit max number of query points
        # if online_features.shape[0] > max_pos:
        #     inds = np.random.choice(online_features.shape[0], max_pos, replace=False)
        #     online_features = online_features[inds]
        #     target_features = target_features[inds]

        # For now don't make it symmetric
        return self._loss_fn(online_features, target_features)

    def loss_fn_3d(self, output1, output2, batch):
        max_pos = 4092

        # Get all positive and negative pairs
        qs, ks = [], []
        for i, matches in enumerate(batch.point_to_point_map):
            voxel_indices_1 = matches[:, 0]
            voxel_indices_2 = matches[:, 1]

            output_batch_1 = output1.features_at(i)
            output_batch_2 = output2.features_at(i)
            q = output_batch_1[voxel_indices_1]
            k = output_batch_2[voxel_indices_2]

            qs.append(q)
            ks.append(k)

        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)

        # limit max number of query points
        if q.shape[0] > max_pos:
            inds = np.random.choice(q.shape[0], max_pos, replace=False)
            q = q[inds]
            k = k[inds]

        return self._loss_fn(q, k)


class ImageTrainer(BackboneTrainer):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(ImageTrainer, self).__init__(cfg)

        # config
        self.feature_dim = cfg.model.net.model_n_out
        self.learning_rate = cfg.model.pretrain.optimizer.lr
        self.warmup_steps = cfg.model.net.warmup_steps

        # 2D feature extraction
        self.model = FeatureDecoder()

        # 3D feature extraction
        # self.model_3d = MinkovskiSemantic(cfg)

        # Projection head
        # self.projection_head = MLP2d(16, inner_dim=64, out_dim=16)

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # Unfreeze encoder after warmup period
        if self.warmup_steps:
            if self.trainer.global_step == self.warmup_steps:
                log.info("Unfreezing encoder!")
                self.model.unfreeze_encoder()
        else:
            log.info("Unfreezing encoder!")
            self.model.unfreeze_encoder()

        # update params
        optimizer.step(closure=optimizer_closure)

    def find_closest_feature(self, coords, features, pixels):

        interpolated_features = torch.zeros((pixels.shape[0], features.shape[0]))

        x_0 = int(coords[0, 0, 0])
        y_0 = int(coords[1, 0, 0])
        dx = coords[0, 0, 1] - coords[0, 0, 0]
        dy = coords[1, 1, 0] - coords[1, 0, 0]

        for feature_ind, (p_x, p_y) in enumerate(pixels):

            # Get indices in the transformed image
            i = int((p_x - x_0) / dx)
            j = int((p_y - y_0) / dy)

            interpolated_features[feature_ind] = features[:, i, j]

        return interpolated_features.to(features.device)

    def training_step(self, batch: ImagePretrainInput, batch_idx: int):

        # Get Encoder values
        z1 = self.model(batch.images1)
        z2 = self.model(batch.images2)

        # 2D loss
        loss = self.loss_fn(z1, z2, batch.correspondences)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def validation_step(self, batch: ImagePretrainInput, batch_idx: int):

        # Get Encoder values
        z1 = self.model(batch.images1)
        z2 = self.model(batch.images2)

        loss = self.loss_fn(z1, z2, batch.correspondences)

        self.log("val_loss", loss, sync_dist=True)

    def _loss_fn(self, output_online, output_target):
        q = nn.functional.normalize(output_online, dim=-1, p=2)
        k = nn.functional.normalize(output_target, dim=-1, p=2)

        tau = 0.4

        # Labels
        npos = q.shape[0]
        labels = torch.arange(npos).to(q.device).long()

        logits = torch.mm(q, k.transpose(1, 0))  # npos by npos
        out = torch.div(logits, tau)
        out = out.squeeze().contiguous()

        return self.criterion(out, labels)

    def loss_fn(self, p, z, correspondances):

        max_pos = 4092
        N, C, H, W = p.shape

        # [bs, feat_dim, 224x224]
        p = p.view(N, C, -1)
        z = z.view(N, C, -1)

        loss = 0
        for batch_ind, matches in enumerate(correspondances):
            if len(matches) == 0:
                continue

            # limit max size per image
            # Randomly select matches
            if matches.shape[0] > max_pos:
                inds = np.random.choice(matches.shape[0], max_pos, replace=False)
                matches = matches[inds, :]

            online = p[batch_ind, :, matches[:, 0]]
            target = z[batch_ind, :, matches[:, 1]]

            loss += self._loss_fn(online.T, target.T).mean()

        return loss / len(correspondances)


class ImagePointTrainer(BackboneTrainer):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(ImageTrainer, self).__init__(cfg)

        # config
        self.feature_dim = cfg.model.net.model_n_out
        self.learning_rate = cfg.model.pretrain.optimizer.lr
        self.warmup_steps = cfg.model.net.warmup_steps

        # 2D feature extraction
        self.model_2d = FeatureDecoder()

        # 3D feature extraction
        self.model_3d = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)

        # Projection head
        self.projection_head = MLP2d(16, inner_dim=64, out_dim=16)

    def training_step(self, batch: ImagePretrainInput, batch_idx: int):

        # Get Encoder values
        z1 = self.model(batch.images1)
        z2 = self.model(batch.images2)

        # Stop grad on target
        z1.detach()
        z2.detach()

        # Get projection values
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)

        # Compute a symmetric loss
        loss = 0.5 * (
            self.loss_fn(p1, z2, batch.correspondences)
            + self.loss_fn(p2, z1, batch.correspondences, backwards=True)
        )

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def validation_step(self, batch: ImagePretrainInput, batch_idx: int):

        # Get Encoder values
        z1 = self.model(batch.images1)
        z2 = self.model(batch.images2)

        # Stop grad on target
        z1.detach()
        z2.detach()

        # Get projection values
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)

        loss = 0.5 * (
            self.loss_fn(p1, z2, batch.correspondences)
            + self.loss_fn(p2, z1, batch.correspondences, backwards=True)
        )

        self.log("val_loss", loss, sync_dist=True)

    def _loss_fn(self, output_online, output_target):
        output_online = nn.functional.normalize(output_online, dim=-1, p=2)
        output_target = nn.functional.normalize(output_target, dim=-1, p=2)
        return 2 - 2 * (output_online * output_target).sum(dim=-1)
        # return -2.0 * torch.einsum("nc, nc->n", [output_online, output_target]).mean()

    def loss_fn(self, p, z, correspondances, backwards=False):

        max_pos = 2000
        N, C, H, W = p.shape

        # [bs, feat_dim, 224x224]
        p = p.view(N, C, -1)
        z = z.view(N, C, -1)

        loss = 0
        for batch_ind, matches in enumerate(correspondances):
            if len(matches) == 0:
                continue

            # limit max size per image
            # Randomly select matches
            if matches.shape[0] > max_pos:
                inds = np.random.choice(matches.shape[0], max_pos, replace=False)
                matches = matches[inds, :]

            if not backwards:
                online = p[batch_ind, :, matches[:, 0]]
                target = z[batch_ind, :, matches[:, 1]]
            else:
                online = p[batch_ind, :, matches[:, 1]]
                target = z[batch_ind, :, matches[:, 0]]

            loss += self._loss_fn(online.T, target.T).mean()

        return loss / len(correspondances) * 10


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

        self.model = Res16UNet34C(3, cfg.dataset.classes, cfg.model, D=3)
        if backbone:
            state_dict = backbone.state_dict()
            head_params = [key for key in state_dict if "final" in key]
            for key in head_params:
                del state_dict[key]
            self.model.load_state_dict(state_dict, strict=False)

        self.semantic_criterion = nn.CrossEntropyLoss(
            ignore_index=cfg.dataset.ignore_label
        )

    @property
    def return_instances(self):
        return False

    def on_before_zero_grad(self, optimizer) -> None:
        set_seed(self.trainer.global_step)

    def training_step(self, batch: MinkowskiInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features, batch.points)
        output = self.model(model_input)
        output = MinkowskiOutput(output=output, semantic_scores=output.F)
        loss = self.loss_fn(batch, output)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def validation_step(self, batch: MinkowskiInput, batch_idx: int):
        model_input = ME.SparseTensor(batch.features, batch.points)
        output = self.model(model_input)
        output = MinkowskiOutput(output=output, semantic_scores=output.F)
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
