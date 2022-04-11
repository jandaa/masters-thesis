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
        input_3d = ME.SparseTensor(batch.features1, batch.points1)
        features_3d = self.model(input_3d)
        projection = self.head(features_3d)

        # Get 2D output & apply stop gradient
        with torch.no_grad():
            features_2d = self.image_feature_extractor(batch.images)
            features_2d.detach()

        loss = self.loss_fn_2d_3d(
            projection, features_2d, batch.point_to_pixel_maps1, batch
        )

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()

    def validation_step(self, batch: MinkowskiPretrainInputNew, batch_idx: int):
        input_3d = ME.SparseTensor(batch.features1, batch.points1)
        features_3d = self.model(input_3d)
        projection = self.head(features_3d)

        # Get 2D output & apply stop gradient
        with torch.no_grad():
            features_2d = self.image_feature_extractor(batch.images)
            features_2d.detach()

        loss = self.loss_fn_2d_3d(
            projection, features_2d, batch.point_to_pixel_maps1, batch
        )

        # loss = self.loss_fn(output, features_2d, batch)
        self.log("val_loss", loss, sync_dist=True)

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
        tau = 0.4

        q = nn.functional.normalize(output_online, dim=-1, p=2)
        k = nn.functional.normalize(output_target, dim=-1, p=2)

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


class ImageSegmentationTrainer(BackboneTrainer):
    def __init__(self, cfg: DictConfig, checkpoint_2d_path=None):
        super(ImageSegmentationTrainer, self).__init__(cfg)

        # 2D feature extraction
        if checkpoint_2d_path:
            self.model = ImageTrainer.load_from_checkpoint(
                cfg=cfg, checkpoint_path=checkpoint_2d_path
            ).model
        else:
            self.model = FeatureDecoder()

        self.head = MLP2d(16, inner_dim=16, out_dim=cfg.dataset.classes + 1)

        self.semantic_criterion = nn.CrossEntropyLoss(
            ignore_index=cfg.dataset.ignore_label
        )

    def training_step(self, batch: ImagePretrainInput, batch_idx: int):

        # Get Encoder values
        z1 = self.model(batch.images1)
        output = self.head(z1)

        semantic_predictions = output.reshape((-1, output.shape[1]))
        semantic_labels = batch.labels.reshape((-1,)).long()
        loss = self.semantic_criterion(semantic_predictions, semantic_labels)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def validation_step(self, batch: ImagePretrainInput, batch_idx: int):

        # Get Encoder values
        z1 = self.model(batch.images1)
        output = self.head(z1)

        semantic_predictions = output.reshape((-1, output.shape[1]))
        semantic_labels = batch.labels.reshape((-1,)).long()
        loss = self.semantic_criterion(semantic_predictions, semantic_labels)

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


class MinkowskiBackboneTrainer(BackboneTrainer):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(MinkowskiBackboneTrainer, self).__init__(cfg)

        # config
        self.feature_dim = cfg.model.net.model_n_out

        self.model = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)

    @property
    def model(self):
        return self._model.backbone

    def training_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        model_input1 = ME.SparseTensor(batch.features1.float(), batch.points1)
        output1 = self._model(model_input1).output
        loss = self.loss_fn(batch, output1)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()

    def validation_step(self, batch: MinkowskiPretrainInput, batch_idx: int):
        model_input1 = ME.SparseTensor(batch.features1.float(), batch.points1)
        output1 = self._model(model_input1).output
        loss = self.loss_fn(batch, output1)
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
        for i in range(batch.batch_size):
            # voxel_indices_1 = [match["frame1"]["voxel_inds"] for match in matches]
            # voxel_indices_2 = [match["frame2"]["voxel_inds"] for match in matches]

            q = output.features_at(2 * i)[batch.point_to_point_map[i][:, 0]]
            k = output.features_at(2 * i + 1)[batch.point_to_point_map[i][:, 1]]

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
