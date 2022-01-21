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

        # Projection head
        self.linear = ME.MinkowskiLinear(
            self.feature_dim, self.dataset_cfg.classes, bias=True
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
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, input):
        """Extract features and predict semantic class."""

        # Get backbone features
        output = self.backbone(input)

        # # Run features through 2-layer non-linear projection head
        output = self.linear(output)
        # output = self.bn1(output)
        # output = self.relu(output)
        # output = self.linear2(output)
        # output = self.bn2(output)
        output = self.relu(output)

        return MinkowskiOutput(output=output, semantic_scores=output.F)


class MinkowskiBackboneTrainer(BackboneTrainer):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(MinkowskiBackboneTrainer, self).__init__(cfg)

        # config
        self.feature_dim = cfg.model.net.model_n_out
        self.difficulty = cfg.model.pretrain.loss.difficulty

        self.model = Res16UNet34C(3, self.feature_dim, cfg.model, D=3)
        # self.model = MinkovskiSemantic(cfg)

        # which training loss to use
        if cfg.model.net.loss == "new":
            self.loss_fn = self.loss_fn_new
        elif cfg.model.net.loss == "entropy":
            self.loss_fn = self.loss_fn_entropy
        elif cfg.model.net.loss == "delta":
            self.loss_fn = self.loss_fn_delta
        elif cfg.model.net.loss == "cluster":
            self.loss_fn = self.loss_fn_cluster
        elif cfg.model.net.loss == "debiased":
            self.loss_fn = self.loss_fn_debiased
        elif cfg.model.net.loss == "hard":  # TODO: rename this
            self.loss_fn = self.loss_fn_hard
        elif cfg.model.net.loss == "select_difficulty":
            self.loss_fn = self.loss_fn_select_difficulty
        elif cfg.model.net.loss == "mixing":
            self.criterion = NCELossMoco(cfg.model)
            self.loss_fn = self.loss_fn_mixing
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
        max_pos = 200
        n = 4092

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

        if q.shape[0] > max_pos:
            inds = np.random.choice(q.shape[0], max_pos, replace=False)
            q = q[inds]
            k = k[inds]
            q_fpfh = q_fpfh[inds]
            k_fpfh = k_fpfh[inds]

        pos = torch.exp(torch.sum(q * k, dim=-1) / tau)
        combined = torch.exp(torch.mm(q, k.t().contiguous()) / tau)

        distances = np.zeros((q.shape[0], q.shape[0]))
        for i, j in itertools.combinations(range(q.shape[0]), 2):
            if i == j:
                distances[i, j] = 0.0
                distances[j, i] = 0.0
            else:
                distances[i, j] = wasserstein_distance(q_fpfh[i], k_fpfh[j])
                distances[j, i] = wasserstein_distance(q_fpfh[j], k_fpfh[i])

        Ng = torch.zeros(q.shape[0], device=q.device)
        for ind in range(q.shape[0]):

            # select row corresponding to query point
            neg = combined.index_select(0, torch.tensor([ind], device=q.device))

            # select the negative values
            select_indices = torch.tensor(
                np.where(distances[ind] > 7.0)[0], device=q.device
            )
            neg = neg.index_select(1, select_indices)

            # Compute the mean and normalize by n
            Ng[ind] = neg.mean(dim=-1) * n

        loss = (-torch.log(pos / (pos + Ng))).mean()

        return loss

    def loss_fn_cluster(self, batch, output):
        tau = 0.4
        max_pos = int(4092 / 1)
        n = 4092

        # Get all positive and negative pairs
        qs, ks = [], []
        qs_clusters = []
        for i, matches in enumerate(batch.correspondences):
            voxel_indices_1 = [match["frame1"]["voxel_inds"] for match in matches]
            voxel_indices_2 = [match["frame2"]["voxel_inds"] for match in matches]

            clusters = [match["frame1"]["clusters"] for match in matches]

            output_batch_1 = output.features_at(2 * i)
            output_batch_2 = output.features_at(2 * i + 1)
            q = output_batch_1[voxel_indices_1]
            k = output_batch_2[voxel_indices_2]

            qs.append(q)
            ks.append(k)

            qs_clusters.append(np.array(clusters))

        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)

        q_clusters = np.concatenate(qs_clusters)

        # normalize to unit vectors
        q = q / torch.norm(q, p=2, dim=1, keepdim=True)
        k = k / torch.norm(k, p=2, dim=1, keepdim=True)

        if q.shape[0] > max_pos:
            inds = np.random.choice(q.shape[0], max_pos, replace=False)
            q = q[inds]
            k = k[inds]
            q_clusters = q_clusters[inds]

        # cluster_pred = self.clusters.predict(q_fpfh)

        pos = torch.exp(torch.sum(q * k, dim=-1) / tau)
        combined = torch.exp(torch.mm(q, k.t().contiguous()) / tau)

        Ng = torch.zeros(q.shape[0], device=q.device)
        for ind in range(q.shape[0]):

            # select row corresponding to query point
            neg = combined.index_select(0, torch.tensor([ind], device=q.device))

            # get cluster
            cluster_ind = q_clusters[ind]

            # select the negative values
            select_indices = torch.tensor(
                np.where(q_clusters != cluster_ind)[0], device=q.device
            )
            neg = neg.index_select(1, select_indices)

            # Compute the mean and normalize by n
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
            voxel_indices_1 = [match["frame1"]["voxel_inds"] for match in matches]
            voxel_indices_2 = [match["frame2"]["voxel_inds"] for match in matches]
            entropies = [match["frame1"]["entropies"] for match in matches]

            output_batch_1 = output.features_at(2 * i)
            output_batch_2 = output.features_at(2 * i + 1)
            q = output_batch_1[voxel_indices_1]
            k = output_batch_2[voxel_indices_2]

            # Visualize sampling
            entropies = np.array(entropies)
            inds = np.where(entropies < 0.4)[0]
            entropies[inds] = 0

            # VISUALIZATION
            # entropies = entropies / entropies.sum()
            # voxel_indices_1 = np.random.choice(
            #     voxel_indices_1, 1000, p=entropies, replace=False
            # )
            # model_input = ME.SparseTensor(batch.features, batch.points)
            # points1 = model_input.coordinates_at(2 * i)
            # features1 = model_input.features_at(2 * i)
            # visualize_mapping(points1, features1, voxel_indices_1, entropies)

            qs.append(q)
            ks.append(k)
            es.append(entropies)

        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)
        es = np.concatenate(es, axis=0)
        es = es / es.sum()

        if q.shape[0] > max_pos:
            max_pos_batch = min(max_pos, np.count_nonzero(es))
            inds = np.random.choice(q.shape[0], max_pos_batch, p=es, replace=False)

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

    def loss_fn_debiased(self, batch, output):
        tau = 0.4
        max_pos = 4092
        n = 4092
        tau_plus = 0.1  # class probability

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

            # Select all but the same index
            select_indices = torch.tensor(
                [i for i in range(neg.shape[1]) if i != ind], device=q.device
            )
            neg = neg.index_select(1, select_indices)

            Ng[ind] = neg.mean(dim=-1) * n

        # Debiased objective
        Ng = torch.max(
            (-n * tau_plus * pos + Ng).sum() / (1 - tau_plus),
            torch.exp(-1 / torch.tensor(tau, device=q.device)),
        )
        loss = (-torch.log(pos / (pos + Ng))).mean()

        return loss

    def loss_fn_hard(self, batch, output):
        tau = 0.4
        max_pos = 4092
        n = 4092
        tau_plus = 0.1  # class probability
        beta = 5.0

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

            # Select all but the same index
            select_indices = torch.tensor(
                [i for i in range(neg.shape[1]) if i != ind], device=q.device
            )
            neg = neg.index_select(1, select_indices)

            reweight = (beta * neg) / neg.mean(dim=-1)
            Ng[ind] = (reweight * neg).mean(dim=-1) * n

        # Debiased objective
        Ng = torch.max(
            (-n * tau_plus * pos + Ng).sum() / (1 - tau_plus),
            torch.exp(-1 / torch.tensor(tau, device=q.device)),
        )
        loss = (-torch.log(pos / (pos + Ng))).mean()

        return loss

    def loss_fn_select_difficulty(self, batch, output):
        initial_max_pos = 10 * 4092
        max_pos = 4092
        tau = 0.4

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

        k = k.detach()

        # limit max number of query points
        k_pos = k
        if q.shape[0] > max_pos:
            inds = np.random.choice(q.shape[0], max_pos, replace=False)
            q = q[inds]
            k_pos = k_pos[inds]

        # Negative keys should not have a gradient
        k_neg = k
        if k.shape[0] > initial_max_pos:
            inds = np.random.choice(k.shape[0], initial_max_pos, replace=False)
            k_neg = k_neg[inds]

        l_pos = torch.einsum("nc,nc->n", [q, k_pos]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, k_neg.T])

        # select difficulty
        if self.difficulty == "hard":
            values, _ = torch.sort(l_neg, dim=1, descending=True)
            l_neg = values[:, :max_pos]
        elif self.difficulty == "medium":
            values, _ = torch.sort(l_neg, dim=1, descending=True)
            min_ind = int(l_neg.shape[1] * 1 / 8)
            max_ind = int(l_neg.shape[1] * 2 / 8)
            l_neg = values[:, min_ind:max_ind]
        elif self.difficulty == "easy":
            values, _ = torch.sort(l_neg, dim=1, descending=False)
            l_neg = values[:, :max_pos]
        elif self.difficulty == "mixing":
            l_neg_new = self.get_mixed_negatives(q, k_neg.T, l_neg, max_pos)
            l_neg = torch.einsum("nc,ck->nk", [q, k_pos.T])
            l_neg = torch.cat((l_neg, l_neg_new), 1)
        else:
            # Just use other positive ks as before
            logits = torch.einsum("nc,ck->nk", [q, k_pos.T])

            npos = q.shape[0]
            labels = torch.arange(npos).to(batch.device).long()

            return self.criterion(logits, labels)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits = torch.div(logits, tau).squeeze().contiguous()

        # Labels
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.int64)

        return self.criterion(logits, labels)

    def get_mixed_negatives(self, pos_features, neg_features, l_neg, cut_off):

        # Select hard examples to mix
        _, indicies = torch.sort(l_neg, dim=1, descending=True)
        n_dim = neg_features.shape[0]
        hard_indices = indicies[:, :cut_off]

        # Select mixing coefficients and random i,j values
        # note that these are same across all positive samples
        n_pos = hard_indices.shape[0]
        n_neg = hard_indices.shape[1]
        alpha_s = torch.rand((n_neg,), device=l_neg.device)
        i_s = torch.randint(n_neg, (n_neg,))
        j_s = torch.randint(n_neg, (n_neg,))

        # # indices
        # _hard_indices_is = hard_indices[:, i_s]
        # _hard_indices_js = hard_indices[:, j_s]

        # # CHECK whole thing
        # l_is = []
        # for pos_ind in range(n_pos):
        #     neg_features_is = neg_features[:, _hard_indices_is[pos_ind]]
        #     neg_features_js = neg_features[:, _hard_indices_js[pos_ind]]

        #     mixed_features = torch.mul(alpha_s, neg_features_is) + torch.mul(
        #         1 - alpha_s, neg_features_js
        #     )

        #     mixed_features = nn.functional.normalize(mixed_features, dim=0, p=2)

        #     # compute new logits

        #     l_i = torch.mm(pos_features[pos_ind].unsqueeze(0), mixed_features)
        #     l_is.append(l_i)

        # l_neg_new_check = torch.cat(l_is, 0)

        # indices
        hard_indices_is = hard_indices[:, i_s].reshape(-1)
        hard_indices_js = hard_indices[:, j_s].reshape(-1)

        hard_neg_is = neg_features[:, hard_indices_is].T.reshape(n_pos, cut_off, -1)
        hard_neg_js = neg_features[:, hard_indices_js].T.reshape(n_pos, cut_off, -1)

        hard_neg_is = hard_neg_is.transpose(1, 2)
        hard_neg_js = hard_neg_js.transpose(1, 2)

        # Perform mixing
        alpha_s = alpha_s.repeat((n_pos, n_dim, 1))
        mixed_neg = torch.mul(alpha_s, hard_neg_is) + torch.mul(
            1 - alpha_s, hard_neg_js
        )

        # Normalize new samples
        mixed_neg = nn.functional.normalize(mixed_neg, dim=1, p=2)
        mixed_neg = mixed_neg.detach()

        # compute new logits
        l_new = torch.einsum("nc,nck->nk", [pos_features, mixed_neg])

        return l_new

    def loss_fn_mixing(self, batch, output):
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

            qs.append(q)
            ks.append(k)

        q = torch.cat(qs, 0)
        k = torch.cat(ks, 0)

        if q.shape[0] > max_pos:
            inds = np.random.choice(q.shape[0], max_pos, replace=False)
            q = q[inds]
            k = k[inds]

        outputs = [q, k]
        return self.criterion(outputs)


import open3d as o3d
from matplotlib import pyplot as plt


def get_color_map(x):
    colours = plt.cm.Spectral(x)
    return colours[:, :3]


def visualize_mapping(points1, features1, voxel_indices_1, entropies):
    points1 = points1.detach().cpu().numpy()
    features1 = features1.detach().cpu().numpy()

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    colors = features1
    # colors = get_color_map(entropies)

    # colors = np.ones(points1.shape) * np.array([0, 0.4, 0.4])
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
