import torch
import torch.nn as nn
import numpy as np
import spconv

from packages.pointgroup_ops.functions import pointgroup_ops


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
