import sys
import shutil
import math
import pprint
from pathlib import Path
import concurrent.futures
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import open3d as o3d
import numpy as np


class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        x = x.squeeze()
        loss = self.criterion(x, label)
        return loss


class NCELossMoco(nn.Module):
    def __init__(self, config):
        super(NCELossMoco, self).__init__()

        self.K = (
            config.pretrain.loss.num_neg_points * config.pretrain.loss.queue_multiple
        )
        self.dim = config.net.model_n_out
        self.T = config.pretrain.loss.temperature
        self.difficulty = config.pretrain.loss.difficulty

        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # cross-entropy loss. Also called InfoNCE
        self.xe_criterion = nn.CrossEntropyLoss()

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

    def forward(self, output):

        normalized_output1 = nn.functional.normalize(output[0], dim=1, p=2)
        normalized_output2 = nn.functional.normalize(output[1], dim=1, p=2)

        # positive logits: Nx1
        l_pos = torch.einsum(
            "nc,nc->n", [normalized_output1, normalized_output2]
        ).unsqueeze(-1)

        # negative logits: NxK
        neg_features = self.queue.clone().detach()
        l_neg = torch.einsum("nc,ck->nk", [normalized_output1, neg_features])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # Why do this at the end?
        self._dequeue_and_enqueue(normalized_output2)

        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.int64)

        return self.xe_criterion(torch.squeeze(logits), labels)

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "loss_type": self.loss_type,
        }
        return pprint.pformat(repr_dict, indent=2)


class Visualizer:
    """
    Visualizer class that will show semantic and instance
    coloured point clouds of test scenes

    Key commands
    ------------

    a - previous scene
    d - next scene
    w - next task
    s - previous task
    q - toggle between source (predictions and ground truth)
    e - toggle between input and segmentation
    """

    def __init__(self, directory: Path):

        if not directory.exists():
            raise RuntimeError(
                "No model predictions available to visualize, please run eval first"
            )

        self.scenes = [scene for scene in directory.iterdir() if scene.is_dir()]
        self.tasks = ["semantic", "instance"]
        self.source = ["pred", "gt"]
        self.show_input = False

        self.scene_ind = 0
        self.task_ind = 0
        self.source_ind = 0

    def get_current_point_cloud(self):
        if self.show_input:
            return o3d.io.read_point_cloud(
                str(self.scenes[self.scene_ind] / "input.pcd")
            )

        return o3d.io.read_point_cloud(
            str(
                self.scenes[self.scene_ind]
                / (
                    self.tasks[self.task_ind]
                    + "_"
                    + self.source[self.source_ind]
                    + ".pcd"
                )
            )
        )

    def load_new_scene(self, vis):
        pcd = self.get_current_point_cloud()

        # Update point cloud
        vis.clear_geometries()
        vis.add_geometry(pcd, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()

        # Update window name
        vis.create_window(
            window_name=f"\
                Scene: {self.scenes[self.scene_ind].stem} \
                Task: {self.tasks[self.task_ind]} \
                Source: {self.source[self.source_ind]}"
        )
        return False

    def next_scene(self, vis):
        self.scene_ind = min(self.scene_ind + 1, len(self.scenes) - 1)
        return self.load_new_scene(vis)

    def previous_scene(self, vis):
        self.scene_ind = max(self.scene_ind - 1, 0)
        return self.load_new_scene(vis)

    def next_task(self, vis):
        self.task_ind = min(self.task_ind + 1, len(self.tasks) - 1)
        return self.load_new_scene(vis)

    def previous_task(self, vis):
        self.task_ind = max(self.task_ind - 1, 0)
        return self.load_new_scene(vis)

    def toggle_source(self, vis):
        self.source_ind = (self.source_ind + 1) % len(self.source)
        return self.load_new_scene(vis)

    def toggle_input(self, vis):
        self.show_input = not self.show_input
        return self.load_new_scene(vis)

    def visualize_results(self):
        """Visualize all point clouds"""

        pcd = self.get_current_point_cloud()

        key_to_callback = {}
        key_to_callback[ord("D")] = self.next_scene
        key_to_callback[ord("A")] = self.previous_scene
        key_to_callback[ord("W")] = self.next_task
        key_to_callback[ord("S")] = self.previous_task
        key_to_callback[ord("Q")] = self.toggle_source
        key_to_callback[ord("E")] = self.toggle_input
        o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


def get_random_colour():
    return np.random.choice(range(256), size=3).astype(np.float) / 255.0


def visualize_pointcloud(points, colours):
    """Convenience function to quickly visualize a point cloud."""

    pcd = o3d.geometry.PointCloud()
    if type(points) == np.ndarray:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colours)
    else:
        pcd.points = o3d.utility.Vector3dVector(points.cpu().detach().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colours.cpu().detach().numpy())
    o3d.visualization.draw_geometries([pcd])


def print_error(message, user_fault=False):
    sys.stderr.write("ERROR: " + str(message) + "\n")
    if user_fault:
        sys.exit(2)
    sys.exit(-1)
