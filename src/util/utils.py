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

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not (torch.distributed.is_initialized()):
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        x = x.squeeze()
        loss = self.criterion(x, label)
        return loss


class NCELossMoco(nn.Module):
    """
    Distributed version of the NCE loss. It performs an "all_gather" to gather
    the allocated buffers like memory no a single gpu. For this, Pytorch distributed
    backend is used. If using NCCL, one must ensure that all the buffer are on GPU.
    This class supports training using both NCE and CrossEntropy (InfoNCE).
    """

    def __init__(self, config):
        super(NCELossMoco, self).__init__()

        assert config.NCE_LOSS.LOSS_TYPE in [
            "cross_entropy",
        ], f"Supported types are cross_entropy."

        self.loss_type = config.NCE_LOSS.LOSS_TYPE
        self.loss_list = config.LOSS_TYPE.split(",")
        self.other_queue = config.OTHER_INPUT

        self.npid0_w = float(config.within_format_weight0)
        self.npid1_w = float(config.within_format_weight1)
        self.cmc0_w = float(config.across_format_weight0)
        self.cmc1_w = float(config.across_format_weight1)

        self.K = int(config.NCE_LOSS.NUM_NEGATIVES)
        self.dim = int(config.NCE_LOSS.EMBEDDING_DIM)
        self.T = float(config.NCE_LOSS.TEMPERATURE)

        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if self.other_queue:
            self.register_buffer("queue_other", torch.randn(self.dim, self.K))
            self.queue_other = nn.functional.normalize(self.queue_other, dim=0)
            self.register_buffer("queue_other_ptr", torch.zeros(1, dtype=torch.long))

        # cross-entropy loss. Also called InfoNCE
        self.xe_criterion = nn.CrossEntropyLoss()

        # other constants
        self.normalize_embedding = config.NCE_LOSS.NORM_EMBEDDING

    @classmethod
    def from_config(cls, config):
        return cls(config)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, okeys=None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = torch.transpose(keys, 0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

        if self.other_queue:
            # gather keys before updating queue
            okeys = concat_all_gather(okeys)

            other_ptr = int(self.queue_other_ptr)

            # replace the keys at ptr (dequeue and enqueue)
            self.queue_other[:, other_ptr : other_ptr + batch_size] = torch.transpose(
                okeys, 0, 1
            )  # okeys.T
            other_ptr = (other_ptr + batch_size) % self.K  # move pointer

            self.queue_other_ptr[0] = other_ptr

    def forward(self, output, is_val=False):
        assert isinstance(
            output, list
        ), "Model output should be a list of tensors. Got Type {}".format(type(output))

        if self.normalize_embedding:
            normalized_output1 = nn.functional.normalize(output[0], dim=1, p=2)
            normalized_output2 = nn.functional.normalize(output[1], dim=1, p=2)
            if self.other_queue:
                normalized_output3 = nn.functional.normalize(output[2], dim=1, p=2)
                normalized_output4 = nn.functional.normalize(output[3], dim=1, p=2)

        # positive logits: Nx1
        l_pos = torch.einsum(
            "nc,nc->n", [normalized_output1, normalized_output2]
        ).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum(
            "nc,ck->nk", [normalized_output1, self.queue.clone().detach()]
        )

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        if self.other_queue:

            l_pos_p2i = torch.einsum(
                "nc,nc->n", [normalized_output1, normalized_output4]
            ).unsqueeze(-1)
            l_neg_p2i = torch.einsum(
                "nc,ck->nk", [normalized_output1, self.queue_other.clone().detach()]
            )
            logits_p2i = torch.cat([l_pos_p2i, l_neg_p2i], dim=1)
            logits_p2i /= self.T

            l_pos_i2p = torch.einsum(
                "nc,nc->n", [normalized_output3, normalized_output2]
            ).unsqueeze(-1)
            l_neg_i2p = torch.einsum(
                "nc,ck->nk", [normalized_output3, self.queue.clone().detach()]
            )
            logits_i2p = torch.cat([l_pos_i2p, l_neg_i2p], dim=1)
            logits_i2p /= self.T

            l_pos_other = torch.einsum(
                "nc,nc->n", [normalized_output3, normalized_output4]
            ).unsqueeze(-1)
            l_neg_other = torch.einsum(
                "nc,ck->nk", [normalized_output3, self.queue_other.clone().detach()]
            )
            logits_other = torch.cat([l_pos_other, l_neg_other], dim=1)
            logits_other /= self.T

        if self.other_queue:
            self._dequeue_and_enqueue(normalized_output2, okeys=normalized_output4)
        else:
            self._dequeue_and_enqueue(normalized_output2)

        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.int64)

        loss_npid = self.xe_criterion(torch.squeeze(logits), labels)

        loss_npid_other = torch.tensor(0)
        loss_cmc_p2i = torch.tensor(0)
        loss_cmc_i2p = torch.tensor(0)

        if self.other_queue:
            loss_cmc_p2i = self.xe_criterion(torch.squeeze(logits_p2i), labels)
            loss_cmc_i2p = self.xe_criterion(torch.squeeze(logits_i2p), labels)
            loss_npid_other = self.xe_criterion(torch.squeeze(logits_other), labels)

            curr_loss = 0
            for ltype in self.loss_list:
                if ltype == "CMC":
                    curr_loss += loss_cmc_p2i * self.cmc0_w + loss_cmc_i2p * self.cmc1_w
                elif ltype == "NPID":
                    curr_loss += loss_npid * self.npid0_w
                    curr_loss += loss_npid_other * self.npid1_w
        else:
            curr_loss = 0
            curr_loss += loss_npid * self.npid0_w

        loss = curr_loss

        if is_val:
            self.queue_ptr[0] = 0

        return loss, [loss_npid, loss_npid_other, loss_cmc_p2i, loss_cmc_i2p]

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


def split_data_amount_threads(data, num_threads):
    """Spread the data accross all the threads as equally as possible."""

    num_threads = min(len(data), num_threads)
    if num_threads == 0:
        return []
    num_rooms_per_thread = math.floor(len(data) / num_threads)

    def start_index(thread_ind):
        return thread_ind * num_rooms_per_thread

    def end_index(thread_ind):
        if thread_ind == num_threads - 1:
            return None
        return start_index(thread_ind) + num_rooms_per_thread

    return [
        data[start_index(thread_ind) : end_index(thread_ind)]
        for thread_ind in range(num_threads)
    ]


def apply_data_operation_in_parallel(callback, datapoints, num_threads):
    """Apply an operation on all datapoints in parallel"""
    data_per_thread = split_data_amount_threads(datapoints, num_threads)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        threads = [
            executor.submit(callback, thread_data) for thread_data in data_per_thread
        ]

    outputs = []
    for thread in threads:
        result = thread.result()
        if result:
            outputs += result

    return outputs


def get_batch_offsets(batch_idxs, bs):
    """
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    """
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def get_cli_override_arguments():
    """Returns a list of config arguements being overriden from the current command line"""
    override_args = []
    for arg in sys.argv:
        arg.replace(" ", "")
        if "=" in arg:
            override_args.append(arg.split("=")[0])

    return override_args


def add_previous_override_args_to_cli(previous_cli_override):
    """Adds override arguments to the cli if they are not already overriden"""
    for override in previous_cli_override:
        override_key, override_value = override.split("=", 1)
        if override_key not in previous_cli_override:
            sys.argv.append(override)


def load_previous_config(previous_dir: Path, current_dir: Path) -> DictConfig:

    for copy_folder in [".hydra", "lightning_logs"]:
        src_dir = previous_dir / copy_folder
        dest_dir = current_dir / copy_folder
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)

    # Load overrides config and convert to Dict
    # with command line arguments taking precedence
    override_args = get_cli_override_arguments()

    overrides_cfg = OmegaConf.load(str(current_dir / ".hydra/overrides.yaml"))
    add_previous_override_args_to_cli(overrides_cfg)
    # save_current_overrides()

    cli_conf = OmegaConf.from_cli()
    main_cfg = OmegaConf.load(str(current_dir / ".hydra/config.yaml"))
    conf = OmegaConf.merge(main_cfg, cli_conf)
    print(OmegaConf.to_yaml(conf))


def load_previous_training(previous_dir: Path, current_dir: Path) -> DictConfig:

    for copy_folder in ["lightning_logs"]:
        src_dir = previous_dir / copy_folder
        dest_dir = current_dir / copy_folder
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)


def print_error(message, user_fault=False):
    sys.stderr.write("ERROR: " + str(message) + "\n")
    if user_fault:
        sys.exit(2)
    sys.exit(-1)
