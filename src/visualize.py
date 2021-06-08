import hydra
import logging

from omegaconf import DictConfig, OmegaConf

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def visualization(cfg: DictConfig) -> None:
    pc_filename = cfg.dataset_dir + "/train/scene0000_00_vh_clean_2.labels.ply"
    print("Load a ply point cloud, print it, and render it")

    pcd = o3d.io.read_point_cloud(pc_filename)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries(
        [pcd],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024],
    )


if __name__ == "__main__":
    visualization()
