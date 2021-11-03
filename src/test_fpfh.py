import copy
import hydra
from omegaconf import DictConfig

import numpy as np
import open3d as o3d

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from dataloaders.data_interface import DataInterfaceFactory

from scipy.stats import wasserstein_distance


def get_color_map(x):
    colours = plt.cm.Spectral(x)
    return colours[:, :3]


def mesh_sphere(pcd, voxel_size, sphere_size=0.6):
    # Create a mesh sphere
    spheres = o3d.geometry.TriangleMesh()
    s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
    s.compute_vertex_normals()

    for i, p in enumerate(pcd.points):
        si = copy.deepcopy(s)
        trans = np.identity(4)
        trans[:3, 3] = p
        si.transform(trans)
        si.paint_uniform_color(pcd.colors[i])
        spheres += si
    return spheres


def get_colored_point_cloud_feature(pcd, feature, voxel_size):
    tsne_results = embed_tsne(feature)

    color = get_color_map(tsne_results)
    pcd.colors = o3d.utility.Vector3dVector(color)
    spheres = mesh_sphere(pcd, voxel_size)

    return spheres


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


@hydra.main(config_path="config", config_name="config")
def test_fpfh(cfg: DictConfig):
    """Test FPFH features."""

    # load a data interface
    data_interface_factory = DataInterfaceFactory(cfg)
    data_interface = data_interface_factory.get_interface()

    # Load in a scene
    scene = data_interface.train_data[0]
    scene = scene.load()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene.points)
    pcd.colors = o3d.utility.Vector3dVector(scene.features)

    # compute the features
    voxel_size = 0.04
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel_size, max_nn=30)
    )

    search_param = o3d.geometry.KDTreeSearchParamHybrid(
        radius=10 * voxel_size, max_nn=100
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)

    features = fpfh.data.T
    dist = wasserstein_distance(features[0], features[100])

    vis_pcd = get_colored_point_cloud_feature(pcd, fpfh.data.T, voxel_size)
    o3d.visualization.draw_geometries([vis_pcd])

    wasserstein_distance()


if __name__ == "__main__":
    test_fpfh()
