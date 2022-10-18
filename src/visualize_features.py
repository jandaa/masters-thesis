import copy
import hydra
from omegaconf import DictConfig
from pathlib import Path

import numpy as np
import open3d as o3d
import pytorch_lightning as pl

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from datasets.interface import DataInterfaceFactory
from models.factory import ModelFactory

from util.utils import get_random_colour

from scipy.stats import wasserstein_distance


def get_color_map(x):
    colours = plt.cm.Spectral(x)
    return colours[:, :3]


def mesh_sphere(pcd, voxel_size, sphere_size=40.0):
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


def get_colored_point_cloud_feature(
    pcd, feature, voxel_size, color_map=None, selected=None
):
    if type(color_map) != np.ndarray:
        tsne_results = embed_tsne(feature)
        color = get_color_map(tsne_results)
    else:
        color = get_color_map(color_map)

    if selected:
        color[selected] = np.array([0, 0, 0])

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
def test_backbone(cfg: DictConfig):
    """Test backbone features."""

    pl.seed_everything(42, workers=True)

    # load a data interface
    data_interface_factory = DataInterfaceFactory(cfg)
    data_interface = data_interface_factory.get_interface()

    # Load a model
    model_factory = ModelFactory(cfg, data_interface)
    backbone_wrapper_type = model_factory.get_backbone_wrapper_type()
    pretrain_checkpoint = str(Path.cwd() / "pretrain_checkpoints" / cfg.checkpoint)
    backbone = backbone_wrapper_type(cfg)
    # backbone = backbone_wrapper_type.load_from_checkpoint(
    #     cfg=cfg,
    #     checkpoint_path=pretrain_checkpoint,
    # )
    import torch

    # ckpt = torch.load(pretrain_checkpoint)
    # backbone.model.load_state_dict(ckpt["state_dict"], strict=False)

    # import torch

    # model = model_factory.get_model()
    # pretrain_checkpoint = str(Path.cwd() / "checkpoints" / cfg.checkpoint)
    # state_dict = torch.load(pretrain_checkpoint)["state_dict"]
    # for weight in state_dict.keys():
    #     if "model.backbone" not in weight:
    #         del state_dict[weight]

    # model.load_state_dict(state_dict, strict=False)
    # backbone = model.model.backbone

    # backbone = backbone_wrapper_type(cfg)

    dataset_type = model_factory.get_backbone_dataset_type()
    dataset = dataset_type(data_interface.pretrain_val_data, cfg)
    collate_fn = dataset.collate
    scene = collate_fn([dataset[133]])
    # 30 & 300

    import MinkowskiEngine as ME

    first_frame = np.where(scene.points[:, 0] == 0)[0]
    points = scene.points[first_frame, :]
    colors = scene.features[first_frame, :]
    model_input = ME.SparseTensor(colors, points)

    backbone = backbone.eval()
    output = backbone.model(model_input)
    features = output.F.detach().numpy()

    points = points[:, 1:4] * 0.02

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis_pcd = get_colored_point_cloud_feature(pcd, features, 0.04)
    o3d.visualization.draw_geometries([vis_pcd])


from math import log, e


def entropy(feature, base=None):
    """Compute entropy of a feature vector."""

    if feature.sum() == 0:
        return 0

    feature = feature / feature.sum()

    entropy = 0

    # Compute entropy
    base = e if base is None else base
    for i in feature:
        if i != 0:
            entropy -= i * log(i, base)

    return entropy


import random


def compute_fpfh(pcd):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=100))

    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=7, max_nn=200)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)

    return fpfh.data.T


import pickle


@hydra.main(config_path="config", config_name="config")
def test_fpfh(cfg: DictConfig):
    """Test FPFH features."""

    pl.seed_everything(70, workers=True)

    # load a data interface
    data_interface_factory = DataInterfaceFactory(cfg)
    data_interface = data_interface_factory.get_interface()

    # Load in a scene
    model_factory = ModelFactory(cfg, data_interface)
    dataset_type = model_factory.get_dataset_type()
    dataset = dataset_type(data_interface.val_data[:100], cfg)
    collate_fn = dataset.collate

    # # get a couple of values
    # fpfh_features = []
    # for i in range(50):

    #     scene_ind = random.randint(0, len(dataset) - 1)
    #     scene = collate_fn([dataset[scene_ind]])

    #     points = scene.points[:, 1:4]

    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points)

    #     # o3d.visualization.draw_geometries([pcd])

    #     features = compute_fpfh(pcd)
    #     fpfh_features.append(features)

    # fpfh_features = np.concatenate(fpfh_features)
    # np.save("fpfh_features.npy", fpfh_features)

    # fpfh_features = np.load("fpfh_features.npy")

    num_clusters = 2
    cluster_colors = np.array([get_random_colour() for i in range(num_clusters)])

    # kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    # kmeans.fit(fpfh_features)

    # pickle.dump(kmeans, open("clustering.pkl", "wb"))

    kmeans = pickle.load(open("clustering.pkl", "rb"))

    # scene_ind = random.randint(0, len(dataset) - 1)
    scene_ind = 25
    scene = collate_fn([dataset[scene_ind]])

    points = scene.points[:, 1:4]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # compute the features
    features = compute_fpfh(pcd)

    predicted_clusters = kmeans.predict(features)
    colours = cluster_colors[predicted_clusters]
    pcd.colors = o3d.utility.Vector3dVector(colours)

    o3d.visualization.draw_geometries([pcd])


@hydra.main(config_path="config", config_name="config")
def test_fpfh_distance(cfg: DictConfig):
    """Test FPFH features."""

    pl.seed_everything(56, workers=True)

    # load a data interface
    data_interface_factory = DataInterfaceFactory(cfg)
    data_interface = data_interface_factory.get_interface()

    # Load in a scene
    model_factory = ModelFactory(cfg, data_interface)
    dataset_type = model_factory.get_backbone_dataset_type()
    dataset = dataset_type(data_interface.pretrain_val_data, cfg)
    collate_fn = dataset.collate
    scene = collate_fn([dataset[0]])

    first_frame = np.where(scene.points[:, 0] == 0)[0]
    points = scene.points[first_frame, :]
    # colors = scene.features[first_frame, :]

    points = points[:, 1:4] * 0.02
    features = scene.features

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(features)

    # compute the features
    voxel_size = 0.02
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=20 * voxel_size, max_nn=500)
    )

    search_param = o3d.geometry.KDTreeSearchParamHybrid(
        radius=30 * voxel_size, max_nn=200
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)

    features = fpfh.data.T

    # Pick a point and compute distances to all other features
    ind = 4500
    color_map = np.array(
        [wasserstein_distance(features[ind], feature) for feature in features]
    )

    # Normalize color map
    color_map = color_map - color_map.min()
    color_map = color_map / color_map.max()

    # dist = wasserstein_distance(features[0], features[100])

    vis_pcd = get_colored_point_cloud_feature(
        pcd, fpfh.data.T, voxel_size, color_map=color_map, selected=np.array([ind])
    )
    o3d.visualization.draw_geometries([vis_pcd])

    # wasserstein_distance()


@hydra.main(config_path="config", config_name="config")
def test_fpfh_entropy(cfg: DictConfig):
    """Test FPFH features."""

    pl.seed_everything(65, workers=True)

    # load a data interface
    data_interface_factory = DataInterfaceFactory(cfg)
    data_interface = data_interface_factory.get_interface()

    # Load in a scene
    model_factory = ModelFactory(cfg, data_interface)
    dataset_type = model_factory.get_backbone_dataset_type()
    dataset = dataset_type(data_interface.pretrain_val_data, cfg)
    collate_fn = dataset.collate
    scene = collate_fn([dataset[20]])

    first_frame = np.where(scene.points[:, 0] == 0)[0]
    points = scene.points[first_frame, :]
    # colors = scene.features[first_frame, :]

    points = points[:, 1:4]
    features = scene.features

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(features)

    # compute the features
    voxel_size = 0.02
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=100))
    # pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=100))

    # search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=6, max_nn=200)
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=7, max_nn=200)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)

    features = fpfh.data.T
    # color_map = np.array([entropy(feature) for feature in features])

    # # Normalize color map
    # color_map = color_map - color_map.min()
    # color_map = color_map / color_map.max()
    # color_map = 1 - color_map
    # dist = wasserstein_distance(features[0], features[100])

    # Select points based off of entropy values
    import random

    # probability = 1 - color_map
    # probability[np.where(probability < 0.4)[0]] = 0.0
    # selected = list(np.where(probability < 0.4)[0])
    # probability = probability / probability.sum()
    # selected = random.choices(range(len(color_map)), weights=probability, k=1000)
    # selected = random.choices(range(len(color_map)), k=1000)

    vis_pcd = get_colored_point_cloud_feature(pcd, fpfh.data.T, voxel_size)
    o3d.visualization.draw_geometries([vis_pcd])

    # wasserstein_distance()


if __name__ == "__main__":
    # test_backbone()
    test_fpfh()
