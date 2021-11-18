import math
from torch.utils.data.dataset import Dataset


class SegmentationDataset(Dataset):
    """Base segmentation dataset"""

    def __init__(self, scenes, cfg, is_test=False):
        super(SegmentationDataset, self).__init__()

        self.cfg = cfg
        self.num_workers = cfg.model.train.train_workers
        self.ignore_label = cfg.dataset.ignore_label
        self.is_test = is_test
        self.voxel_size = cfg.dataset.voxel_size
        self.scale = 1 / cfg.dataset.voxel_size
        self.mode = cfg.dataset.mode

        self.max_npoint = cfg.dataset.max_npoint
        self.max_pointcloud_size = cfg.model.test.max_pointcloud_size

        # Duplicate large scenes because they will be
        if not is_test:
            scenes = self.duplicate_large_scenes(scenes)
        self.scenes = scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        raise NotImplementedError("Need to implement __getitem__")

    def collate(self, batch):
        raise NotImplementedError("Need to implement a custom collate function")

    def duplicate_large_scenes(self, datapoints):
        """
        Duplicate large scenes by the number of times they are larger than the
        max number of points. This is because they will be cropped later in the
        merge function to the max points, but we want to use more of the scene.
        """
        new_datapoints = []
        for datapoint in datapoints:
            number_of_splits = math.floor(datapoint.num_points / self.max_npoint)
            number_of_splits = max(number_of_splits, 1)
            new_datapoints += [datapoint] * number_of_splits

        return new_datapoints


class PretrainDataset(Dataset):
    """Base dataset for pretraining tasks."""

    def __init__(self, scenes, cfg, is_test=False):
        super(PretrainDataset, self).__init__()

        self.cfg = cfg
        self.scenes = scenes
        self.num_workers = cfg.model.train.train_workers
        self.ignore_label = cfg.dataset.ignore_label
        self.voxel_size = cfg.dataset.voxel_size

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        raise NotImplementedError("Need to implement __getitem__")

    def collate(self, batch):
        raise NotImplementedError("Need to implement a custom collate function")
