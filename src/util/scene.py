# hold all the scene measurements

import pickle
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
import random

from scipy.spatial import KDTree

measurements_dir_name = "measurements"
colour_image_extension = ".color.jpg"
depth_image_extension = ".depth.pgm"
pose_extension = ".pose.txt"


def get_point_cloud(depth, intrinsics):
    """
    Convert the depth image into points in the camera's reference frame.
    Apply pose transformation if supplied
    """

    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]

    x_coords = np.linspace(0, depth.shape[1] - 1, depth.shape[1])
    y_coords = np.linspace(0, depth.shape[0] - 1, depth.shape[0])
    x, y = np.meshgrid(x_coords, y_coords)

    mask = np.where(depth != 0)
    x = x[mask]
    y = y[mask]
    depth = depth[mask]

    n = depth.shape[0]
    points = np.ones((n, 4))
    points[:, 0] = (x - cx) * depth / fx
    points[:, 1] = (y - cy) * depth / fy
    points[:, 2] = np.copy(depth)

    return points


class SceneMeasurement:
    """Stores sensor measurements for a single frame"""

    def __init__(self, directory: Path, info: dict, frame_id: str):
        """
        Load all measurements of a single frame from a scene catpure

        Parameters
        ----------

        directory: Path
        path to the root of the .sens extracted outputs

        info: dict
        scene capture parameters stored in the _info.txt file of extracted
        .sens measurements

        frame_id: str
        the id of the frame

        """

        self.frame_id = frame_id

        # Set filenames
        measurements_dir = directory / measurements_dir_name
        color_image_name = measurements_dir / (
            f"frame-{frame_id}" + colour_image_extension
        )
        depth_image_name = measurements_dir / (
            f"frame-{frame_id}" + depth_image_extension
        )
        pose_info_name = measurements_dir / (f"frame-{frame_id}" + pose_extension)

        frame_number = int(self.frame_id)
        instance_image_name = directory / (f"instance-filt/{frame_number}.png")
        label_image_name = directory / (f"label-filt/{frame_number}.png")

        # load calibrations
        self.colour_intrinsics = np.fromstring(
            info["m_calibrationColorIntrinsic"], sep=" "
        ).reshape(4, 4)
        self.depth_intrinsics = np.fromstring(
            info["m_calibrationDepthIntrinsic"], sep=" "
        ).reshape(4, 4)

        # load raw data
        self.color_image = cv2.imread(str(color_image_name))
        self.depth_image = cv2.imread(str(depth_image_name), -1)
        self.depth_image = self.depth_image.astype(np.float32) / float(
            info["m_depthShift"]
        )
        self.pose = np.loadtxt(str(pose_info_name), delimiter=" ").astype(np.float32)

        # load annotations
        self.label_image = cv2.imread(str(label_image_name))
        self.instance_image = cv2.imread(str(directory / instance_image_name))

        # Resize colour image to be same as depth image
        depth_resolution = (int(info["m_depthWidth"]), int(info["m_depthHeight"]))
        self.color_image = cv2.resize(self.color_image, depth_resolution)
        self.instance_image = cv2.resize(
            self.instance_image,
            depth_resolution,
            interpolation=cv2.INTER_NEAREST,
        )
        self.label_image = cv2.resize(
            self.label_image,
            depth_resolution,
            interpolation=cv2.INTER_NEAREST,
        )

        # Convert depth map into point cloud
        self.points = get_point_cloud(self.depth_image, self.depth_intrinsics)
        self.points = np.dot(self.points, np.transpose(self.pose))

        # Get colours of points in depth map
        mask = self.depth_image != 0
        self.point_colors = np.reshape(self.color_image_for_point_cloud[mask], [-1, 3])
        self.point_colors = self.point_colors / 127.5 - 1

        # Downsample point cloud according to specified voxel size
        pcd = self.get_open3d_point_cloud()
        pcd = pcd.voxel_down_sample(info["voxel_size"])
        self.points = np.asarray(pcd.points)
        self.point_colors = np.asarray(pcd.colors)

        # save all processed data to file
        self.save_to_file(directory)

    @property
    def color_image_for_point_cloud(self):
        """Point cloud uses a different ordering of RGB."""
        return self.color_image[:, :, [2, 1, 0]]

    def colour_annotations(self, image, id_to_colour):
        """Assign colours to each annotation in the image."""
        annotations = np.ones(image.shape).astype(np.uint8)
        id_by_pixel = image[:, :, 0]
        for id in np.unique(id_by_pixel):
            colour = id_to_colour[id]
            annotations[id_by_pixel == id] = colour

        return annotations

    def get_open3d_point_cloud(self):
        """Generate an open3D point cloud for visualization."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[:, 0:3])
        pcd.colors = o3d.utility.Vector3dVector(self.point_colors)
        return pcd

    def show_pointcloud(self):
        """Display the point cloud."""
        pcd = self.get_open3d_point_cloud()
        o3d.visualization.draw_geometries([pcd])

    def show_depth_image(self):
        """Display the depth image."""
        depth = self.depth_image.astype(np.float32)
        depth = (depth * 255) / np.max(depth)

        cv2.imshow("Depth", depth.astype(np.uint8))
        cv2.waitKey()

    def show_colour_image(self):
        """Display the colour image."""
        cv2.imshow("Color", self.color_image)
        cv2.waitKey()

    @staticmethod
    def get_pickle_file(directory, frame_id):
        if not type(frame_id) == str:
            frame_id = format(frame_id, "06d")
        return directory / "frames" / f"frame_{frame_id}.pkl"

    def save_to_file(self, directory):
        """Pickle the entire object into a single binary file."""
        pickle.dump(self, self.get_pickle_file(directory, self.frame_id).open("wb"))

    @staticmethod
    def load(directory: Path, frame_id):
        """Load from a previously pickled measurements file."""
        return pickle.load(
            SceneMeasurement.get_pickle_file(directory, frame_id).open("rb")
        )


class SceneMeasurements:
    """Stores all sensor measurements for a single scene."""

    def __init__(self, directory: Path, frame_skip=25, voxel_size=0.05):
        self.set_directories(directory)

        # create frame measurements directory
        frame_dir = self.directory / "frames"
        if not frame_dir.exists():
            frame_dir.mkdir()

        # Extract info from info file
        self.info = self.extract_scene_info()
        self.info["voxel_size"] = voxel_size
        self.info["frame_skip"] = frame_skip

        # Initalize colours
        self.label_id_to_colour = {i: self.get_random_colour() for i in range(100)}
        self.instance_id_to_colour = {i: self.get_random_colour() for i in range(100)}

        measurements = [
            SceneMeasurement(directory, self.info, f"{frame:06d}")
            for frame in range(0, int(self.info["m_frames.size"]), frame_skip)
        ]
        self.num_measurements = len(measurements)

        # extract inter-measurement info
        indices_map = self.get_overlap_indices(measurements)
        self.overlap_matrix = self.get_overlapping_measurements(indices_map)
        self.correspondance_map = self.get_correspondance_map(
            indices_map, self.overlap_matrix
        )
        self.matching_frames_map = self.get_matching_frames_map(self.overlap_matrix)

    def set_directories(self, root_dir):
        """Set all the directories."""
        self.directory = root_dir
        self.measurements_dir = root_dir / measurements_dir_name

    def get_measurement(self, ind):
        """Get the measurement for a specific measurement index"""
        frame_id = ind * self.info["frame_skip"]
        return SceneMeasurement.load(self.directory, frame_id)

    def get_overlap_indices(self, measurements, search_mult=2.0):
        """Get the overlap of measurements based on their relative pose error."""
        indexes_map = {}

        kd_trees = {i: KDTree(frame.points) for i, frame in enumerate(measurements)}

        for i, frame1 in enumerate(measurements):
            indexes_map[i] = {}
            for j in range(0, i):

                frame2 = measurements[j]

                # Compute pose1 inverse
                R1 = frame1.pose[0:3, 0:3]
                T1 = frame1.pose[0:3, 3]
                pose1_inv = frame1.pose.copy()
                pose1_inv[0:3, 0:3] = R1.T
                pose1_inv[0:3, 3] = -R1.T @ T1

                pose_diff = pose1_inv @ frame2.pose
                dT = np.linalg.norm(pose_diff[0:3, 3])
                dtheta = np.arccos((np.trace(pose_diff[0:3, 0:3]) - 1) / 2)

                if dT < 2.0 and dtheta < (3.14 / 1.5):
                    kd_tree1 = kd_trees[i]
                    kd_tree2 = kd_trees[j]

                    indexes_map[i][j] = kd_tree1.query_ball_tree(
                        kd_tree2, search_mult * self.info["voxel_size"], p=1
                    )

        return indexes_map

    def get_overlapping_measurements(self, indices_map):
        overlap_matrix = np.zeros((self.num_measurements, self.num_measurements))

        for i, indices_map_i in indices_map.items():
            for j, indices_i_j in indices_map_i.items():

                num_matches = sum(1 for matches in indices_i_j if matches)

                if num_matches > 1000:
                    overlap_matrix[i, j] = 1.0
                    # overlap_matrix[j, i] = 1.0

        return overlap_matrix

    def get_correspondance_map(self, indices_map, overlap_matrix, max=4092):
        """Returns the number of correspondacnes."""
        correspondances_map = {}
        for i, indices_map_i in indices_map.items():
            correspondances_map[i] = {}
            for j, indices_i_j in indices_map_i.items():
                if overlap_matrix[i, j]:
                    correspondances_map[i][j] = {
                        point: index[0]
                        for point, index in enumerate(indices_i_j)
                        if index
                    }

        return correspondances_map

    def get_matching_frames_map(self, overlap_matrix, threshold=0.3):
        """Get the map of each frame with it's corresponding matching frames."""
        return {
            i: np.where(overlap_matrix[i] > threshold)[0]
            for i in range(overlap_matrix.shape[0])
            if np.sum(overlap_matrix[i]) != 0
        }

    def get_random_colour(self):
        return np.random.choice(range(256), size=3).astype(np.uint8)

    def extract_scene_info(self):
        """Extract key value pairs stored in _info.txt."""
        info = {}
        info_file = self.measurements_dir / "_info.txt"

        with info_file.open("r") as f:
            for line in f:
                line = line.strip("\n")
                elements = line.split(" = ")
                info[elements[0]] = elements[1]

        return info

    def visualize_scene(self):
        """Visulaize the entire scene put together."""
        o3d.visualization.draw_geometries(
            [measurement.pcd for measurement in self.measurements]
        )

    def visualize_everything_as_video(self, fps):
        """Visualize labels, instance and raw input as seperate video frames."""
        streams = {
            "color": self.get_colour_frames(),
            "label": self.get_label_frames(),
            "instance": self.get_instance_frames(),
        }
        self.visualize_as_video_multiple(streams, 2)

    def visualize_labels_as_video(self, fps):
        """Visualize all frames with instances as a video."""
        self.visualize_as_video(self.get_label_frames(), fps)

    def visualize_instances_as_video(self, fps):
        """Visualize all frames with instances as a video."""
        self.visualize_as_video(self.get_instance_frames(), fps)

    def visualize_as_video(self, images, fps):
        """Turn a set of images into a video for visualization."""
        for image in images:
            cv2.imshow("Frame", image)
            cv2.waitKey(int(1 / fps * 1000))

        cv2.destroyAllWindows()

    def visualize_as_video_multiple(self, image_streams: dict, fps):
        """Visualize multiple streams of images such as colour and
        labels at the same time."""

        # Make sure all the steams have the same length
        num_frames = len(self.measurements)
        for stream in image_streams.values():
            if len(stream) != num_frames:
                raise RuntimeError("Streams must be same length")

        # Run through each video stream
        for frame_ind in range(num_frames):
            for name, stream in image_streams.items():
                cv2.imshow(name, stream[frame_ind])
            cv2.waitKey(int(1 / fps * 1000))

        cv2.destroyAllWindows()

    def save_as_video(self, images, video_name, fps):
        """Turn a set of images in a video."""
        dimensions = images[0].shape[0:2]
        fourcc = cv2.VideoWriter_fourcc(*"MP42")
        video = cv2.VideoWriter(video_name, fourcc, float(fps), dimensions)

        for image in images:
            video.write(image)

        cv2.destroyAllWindows()
        video.release()

    def get_label_frames(self):
        """Get a list of all label images."""
        return [
            measurement.colour_annotations(
                measurement.label_image, self.label_id_to_colour
            )
            for measurement in self.measurements
        ]

    def get_instance_frames(self):
        """Get a list of all instance images."""
        return [
            measurement.colour_annotations(
                measurement.instance_image, self.instance_id_to_colour
            )
            for measurement in self.measurements
        ]

    def get_colour_frames(self):
        """Get a list of all colour images."""
        return [measurement.color_image for measurement in self.measurements]

    def save_to_file(self, directory):
        """Pickle the entire object into a single binary file."""
        pickled_file = directory / (directory.name + "_measurements.pkl")
        pickle.dump(self, pickled_file.open("wb"))

    @staticmethod
    def load_scene(directory: Path):
        """
        Load from a previously pickled measurements file.

        Parameters
        ----------

        directory: Path
        Root path to the directory with the pickled measurements file.
        The name of the directory should match the root of the pickled filename.

        returns
        -------
        SceneMeasurements object loaded from the pickled measurements file.
        """
        pickled_file = directory / (directory.name + "_measurements.pkl")
        scene = pickle.load(pickled_file.open("rb"))
        scene.set_directories(directory)
        return scene


# if __name__ == "__main__":
#     scans_dir = Path.cwd() / "/media/starslab/datasets/aribic/scannet/scans"
#     output_dir = Path.cwd() / "/media/starslab/datasets/aribic/scannet/outputs"

#     extracted_scenes = [scan for scan in output_dir.iterdir() if scan.is_dir()]
#     for scene in extracted_scenes:
#         measurements = SceneMeasurements(scene)
#         measurements.visualize_everything_as_video(2)


# Extracting code
# if not output_dir.exists():
#     output_dir.mkdir()

# file_extensions = [
#     ".sens",
#     "_2d-label.zip",
#     "_2d-instance.zip",
#     "_2d-label-filt.zip",
#     "_2d-instance-filt.zip",
# ]


# def is_download_complete(scene):
#     for ext in file_extensions:
#         filename = scene / (scene.name + ext)
#         if not filename.exists():
#             return False

#     return True


# current_scenes = [scan for scan in scans_dir.iterdir() if scan.is_dir()]
# scenes_to_extract = [scene for scene in current_scenes if is_download_complete(scene)]

# import zipfile

# zipfiles_to_extract = [
#     "_2d-instance.zip",
#     "_2d-instance-filt.zip",
#     "_2d-label.zip",
#     "_2d-label-filt.zip",
# ]

# # Load scene measurements
# extracted_scenes = [scan for scan in output_dir.iterdir() if scan.is_dir()]
# for scene in extracted_scenes:

#     for ext in zipfiles_to_extract:
#         file_to_extract = scans_dir / scene.name
#         file_to_extract /= scene.name + ext
#         with zipfile.ZipFile(file_to_extract) as zp:
#             zp.extractall(output_dir / scene.name)

#     measurements = SceneMeasurements(scene)
#     measurements.save_to_file()

#     loaded_measurements = SceneMeasurements.load_scene(scene)

# for scene in scenes_to_extract:

#     scene_output = output_dir / scene.name
#     if not scene_output.exists():
#         scene_output.mkdir()

#     p = Popen(
#         [
#             "./sens",
#             scene / (scene.name + ".sens"),
#             scene_output,
#         ],
#         stdin=PIPE,
#     )
#     while True:
#         if not p.poll() is None:
#             break
