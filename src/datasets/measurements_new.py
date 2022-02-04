# hold all the scene measurements

import pickle
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2

from util.types import SceneWithLabels

measurements_dir_name = "measurements"
colour_image_extension = ".color.jpg"
depth_image_extension = ".depth.pgm"
pose_extension = ".pose.txt"

D = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]
)


class SceneMeasurement:
    """Stores sensor measurements for a single frame"""

    def __init__(
        self,
        directory: Path,
        output_dir: Path,
        info: dict,
        scene: SceneWithLabels,
        frame_id: str,
    ):
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
        pose_info_name = measurements_dir / (f"frame-{frame_id}" + pose_extension)

        # load calibrations
        self.colour_intrinsics = np.fromstring(
            info["m_calibrationColorIntrinsic"], sep=" "
        ).reshape(4, 4)
        self.depth_intrinsics = np.fromstring(
            info["m_calibrationDepthIntrinsic"], sep=" "
        ).reshape(4, 4)

        # load raw data
        self.color_image = cv2.imread(str(color_image_name))
        self.pose = np.loadtxt(str(pose_info_name), delimiter=" ").astype(np.float32)

        # Resize colour image to be same as depth image
        depth_resolution = (int(info["m_depthWidth"]), int(info["m_depthHeight"]))
        self.color_image = cv2.resize(self.color_image, depth_resolution)

        # Get points visible in camera frame
        # first project points into camera frame
        T_world_to_camera = np.linalg.inv(self.pose)
        points = np.concatenate(
            (scene.points, np.ones((scene.points.shape[0], 1))), axis=1
        )
        points_camera_frame = T_world_to_camera @ points.T
        points_infront = points_camera_frame[2, :] > 0
        pixels = self.colour_intrinsics @ points_camera_frame
        pixels = pixels / pixels[2]

        valid_points = np.ones(pixels.shape[1], dtype=np.bool8)
        valid_points = np.logical_and(valid_points, pixels[0, :] > 0)
        valid_points = np.logical_and(
            valid_points, pixels[0, :] < float(info["m_colorWidth"])
        )
        valid_points = np.logical_and(valid_points, pixels[1, :] > 0)
        valid_points = np.logical_and(
            valid_points, pixels[1, :] < float(info["m_colorHeight"])
        )
        valid_points = np.logical_and(valid_points, points_infront)

        # Get point cloud for camera view
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene.points[valid_points])
        pcd.colors = o3d.utility.Vector3dVector((scene.features[valid_points] + 0.5))
        # o3d.visualization.draw_geometries([pcd])
        # mesh, pt_map = pcd.hidden_point_removal(-np.linalg.inv(self.pose)[0:3, 3], 50.0)
        # o3d.visualization.draw_geometries([mesh])

        # pcd_new = pcd.select_by_index(valid_points)
        o3d.visualization.draw_geometries([pcd])

        self.show_colour_image()

        # # Convert depth map into point cloud
        # self.points = get_point_cloud(self.depth_image, self.depth_intrinsics)
        # self.points = np.dot(self.points, np.transpose(self.pose))

        # # Get colours of points in depth map
        # mask = self.depth_image != 0
        # self.point_colors = np.reshape(self.color_image_for_point_cloud[mask], [-1, 3])
        # self.point_colors = self.point_colors / 127.5 - 1

        # # Downsample point cloud according to specified voxel size
        # pcd = self.get_open3d_point_cloud()
        # pcd = pcd.voxel_down_sample(info["voxel_size"])
        # self.points = np.asarray(pcd.points)v
        # self.point_colors = np.asarray(pcd.colors)

        # save all processed data to file
        self.save_to_file(output_dir)

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

        measurement = pickle.load(
            SceneMeasurement.get_pickle_file(directory, frame_id).open("rb")
        )

        return measurement


class SceneMeasurements:
    """Stores all sensor measurements for a single scene."""

    def __init__(
        self,
        directory: Path,
        output_dir: Path,
        scene: SceneWithLabels,
        frame_skip=25,
        voxel_size=0.02,
    ):
        self.set_directories(directory)

        # create frame measurements directory
        frame_dir = output_dir / "frames"
        if not frame_dir.exists():
            frame_dir.mkdir()

        # Extract info from info file
        self.info = self.extract_scene_info()
        self.info["voxel_size"] = voxel_size
        self.info["frame_skip"] = frame_skip

        measurements = [
            SceneMeasurement(directory, output_dir, self.info, scene, f"{frame:06d}")
            for frame in range(0, int(self.info["m_frames.size"]), frame_skip)
        ]
        self.num_measurements = len(measurements)

    def set_directories(self, root_dir):
        """Set all the directories."""
        self.directory = root_dir
        self.measurements_dir = root_dir / measurements_dir_name

    def get_measurement(self, ind):
        """Get the measurement for a specific measurement index"""
        frame_id = ind * self.info["frame_skip"]
        return SceneMeasurement.load(self.directory, frame_id)

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
