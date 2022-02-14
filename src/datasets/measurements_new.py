# hold all the scene measurements

import pickle
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2

from util.types import Scene, SceneWithLabels

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

    pixels = [(int(x_coord), int(y_coord)) for x_coord, y_coord in zip(x, y)]

    return points, pixels


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
        depth_image_name = measurements_dir / (
            f"frame-{frame_id}" + depth_image_extension
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
        depth_image = cv2.imread(str(depth_image_name), -1)
        depth_image = depth_image.astype(np.float32) / float(info["m_depthShift"])
        self.pose = np.loadtxt(str(pose_info_name), delimiter=" ").astype(np.float32)

        # Resize colour image to be same as depth image
        depth_resolution = (int(info["m_depthWidth"]), int(info["m_depthHeight"]))
        self.color_image = cv2.resize(self.color_image, depth_resolution)

        # Convert depth map into point cloud
        self.scan_points, self.point_to_pixel_map = get_point_cloud(
            depth_image, self.depth_intrinsics
        )
        self.scan_points = np.dot(self.scan_points, np.transpose(self.pose))

        # Point cloud uses a different ordering of RGB
        color_image = self.color_image[:, :, [2, 1, 0]]

        # Get colours of points in depth map
        mask = depth_image != 0
        self.scan_point_colors = np.reshape(color_image[mask], [-1, 3])
        self.scan_point_colors = self.scan_point_colors / 127.5 - 1

        # Get points visible in camera frame
        frame = self.get_projected_scene(info, scene)

        # # Get point cloud for camera view
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(frame.points)
        # pcd.colors = o3d.utility.Vector3dVector(frame.features + 0.5)

        # # Downsample point cloud according to specified voxel size
        # pcd = pcd.voxel_down_sample(info["voxel_size"])

        # # Visualize
        # o3d.visualization.draw_geometries([pcd])
        # self.show_colour_image()

        # Store scene
        self.points = frame.points
        self.point_colors = frame.features
        self.semantic_labels = frame.semantic_labels
        self.instance_labels = frame.instance_labels

        # save all processed data to file
        self.save_to_file(output_dir)

    def get_points_in_camera_frame(self, points):
        T_world_to_camera = np.linalg.inv(self.pose)
        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

        # Convert to camera frame and remove points behind camera
        return T_world_to_camera @ points.T

    def get_projected_scene(self, info, scene):
        """Get array of points that project into image plane."""

        points_in_camera_frame = self.get_points_in_camera_frame(scene.points)

        # Image plane projection
        pixels = self.colour_intrinsics @ points_in_camera_frame
        pixels = pixels / pixels[2]

        # Get all points that fit inside the image
        valid = [
            pixels[0, :] > 0,
            pixels[0] < float(info["m_colorWidth"]),
            pixels[1, :] > 0,
            pixels[1] < float(info["m_colorHeight"]),
            points_in_camera_frame[2, :] > 0,
        ]
        valid_points = np.ones(pixels.shape[1], dtype=np.bool8)
        for condition in valid:
            valid_points = np.logical_and(valid_points, condition)

        return SceneWithLabels(
            name="",
            points=points_in_camera_frame[0:3].T[valid_points],
            features=scene.features[valid_points],
            semantic_labels=scene.semantic_labels[valid_points],
            instance_labels=scene.instance_labels[valid_points],
        )

    @classmethod
    def remove_hidden_points(self, pcd):
        diameter = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
        diameter = np.linalg.norm(diameter)
        camera = [0, 0, -diameter]
        radius = diameter * 10000
        _, pt_map = pcd.hidden_point_removal(camera, radius)
        return pcd.select_by_index(pt_map)

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
