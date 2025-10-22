import numpy as np
import torch


class ModelDoubleSphere:
    """Implemented according to:
    V. Usenko, N. Demmel, and D. Cremers: The Double Sphere Camera Model.
    Proceedings of the International Conference on 3D Vision (3DV) (2018).
    URL: https://arxiv.org/pdf/1807.08957.pdf."""

    def __init__(self, xi, alpha, fx, fy, cx, cy, shape_image):
        self.alpha = alpha
        self.cx = cx
        self.cy = cy
        self.device = None
        self.fx = fx
        self.fy = fy
        self.shape_image = shape_image
        self.xi = xi

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_camera_info_message(cls, message):
        try:
            binning_x = message.binning_x if message.binning_x != 0 else 1
            binning_y = message.binning_y if message.binning_y != 0 else 1
            offset_x = message.roi.offset_x
            offset_y = message.roi.offset_y
        except AttributeError:
            binning_x = 1
            binning_y = 1
            offset_x = 0
            offset_y = 0

        # Do not know channel dimension from camera info message
        shape_image = (-1, message.height, message.width)

        xi = message.d[0]
        alpha = message.d[1]
        fx = message.k[0] / binning_x
        fy = message.k[4] / binning_y
        cx = (message.k[2] - offset_x) / binning_x
        cy = (message.k[5] - offset_y) / binning_y

        instance = cls(xi, alpha, fx, fy, cx, cy, shape_image)
        return instance

    @torch.inference_mode()
    def project_points_onto_image(self, coords_xyz, use_invalid_coords=True, use_mask_fov=True, use_half_precision=True):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        if use_half_precision:
            coords_xyz = coords_xyz.half()

        coords_xyz = coords_xyz.to(self.device)

        x, y, z = coords_xyz[:, 0, :], coords_xyz[:, 1, :], coords_xyz[:, 2, :]

        # Eq. (41)
        d1 = torch.sqrt(x**2 + y**2 + z**2)
        # Eq. (45)
        w1 = self.alpha / (1.0 - self.alpha) if self.alpha <= 0.5 else (1.0 - self.alpha) / self.alpha
        # Eq. (44)
        w2 = (w1 + self.xi) / np.sqrt(2.0 * w1 * self.xi + self.xi**2 + 1.0)
        # Eq. (43)
        mask_valid = z > -w2 * d1

        # Note: Only working for batchsize 1
        if not use_invalid_coords and mask_valid.shape[0] == 1:
            x = x[mask_valid][None, ...]
            y = y[mask_valid][None, ...]
            z = z[mask_valid][None, ...]
            d1 = d1[mask_valid][None, ...]
            mask_valid = torch.ones_like(z, dtype=torch.bool)

        # Eq. (42)
        z_shifted = self.xi * d1 + z
        d2 = torch.sqrt(x**2 + y**2 + z_shifted**2)
        # Eq. (40)
        denominator = self.alpha * d2 + (1 - self.alpha) * z_shifted
        u = self.fx * x / denominator + self.cx
        v = self.fy * y / denominator + self.cy
        coords_uv = torch.stack((u, v), dim=1)

        if use_mask_fov:
            mask_left = coords_uv[:, 0, :] >= 0
            mask_top = coords_uv[:, 1, :] >= 0
            mask_right = coords_uv[:, 0, :] < self.shape_image[2]
            mask_bottom = coords_uv[:, 1, :] < self.shape_image[1]
            mask_valid *= mask_left * mask_top * mask_right * mask_bottom

        return coords_uv, mask_valid

    @torch.inference_mode()
    def project_image_onto_points(self, coords_uv, use_invalid_coords=True, use_half_precision=True):
        """Project 2D image onto 3D unit sphere.
        Shape of coords_uv: (B, 2, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        if use_half_precision:
            coords_uv = coords_uv.half()

        coords_uv = coords_uv.to(self.device)

        u, v = coords_uv[:, 0, :], coords_uv[:, 1, :]

        # Eq. (47)
        mx = (u - self.cx) / self.fx
        # Eq. (48)
        my = (v - self.cy) / self.fy
        # Eq. (49)
        square_r = mx**2 + my**2
        # Eq. (51) can be written to use this
        term = 1.0 - (2.0 * self.alpha - 1.0) * square_r
        # Eq. (51)
        mask_valid = term >= 0.0 if self.alpha > 0.5 else torch.ones_like(term, dtype=torch.bool)

        # Note: Only working for batchsize 1
        if not use_invalid_coords and mask_valid.shape[0] == 1:
            mx = mx[mask_valid][None, ...]
            my = my[mask_valid][None, ...]
            square_r = square_r[mask_valid][None, ...]
            term = term[mask_valid][None, ...]
            mask_valid = torch.ones_like(term, dtype=torch.bool)

        # Eq. (50)
        mz = (1.0 - self.alpha**2 * square_r) / (self.alpha * torch.sqrt(term) + 1.0 - self.alpha)
        # Eq. (46)
        factor = (mz * self.xi + torch.sqrt(mz**2 + (1.0 - self.xi**2) * square_r)) / (mz**2 + square_r)
        coords_xyz = factor[:, None, :] * torch.stack((mx, my, mz), dim=1)
        coords_xyz[:, 2, :] -= self.xi

        return coords_xyz, mask_valid


class ModelCamera:
    """Base class for camera models."""

    def __init__(self, fx, fy, cx, cy, model_distortion, params_distortion, shape_image):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.model_distortion = model_distortion
        self.params_distortion = params_distortion
        self.shape_image = shape_image

    @classmethod
    def from_camera_info_message(cls, message):
        """Create an instance from a camera info message."""
        try:
            binning_x = message.binning_x if message.binning_x != 0 else 1
            binning_y = message.binning_y if message.binning_y != 0 else 1
        except AttributeError:
            binning_x = 1
            binning_y = 1

        try:
            offset_x = message.roi.offset_x
            offset_y = message.roi.offset_y
            # Do not know channel dimension from camera info message but keep it for pytorch-like style
            shape_image = (-1, message.roi.height, message.roi.width)
        except AttributeError:
            offset_x = 0
            offset_y = 0
            shape_image = (-1, message.height, message.width)

        fx = message.k[0] / binning_x
        fy = message.k[4] / binning_y
        cx = (message.k[2] - offset_x) / binning_x
        cy = (message.k[5] - offset_y) / binning_y

        model_distortion = message.distortion_model
        params_distortion = cls.create_dict_params_distortion(message.d)

        instance = cls(fx, fy, cx, cy, model_distortion, params_distortion, shape_image)
        return instance

    @classmethod
    def create_dict_params_distortion(cls, list_params_distortion):
        try:
            params_distortion = dict(zip(cls.keys_params_distortion, list_params_distortion))
        except AttributeError:
            params_distortion = dict(enumerate(list_params_distortion))

        return params_distortion

    def project_points_onto_image(self, coords_xyz):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        raise NotImplementedError()

    def project_image_onto_points(self, coords_uv):
        """Project 2D image onto 3D unit sphere.
        Shape of coords_uv: (B, 2, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        raise NotImplementedError()


class ModelRationalPolynomial(ModelCamera):
    # Note: Distortion is ignored for now since they do not have a noticeable impact for the gemini.

    keys_params_distortion = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.k1 = self.params_distortion["k1"]
        # self.k2 = self.params_distortion["k2"]
        # self.k3 = self.params_distortion["k3"]
        # self.k4 = self.params_distortion["k4"]
        # self.k5 = self.params_distortion["k5"]
        # self.k6 = self.params_distortion["k6"]
        # self.p1 = self.params_distortion["p1"]
        # self.p2 = self.params_distortion["p2"]

    def project_points_onto_image(self, coords_xyz):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        coords_xyz = coords_xyz.to(self.device)

        x, y, z = coords_xyz[:, 0, :], coords_xyz[:, 1, :], coords_xyz[:, 2, :]

        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy

        coords_uv = torch.stack((u, v), axis=1)

        mask_valid = torch.ones_like(u, dtype=bool)

        return coords_uv, mask_valid

    def project_image_onto_points(self, coords_uv):
        """Project 2D image onto 3D unit sphere.
        Shape of coords_uv: (B, 2, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        coords_uv = coords_uv.to(self.device)

        u, v = coords_uv[:, 0, :], coords_uv[:, 1, :]

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        mz = torch.ones_like(mx)

        factor = 1.0 / torch.sqrt(mx**2 + my**2 + 1.0)
        coords_xyz = factor[:, None, :] * torch.stack((mx, my, mz), axis=1)

        mask_valid = torch.ones_like(mx, dtype=bool)

        return coords_xyz, mask_valid


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorrt as trt
import torch
from torch2trt import TRTModule
import torchvision.transforms.v2 as tv_transforms


from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Cache, Subscriber as SubscriberFilter
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, ReliabilityPolicy, QoSProfile
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, ParameterType
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import ros2_utils.compat.point_cloud2 as point_cloud2
from ros2_utils.parameter_handler import ParameterHandler
from ros2_utils.tf_oracle import TFOracle

import ros2_monocular_depth.transforms as transforms
import ros2_monocular_depth.config as config


class NodeMonocularDepth(Node):
    def __init__(
        self,
        topic_image="/camera_ids/image_color",
        topic_inferred_depth="/camera_ids/inferred/depth/image",
        shape_original=(1, 3, 1920, 2556),
        min_size_resized=400,
        base=14,
        name_model="depth_anything_v2_metric_hypersim_vits",
        max_depth=20,
    ):
        super().__init__(node_name="depth_anything")

        self.bridge_cv = None
        self.device = None
        self.handler_parameters = None
        self.max_depth = max_depth
        self.name_model = name_model
        self.shape_original = shape_original
        self.min_size_resized = min_size_resized
        self.base = base
        self.profile_qos = None
        self.subscriber_image = None
        self.topic_image = topic_image
        self.topic_inferred_depth = topic_inferred_depth

        self._init()

    @torch.inference_mode()
    def _init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bridge_cv = CvBridge()
        self.profile_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.handler_parameters = ParameterHandler(self, verbose=False)

        # self._init_parameters()

        # self._init_tf_oracle()
        # self._del_publishers()

        # self._del_services()
        # self._init_services()
        # self._del_subscribers()

        path_engine = Path(config._PATH_DIR_ENGINES) / f"{self.name_model}.engine"
        logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(logger)
        with open(path_engine, "rb") as file_engine:
            engine_serialized = file_engine.read()
        engine = runtime.deserialize_cuda_engine(engine_serialized)
        engine.get_tensor_shape("input")
        self.model = TRTModule(
            engine=engine,
            input_names=["input"],
            output_names=["output"],
        )
        self.model = self.model.to(self.device)

        self.transform = tv_transforms.Compose(
            [
                tv_transforms.PILToTensor(),
                transforms.ResizeToMultipleOfBase(min_size_resized=self.min_size_resized, base=self.base, mode_interpolation=transforms.MODES_INTERPOLATION.BICUBIC, use_antialiasing=False),
                tv_transforms.ToDtype(dtype=torch.float32, scale=True),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # self.i = 0
        self.stream = torch.cuda.Stream()

        self.get_logger().info(f"Initialized model")

        # self.model_camera = ModelDoubleSphere(
        #     xi=-0.2996439614293713,
        #     alpha=0.5537226081641069,
        #     fx=571.4448814063372,
        #     fy=570.6090733170088,
        #     cx=1334.1674886529015,
        #     cy=985.4219057464759,
        #     shape_image=(-1, 1920, 2556),
        # )
        # self.model_camera = ModelRationalPolynomial(
        #     571.4448814063372,
        #     570.6090733170088,
        #     1334.1674886529015,
        #     985.4219057464759,
        #     model_distortion="rational_polynomial",
        #     params_distortion={},
        #     shape_image=(-1, 1920, 2556),
        # )
        self.model_camera = ModelRationalPolynomial(
            960.4448814063372,
            960.6090733170088,
            1334.1674886529015,
            985.4219057464759,
            model_distortion="rational_polynomial",
            params_distortion={},
            shape_image=(-1, 1920, 2556),
        )

        self.factor_downsampling = 8
        u_full_downsampled = torch.arange(self.shape_original[3] // self.factor_downsampling, device=self.device) * self.factor_downsampling
        v_full_downsampled = torch.arange(self.shape_original[2] // self.factor_downsampling, device=self.device) * self.factor_downsampling
        coords_u, coords_v = torch.meshgrid(u_full_downsampled, v_full_downsampled)
        self.coords_uv = torch.stack((coords_u, coords_v), axis=0).view((1, 2, -1))
        self.coords_xyz, mask_valid = self.model_camera.project_image_onto_points(self.coords_uv)
        mask = self.coords_xyz[0, 2, :] > 0
        self.coords_xyz[0, :, mask] = torch.divide(self.coords_xyz[0, :, mask], self.coords_xyz[0, 2, mask])
        self.coords_xyz[0, :, ~mask_valid[0]] = 0.0
        self.coords_xyz[0, :, ~mask] = 0.0

        self._init_subscribers()
        self._init_publishers()

    def _init_subscribers(self):
        self.subscriber_image = self.create_subscription(Image, self.topic_image, self.infer_depth, qos_profile=self.profile_qos, callback_group=MutuallyExclusiveCallbackGroup())

    def _init_publishers(self):
        self.publisher_depth = self.create_publisher(msg_type=Image, topic=self.topic_inferred_depth, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())
        self.topic_projected_points = "/camera_ids/projected/points"
        self.publisher_points = self.create_publisher(msg_type=PointCloud2, topic=self.topic_projected_points, qos_profile=self.profile_qos, callback_group=ReentrantCallbackGroup())

    def publish_image(self, image, name_frame, stamp):
        header = Header(stamp=stamp, frame_id=name_frame)
        message = self.bridge_cv.cv2_to_imgmsg(image, header=header, encoding="mono16")

        self.publisher_depth.publish(message)

    def publish_points(self, pointcloud, name_frame, stamp):
        header = Header(stamp=stamp, frame_id=name_frame)
        message = point_cloud2.create_cloud_xyz32(header, pointcloud)

        self.publisher_points.publish(message)

    def publish_points_colored(self, pointcloud_colored, name_frame, stamp):
        header = Header(stamp=stamp, frame_id=name_frame)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        message = point_cloud2.create_cloud(header, fields, pointcloud_colored)

        self.publisher_points.publish(message)

    def visualize_depth_image(self, image_depth, image_rgb, time_needed):
        din_a4 = np.array([210, 297]) / 25.4
        din_a4_landscape = din_a4[::-1]
        fig = plt.figure(figsize=din_a4_landscape)

        def visualize_image(image):
            ax = plt.gca()
            ax.set_axis_off()
            ax.imshow(image, cmap="turbo" if image.shape[2] == 1 else None, vmin=0)

        fig.add_subplot(1, 2, 1)
        visualize_image(image_rgb)

        fig.add_subplot(1, 2, 2)
        visualize_image(image_depth)

        plt.tight_layout()
        plt.title(f"Inference time: {time_needed:.4f}, Hz {1 / time_needed}")
        # plt.savefig(Path(".") / "gif" / f"image_{self.i}")
        # self.i += 1
        plt.show()

    def colors_to_numpy(self, colors):
        colors = colors.detach().cpu().numpy().astype(np.uint32)

        r = colors[0]
        g = colors[1]
        b = colors[2]
        colors = (r << 16) | (g << 8) | b

        return colors

    @torch.inference_mode()
    def rectify_image(self, image, f=0.5):
        h, w = (406, 532)
        z = f * min(h, w)
        x = torch.arange(w) - w / 2
        y = torch.arange(h) - h / 2
        x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")
        point3D = torch.stack([x_grid, y_grid, torch.full_like(x_grid, z)], axis=0)
        point3D = point3D.reshape((3, -1))[None, ...]
        point3D /= 10

        model_camera = ModelDoubleSphere(
            xi=-0.2996439614293713,
            alpha=0.5537226081641069,
            fx=571.4448814063372,
            fy=570.6090733170088,
            cx=1334.1674886529015,
            cy=985.4219057464759,
            shape_image=(-1, 1920, 2556),
        )
        coords_uv, mask_valid = model_camera.project_points_onto_image(point3D)
        coords_uv[:, 0, :] = 2.0 * coords_uv[:, 0, :] / 2556 - 1.0
        coords_uv[:, 1, :] = 2.0 * coords_uv[:, 1, :] / 1920 - 1.0
        coords_uv = coords_uv.permute(0, 2, 1)

        # grid_sample not implemented for dtype byte
        out = torch.nn.functional.grid_sample(input=image, grid=coords_uv[:, None, :, :].float(), align_corners=True)
        out[:, 0, :].masked_fill_(~mask_valid, 0)
        out[:, 1, :].masked_fill_(~mask_valid, 0)
        out[:, 2, :].masked_fill_(~mask_valid, 0)

        out = out.reshape((1, 3, h, w))

        # coords_uv = coords_uv.int()
        # out = torch.zeros((1, 3, h, w), device=self.device, dtype=torch.uint8)
        # out[0, :, coords_uv[0, 1, :], coords_uv[0, 0, :]] = image[:, coords_uv[0, 1, :], coords_uv[0, 0, :]]

        return out

    @torch.inference_mode()
    def infer_depth(self, message_image):
        # s = time.time()
        with torch.cuda.stream(self.stream):
            image = self.bridge_cv.imgmsg_to_cv2(message_image, desired_encoding="passthrough")
            image = torch.from_numpy(image)
            image = image.to(self.device)
            image = image.permute((2, 0, 1))

            input = image[None, ...]
            input = self.transform(input)

            input = self.rectify_image(input)
            # plt.imshow(input[0].permute((1, 2, 0)).cpu().numpy())
            # plt.show()

            output = self.model(input)
            output = output[:, None, :, :]
            # output = torch.nn.functional.interpolate(output, (self.shape_original[2], self.shape_original[3]), mode="bilinear", align_corners=True)
            output = torch.nn.functional.interpolate(output, (self.shape_original[2] // self.factor_downsampling, self.shape_original[3] // self.factor_downsampling), mode="bilinear", align_corners=True)
            output = output[0]

            # self.visualize_depth_image(output.permute((1, 2, 0)).cpu().numpy(), image.permute((1, 2, 0)).cpu().numpy(), 0.1)

            image_depth = output * 1000
            image_depth = image_depth.permute((1, 2, 0)).detach().cpu().numpy().astype(np.uint16)
            self.publish_image(image_depth, name_frame="gemini2_depth_optical_frame", stamp=message_image.header.stamp)

            output = output.permute(0, 2, 1)
            mask = torch.isfinite(output) * output > 0
            pointcloud = self.coords_xyz.view(3, -1) * output[mask]
            pointcloud = pointcloud.permute((1, 0)).detach().cpu().numpy().astype(np.float32)
            # self.publish_points(pointcloud, name_frame="gemini2_depth_optical_frame", stamp=message_image.header.stamp)

            colors = tv_transforms.functional.resize(image, size=[self.shape_original[2] // self.factor_downsampling, self.shape_original[3] // self.factor_downsampling]).permute(0, 2, 1).reshape(3, -1)
            # plt.imshow(tv_transforms.functional.resize(image, size=[self.shape_original[2] // self.factor_downsampling, self.shape_original[3] // self.factor_downsampling]).permute((1, 2, 0)).cpu().numpy())
            colors = self.colors_to_numpy(colors)
            dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.uint32)])
            pointcloud_colored = np.zeros(pointcloud.shape[0], dtype=dtype)
            # pointcloud_colored["x"] = self.coords_xyz[0, 0].cpu().numpy()
            # pointcloud_colored["y"] = self.coords_xyz[0, 1].cpu().numpy()
            # pointcloud_colored["z"] = self.coords_xyz[0, 2].cpu().numpy()
            # pointcloud_colored["x"] = pointcloud[:, 0]
            # pointcloud_colored["y"] = pointcloud[:, 1]
            # pointcloud_colored["z"] = pointcloud[:, 2]
            pointcloud_colored["x"] = pointcloud[:, 0]
            pointcloud_colored["y"] = pointcloud[:, 2]
            pointcloud_colored["z"] = -pointcloud[:, 1]
            pointcloud_colored["rgb"] = colors

            self.publish_points_colored(pointcloud_colored, name_frame="gemini2_depth_optical_frame", stamp=message_image.header.stamp)

        # e = time.time()
        # time_needed = e - s
