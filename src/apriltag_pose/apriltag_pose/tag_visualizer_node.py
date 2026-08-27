"""tag_visualizer_node.

Subscribes to a (rectified) camera image + CameraInfo + AprilTag detections,
runs OpenCV solvePnP to compute each tag's rvec/tvec explicitly, and draws:

    1. the 4 edges of the tag
    2. the 4 corners numbered 0..3 (apriltag order: BL, BR, TR, TL)
    3. the tag centre (cross)
    4. the tag ID + family
    5. the decision margin
    6. tx, ty, tz (meters) and z-depth
    7. euclidean distance = sqrt(tx^2 + ty^2 + tz^2)
    8. a 3-D coordinate axis (cv2.drawFrameAxes), length = 0.5 * tag_size
    9. optional FPS

Publishes:
    /apriltag/image_annotated  (sensor_msgs/Image, bgr8)
    /apriltag/pose             (geometry_msgs/PoseArray, frame = optical frame)
    /apriltag/distance         (std_msgs/Float64, norm distance of target tag)
    /apriltag/markers          (visualization_msgs/MarkerArray, for RViz)

QoS rule used throughout: subscribers = BEST_EFFORT (receives from any
publisher reliability), publishers = RELIABLE (matches any subscriber).
This avoids the common Humble "subscribed but no data" QoS mismatch.

Note: apriltag_ros (>= 3.4.0) already publishes camera_optical_frame -> tag_N
TF when pose_estimation_method is set (see config/apriltag.yaml). This node
does NOT re-broadcast that TF (would double-publish); it only computes its
own pose for drawing / pose / distance / markers, which doubles as an
independent cross-check of apriltag_ros's TF.

Deliberately UNFILTERED: this node is the *measurement* path. The distance-error
experiment (task 11) reads /apriltag/distance, so smoothing here would hide the
system's true per-frame accuracy and make the reported error look better than it
really is. Jitter suppression lives in ar_object_node (the *display* path) via
pose_filter.PoseStabilizer.
"""

from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray

from apriltag_pose.pose_math import (
    camera_matrix_from_camera_info, euclidean_distance, is_camera_info_valid,
    meters_to_mm, rvec_to_quaternion, solve_tag_pose,
)


def _best_effort_qos(depth: int = 5) -> QoSProfile:
    """Subscriber QoS that receives from any publisher reliability."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.VOLATILE,
    )


def _reliable_qos(depth: int = 5) -> QoSProfile:
    """Publisher QoS that matches any subscriber reliability."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.VOLATILE,
    )


class TagVisualizerNode(Node):
    def __init__(self) -> None:
        super().__init__('tag_visualizer_node')

        # ---- parameters
        self.declare_parameter('image_topic', '/camera/image_rect')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('detections_topic', '/apriltag/detections')
        self.declare_parameter('output_image_topic', '/apriltag/image_annotated')
        self.declare_parameter('pose_topic', '/apriltag/pose')
        self.declare_parameter('distance_topic', '/apriltag/distance')
        self.declare_parameter('markers_topic', '/apriltag/markers')
        self.declare_parameter('tag_size', 0.080)
        self.declare_parameter('axis_length', 0.040)        # ~0.5 * tag_size
        self.declare_parameter('use_rectified_image', True)
        self.declare_parameter('show_fps', True)
        self.declare_parameter('log_hz', 1.0)               # throttled terminal log
        self.declare_parameter('target_tag_id', -1)         # -1 = first detected
        self.declare_parameter('optical_frame', 'camera_optical_frame')
        self.declare_parameter('sync_slop_sec', 0.15)
        self.declare_parameter('sync_queue_size', 20)

        self._image_topic = self.get_parameter('image_topic').value
        self._info_topic = self.get_parameter('camera_info_topic').value
        self._det_topic = self.get_parameter('detections_topic').value
        self._out_topic = self.get_parameter('output_image_topic').value
        self._pose_topic = self.get_parameter('pose_topic').value
        self._dist_topic = self.get_parameter('distance_topic').value
        self._markers_topic = self.get_parameter('markers_topic').value
        self._tag_size = float(self.get_parameter('tag_size').value)
        self._axis_len = float(self.get_parameter('axis_length').value)
        self._use_rect = bool(self.get_parameter('use_rectified_image').value)
        self._show_fps = bool(self.get_parameter('show_fps').value)
        self._log_dt = 1.0 / max(1e-3, float(self.get_parameter('log_hz').value))
        self._target_id = int(self.get_parameter('target_tag_id').value)
        self._optical_frame = str(self.get_parameter('optical_frame').value)
        slop = float(self.get_parameter('sync_slop_sec').value)
        qsize = int(self.get_parameter('sync_queue_size').value)

        if self._tag_size <= 0.0:
            raise ValueError(f'tag_size must be > 0, got {self._tag_size}')

        # ---- state
        self._bridge = CvBridge()
        self._latest_info: Optional[CameraInfo] = None
        self._warned_missing_info = False
        self._warned_invalid_info = False
        self._last_log_t = 0.0
        # FPS (wall-clock rolling window)
        self._frame_times: List[float] = []
        # marker id counter (markers must have unique ns+id per stamp)
        self._marker_seq = 0

        # ---- pubs & subs
        sub_qos = _best_effort_qos()
        pub_qos = _reliable_qos()
        self._pub_annotated = self.create_publisher(Image, self._out_topic, pub_qos)
        self._pub_pose = self.create_publisher(PoseArray, self._pose_topic, pub_qos)
        self._pub_distance = self.create_publisher(Float64, self._dist_topic, pub_qos)
        self._pub_markers = self.create_publisher(MarkerArray, self._markers_topic, pub_qos)

        # CameraInfo arrives independently; latch the latest one.
        self.create_subscription(CameraInfo, self._info_topic,
                                 self._on_camera_info, sub_qos)

        # Image + detections are synchronized on header.stamp.
        self._sub_img = Subscriber(self, Image, self._image_topic, qos_profile=sub_qos)
        self._sub_det = Subscriber(self, AprilTagDetectionArray, self._det_topic,
                                   qos_profile=sub_qos)
        self._sync = ApproximateTimeSynchronizer(
            [self._sub_img, self._sub_det], queue_size=qsize, slop=slop)
        self._sync.registerCallback(self._on_pair)

        self.get_logger().info(
            f'tag_visualizer_node up:\n'
            f'  image:       {self._image_topic}   (rectified={self._use_rect})\n'
            f'  camera_info: {self._info_topic}\n'
            f'  detections:  {self._det_topic}\n'
            f'  outputs:     {self._out_topic}, {self._pose_topic}, '
            f'{self._dist_topic}, {self._markers_topic}\n'
            f'  tag_size:    {self._tag_size:.4f} m\n'
            f'  axis_length: {self._axis_len:.4f} m\n'
            f'  optical_frame: {self._optical_frame}'
        )

    # ------------------------------------------------------------------
    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._latest_info = msg

    # ------------------------------------------------------------------
    def _on_pair(self, img_msg: Image, det_msg: AprilTagDetectionArray) -> None:
        if self._latest_info is None:
            if not self._warned_missing_info:
                self.get_logger().warn('waiting for CameraInfo on '
                                       f'{self._info_topic} ...')
                self._warned_missing_info = True
            return

        # FPS bookkeeping (wall clock).
        now_w = time.monotonic()
        self._frame_times.append(now_w)
        # keep last ~1 s
        while self._frame_times and now_w - self._frame_times[0] > 1.0:
            self._frame_times.pop(0)
        fps = len(self._frame_times)

        try:
            frame = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        info = self._latest_info
        try:
            if self._use_rect:
                # Rectified image: use P[:3,:3] as the new K, zero distortion.
                camera_matrix = camera_matrix_from_camera_info(
                    info.p, use_projection_matrix=True)
                dist = np.zeros((5,), dtype=np.float64)
            else:
                camera_matrix = camera_matrix_from_camera_info(
                    info.k, use_projection_matrix=False)
                dist = np.asarray(info.d, dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'bad CameraInfo: {exc}')
            return

        pose_valid = is_camera_info_valid(camera_matrix)
        if not pose_valid:
            if not self._warned_invalid_info:
                self.get_logger().warn(
                    'CameraInfo has zero fx/fy/cx/cy -- calibration not loaded? '
                    'Drawing 2D annotations only, no 3D pose.')
                self._warned_invalid_info = True

        pose_array = PoseArray()
        pose_array.header = img_msg.header
        pose_array.header.frame_id = self._optical_frame
        markers = MarkerArray()
        target_distance: Optional[float] = None

        for det in det_msg.detections:
            pose_result = self._draw_one_detection(
                frame, det, camera_matrix if pose_valid else None,
                dist if pose_valid else None)

            if pose_result is not None:
                # PoseArray entry: position = tvec, orientation = rvec->quat.
                p = Pose()
                p.position.x = float(pose_result.tvec[0])
                p.position.y = float(pose_result.tvec[1])
                p.position.z = float(pose_result.tvec[2])
                qx, qy, qz, qw = rvec_to_quaternion(pose_result.rvec)
                p.orientation.x = qx
                p.orientation.y = qy
                p.orientation.z = qz
                p.orientation.w = qw
                pose_array.poses.append(p)

                dist_m = euclidean_distance(pose_result.tvec.reshape(-1))
                # Pick the distance to publish: target id if set, else first.
                is_target = (self._target_id < 0) or (det.id == self._target_id)
                if target_distance is None and is_target:
                    target_distance = dist_m

                # RViz markers: text (ID + distance) and a line camera->tag.
                self._append_markers(markers, img_msg.header, det.id,
                                     pose_result.tvec, dist_m)

        # Publish annotated image.
        try:
            if self._show_fps:
                cv2.putText(frame, f'FPS={fps}', (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),
                            2, cv2.LINE_AA)
            out = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            out.header = img_msg.header
            self._pub_annotated.publish(out)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'cv_bridge publish failed: {exc}')

        # Publish pose array + distance (only when we have detections).
        if pose_array.poses:
            self._pub_pose.publish(pose_array)
        if target_distance is not None:
            msg = Float64()
            msg.data = target_distance
            self._pub_distance.publish(msg)
        if markers.markers:
            self._pub_markers.publish(markers)

        # 1 Hz throttled terminal log.
        if pose_array.poses and (now_w - self._last_log_t) >= self._log_dt:
            self._last_log_t = now_w
            self._log_terminal(det_msg, pose_array)

    # ------------------------------------------------------------------
    def _draw_one_detection(self, frame: np.ndarray, det,
                            camera_matrix: Optional[np.ndarray],
                            dist_coeffs: Optional[np.ndarray]):
        """Draw one tag. Returns PoseResult or None."""
        corners = np.array([[c.x, c.y] for c in det.corners], dtype=np.float64)
        if corners.shape != (4, 2):
            return None
        centre = (float(det.centre.x), float(det.centre.y))

        color_edge = (0, 255, 0)
        color_corner = (0, 200, 255)
        color_id = (255, 255, 0)
        color_axis = (255, 255, 255)

        # 1. Edges.
        pts = corners.astype(int)
        for i in range(4):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]),
                     color_edge, 2, cv2.LINE_AA)

        # 2-3. Corners + numbers (apriltag order: 0=BL, 1=BR, 2=TR, 3=TL).
        for i, (x, y) in enumerate(pts):
            cv2.circle(frame, (int(x), int(y)), 5, color_corner, -1, cv2.LINE_AA)
            cv2.putText(frame, str(i), (int(x) + 7, int(y) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_corner, 1, cv2.LINE_AA)

        # 4. Centre (cross).
        cx_i, cy_i = int(centre[0]), int(centre[1])
        cv2.drawMarker(frame, (cx_i, cy_i), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        # 5-7. ID, family, decision margin.
        label = f'ID={det.id} ({det.family})'
        cv2.putText(frame, label, (cx_i + 10, cy_i + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_id, 2, cv2.LINE_AA)
        margin_line = f'margin={float(det.decision_margin):.1f}'
        cv2.putText(frame, margin_line, (cx_i + 10, cy_i + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if camera_matrix is None:
            return None

        # 9-10. solvePnP -> rvec, tvec, draw axes.
        pose = solve_tag_pose(
            corners_apriltag_xy=corners,
            tag_size_m=self._tag_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        if pose is None:
            cv2.putText(frame, 'PnP FAILED', (cx_i + 10, cy_i + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            return None

        cv2.drawFrameAxes(frame, camera_matrix,
                          dist_coeffs if dist_coeffs is not None else np.zeros(5),
                          pose.rvec, pose.tvec, self._axis_len, 2)

        # 8. tvec + distance text (stacked, no overlap).
        tx, ty, tz = float(pose.tvec[0]), float(pose.tvec[1]), float(pose.tvec[2])
        dist_m = euclidean_distance(pose.tvec.reshape(-1))
        lines = [
            f'tx={meters_to_mm(tx):+7.1f} mm',
            f'ty={meters_to_mm(ty):+7.1f} mm',
            f'tz={meters_to_mm(tz):+7.1f} mm  (depth)',
            f'd ={meters_to_mm(dist_m):7.1f} mm  ({pose.method}, '
            f'reproj={pose.reprojection_error_px:.2f}px)',
        ]
        y0 = cy_i + 62
        for k, line in enumerate(lines):
            cv2.putText(frame, line, (cx_i + 10, y0 + k * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_axis, 1, cv2.LINE_AA)
        return pose

    # ------------------------------------------------------------------
    def _append_markers(self, markers: MarkerArray, header, tag_id,
                        tvec: np.ndarray, dist_m: float) -> None:
        """Text marker (ID + distance) and a line camera->tag, in optical frame."""
        ns = f'tag_{tag_id}'
        # Text marker.
        text = Marker()
        text.header = header
        text.header.frame_id = self._optical_frame
        text.ns = ns
        text.id = 0
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.scale.z = self._tag_size * 0.4
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 0.2
        text.color.a = 1.0
        text.text = f'ID={tag_id} d={dist_m:.3f}m'
        text.pose.position.x = float(tvec[0])
        text.pose.position.y = float(tvec[1])
        text.pose.position.z = float(tvec[2])
        text.lifetime.sec = 0
        text.lifetime.nanosec = 200_000_000
        markers.markers.append(text)

        # Line from camera origin (0,0,0) to tag centre.
        line = Marker()
        line.header = header
        line.header.frame_id = self._optical_frame
        line.ns = ns
        line.id = 1
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.scale.x = 0.003                      # line width (m)
        line.color.r = 0.2
        line.color.g = 1.0
        line.color.b = 0.2
        line.color.a = 0.8
        from geometry_msgs.msg import Point
        a = Point(x=0.0, y=0.0, z=0.0)
        b = Point(x=float(tvec[0]), y=float(tvec[1]), z=float(tvec[2]))
        line.points = [a, b]
        line.lifetime.nanosec = 200_000_000
        markers.markers.append(line)

    # ------------------------------------------------------------------
    def _log_terminal(self, det_msg: AprilTagDetectionArray,
                      pose_array: PoseArray) -> None:
        for det, pose in zip(det_msg.detections, pose_array.poses):
            tx, ty, tz = pose.position.x, pose.position.y, pose.position.z
            d = euclidean_distance((tx, ty, tz))
            self.get_logger().info(
                f'Tag ID={det.id} ({det.family})  '
                f'tvec=[{tx:+.4f}, {ty:+.4f}, {tz:+.4f}] m  '
                f'distance={d:.4f} m  margin={float(det.decision_margin):.1f}')


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = TagVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
