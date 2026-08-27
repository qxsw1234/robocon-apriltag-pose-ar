"""ar_object_node - 拓展题 1.2: 以 AprilTag 坐标系为世界系的固定虚拟三维物体 AR 显示.

订阅 (去畸变后的)图像 + CameraInfo + AprilTag 检测, 对目标标签:
    1. solvePnP 求 tag->相机 的 (rvec, tvec)   [复用 pose_math, 与主可视化节点同源]
    2. 在 **tag 坐标系** 中定义一个固定的虚拟三维物体 (默认立方体,
       沿标签法线 +z 偏移, 不与标签重合)
    3. cv2.projectPoints 把物体顶点投到图像, 背面剔除 + 画家算法排序后
       半透明填充并描边  -> /apriltag/image_ar
    4. 同时在 **tag_N frame** 里发一个等价的 RViz Marker -> /apriltag/ar_marker
       因为 marker 挂在 tag frame 上, 它在 RViz 里会随 TF 自动跟随,
       这本身就是"变换关系是否正确"的一个独立交叉验证。

正确性验证 (对应题目"可自行设法验证这个变换关系是否正确"):
    * 图像侧: 相机移动/旋转时物体近大远小、朝向随视角变化, 但始终"钉"在
      标签的同一位置。
    * 数学侧: 物体顶点在 tag 系中的坐标恒定不变 —— 节点启动时会
      打印 verify_static_in_world 的自检结果 (tag 系变化量应为 0)。
    * RViz 侧: 物体与 tag_N 坐标系的相对位置恒定。
    * 单元测试: test/test_ar_geometry.py (24 项)。

参数化 (不硬编码):
    object_type   cube | pyramid | arrow
    object_size_m 物体尺寸(米); <=0 时取 object_size_ratio * tag_size
    offset_xyz_m  物体在 tag 系中的偏移 [x,y,z]; 默认沿法线抬升
"""

from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

from apriltag_pose.ar_geometry import (
    build_object, draw_mesh, draw_offset_indicator, verify_static_in_world,
)
from apriltag_pose.pose_filter import PoseStabilizer
from apriltag_pose.pose_math import (
    camera_matrix_from_camera_info, is_camera_info_valid,
)


def _best_effort_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, depth=depth,
                      durability=DurabilityPolicy.VOLATILE)


def _reliable_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      history=HistoryPolicy.KEEP_LAST, depth=depth,
                      durability=DurabilityPolicy.VOLATILE)


class ArObjectNode(Node):
    def __init__(self) -> None:
        super().__init__('ar_object_node')

        # ---- parameters
        self.declare_parameter('image_topic', '/camera/image_rect')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('detections_topic', '/apriltag/detections')
        self.declare_parameter('output_image_topic', '/apriltag/image_ar')
        self.declare_parameter('marker_topic', '/apriltag/ar_marker')
        self.declare_parameter('tag_size', 0.080)
        self.declare_parameter('target_tag_id', 0)
        self.declare_parameter('tag_frame', 'tag_0')
        self.declare_parameter('use_rectified_image', True)
        # 物体定义
        self.declare_parameter('object_type', 'cube')
        self.declare_parameter('object_size_m', 0.0)      # <=0 -> 用 ratio
        self.declare_parameter('object_size_ratio', 0.5)  # 相对 tag_size
        self.declare_parameter('offset_xyz_m', [0.0, 0.06, 0.04])
        # 渲染
        self.declare_parameter('fill_alpha', 0.45)
        self.declare_parameter('edge_thickness', 2)
        self.declare_parameter('cull_backfaces', True)
        self.declare_parameter('show_offset_indicator', True)
        self.declare_parameter('draw_tag_axes', True)
        # 稳定性: 滑动窗口滤波 + 异常帧剔除 + PnP 二义性择优 + 丢检保持
        self.declare_parameter('filter_window', 5)
        self.declare_parameter('use_median_translation', True)
        self.declare_parameter('pose_hold_sec', 0.25)
        self.declare_parameter('max_reproj_error_px', 4.0)
        self.declare_parameter('min_decision_margin', 20.0)
        self.declare_parameter('max_jump_m', 0.25)
        self.declare_parameter('max_jump_deg', 60.0)
        self.declare_parameter('sync_slop_sec', 0.15)
        self.declare_parameter('sync_queue_size', 20)

        g = self.get_parameter
        self._image_topic = str(g('image_topic').value)
        self._info_topic = str(g('camera_info_topic').value)
        self._det_topic = str(g('detections_topic').value)
        self._out_topic = str(g('output_image_topic').value)
        self._marker_topic = str(g('marker_topic').value)
        self._tag_size = float(g('tag_size').value)
        self._target_id = int(g('target_tag_id').value)
        self._tag_frame = str(g('tag_frame').value)
        self._use_rect = bool(g('use_rectified_image').value)
        self._object_type = str(g('object_type').value)
        self._fill_alpha = float(g('fill_alpha').value)
        self._edge_thickness = int(g('edge_thickness').value)
        self._cull = bool(g('cull_backfaces').value)
        self._show_offset = bool(g('show_offset_indicator').value)
        self._draw_axes = bool(g('draw_tag_axes').value)
        self._hold_sec = float(g('pose_hold_sec').value)
        self._max_reproj = float(g('max_reproj_error_px').value)
        self._min_margin = float(g('min_decision_margin').value)
        slop = float(g('sync_slop_sec').value)
        qsize = int(g('sync_queue_size').value)

        if self._tag_size <= 0.0:
            raise ValueError(f'tag_size must be > 0, got {self._tag_size}')

        size = float(g('object_size_m').value)
        if size <= 0.0:
            size = self._tag_size * float(g('object_size_ratio').value)
        self._object_size = size

        off = [float(v) for v in g('offset_xyz_m').value]
        if len(off) != 3:
            raise ValueError(f'offset_xyz_m needs 3 values, got {off}')
        self._offset = off

        # 物体只构造一次 —— 它在世界(tag)系中是固定的, 不随帧变化.
        # 这正是"物体相对 AprilTag 世界系保持静止"在代码上的体现。
        self._mesh = build_object(self._object_type, self._object_size, self._offset)

        # ---- state
        self._bridge = CvBridge()
        self._latest_info: Optional[CameraInfo] = None
        self._warned_info = False
        self._warned_invalid = False
        self._frames_drawn = 0
        self._frames_held = 0

        # 位姿稳定器: 异常帧剔除 -> 平面 PnP 二义性择优 -> 滑动窗口滤波
        # -> 丢检保持. 详见 pose_filter.py。
        self._stabilizer = PoseStabilizer(
            tag_size_m=self._tag_size,
            window=int(g('filter_window').value),
            use_median_translation=bool(g('use_median_translation').value),
            min_decision_margin=self._min_margin,
            max_reproj_error_px=self._max_reproj,
            max_jump_m=float(g('max_jump_m').value),
            max_jump_deg=float(g('max_jump_deg').value),
            hold_sec=self._hold_sec,
        )

        # ---- pubs / subs
        sub_qos, pub_qos = _best_effort_qos(), _reliable_qos()
        self._pub_img = self.create_publisher(Image, self._out_topic, pub_qos)
        self._pub_marker = self.create_publisher(MarkerArray, self._marker_topic,
                                                 pub_qos)
        self.create_subscription(CameraInfo, self._info_topic,
                                 self._on_info, sub_qos)
        self._sub_img = Subscriber(self, Image, self._image_topic,
                                   qos_profile=sub_qos)
        self._sub_det = Subscriber(self, AprilTagDetectionArray, self._det_topic,
                                   qos_profile=sub_qos)
        self._sync = ApproximateTimeSynchronizer(
            [self._sub_img, self._sub_det], queue_size=qsize, slop=slop)
        self._sync.registerCallback(self._on_pair)

        self._log_startup_selfcheck()

    # ------------------------------------------------------------------
    def _log_startup_selfcheck(self) -> None:
        """启动时验证"物体相对 tag 世界系静止", 把结论打到日志里."""
        poses = [
            (np.array([[np.pi], [0.0], [0.0]]), np.array([[0.0], [0.0], [0.50]])),
            (np.array([[np.pi], [0.4], [0.0]]), np.array([[0.05], [0.0], [0.70]])),
            (np.array([[2.8], [0.0], [0.5]]), np.array([[-0.04], [0.03], [0.40]])),
        ]
        chk = verify_static_in_world(self._mesh, poses)
        self.get_logger().info(
            f'ar_object_node up:\n'
            f'  image:        {self._image_topic} (rectified={self._use_rect})\n'
            f'  detections:   {self._det_topic}\n'
            f'  outputs:      {self._out_topic}, {self._marker_topic}\n'
            f'  target tag:   id={self._target_id}  frame={self._tag_frame}\n'
            f'  tag_size:     {self._tag_size:.4f} m\n'
            f'  object:       {self._object_type}  size={self._object_size:.4f} m\n'
            f'  offset(tag):  [{self._offset[0]:+.3f}, {self._offset[1]:+.3f}, '
            f'{self._offset[2]:+.3f}] m\n'
            f'  self-check "static in tag/world frame":\n'
            f'    max delta in TAG frame    = '
            f'{chk["max_tag_frame_delta_m"]:.9f} m  (must be 0)\n'
            f'    max delta in CAMERA frame = '
            f'{chk["max_camera_frame_delta_m"]:.6f} m  (must be > 0)')
        if chk['max_tag_frame_delta_m'] != 0.0:
            self.get_logger().error(
                'self-check FAILED: object is not static in the tag frame!')

    # ------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        self._latest_info = msg

    # ------------------------------------------------------------------
    def _on_pair(self, img_msg: Image, det_msg: AprilTagDetectionArray) -> None:
        if self._latest_info is None:
            if not self._warned_info:
                self.get_logger().warn(f'waiting for CameraInfo on '
                                       f'{self._info_topic} ...')
                self._warned_info = True
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        info = self._latest_info
        try:
            if self._use_rect:
                cam_k = camera_matrix_from_camera_info(info.p,
                                                       use_projection_matrix=True)
                dist = np.zeros((5,), dtype=np.float64)
            else:
                cam_k = camera_matrix_from_camera_info(info.k,
                                                       use_projection_matrix=False)
                dist = np.asarray(info.d, dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'bad CameraInfo: {exc}')
            return

        if not is_camera_info_valid(cam_k):
            if not self._warned_invalid:
                self.get_logger().warn(
                    'CameraInfo has zero fx/fy/cx/cy -- camera not calibrated. '
                    'AR rendering needs intrinsics; publishing the raw image.')
                self._warned_invalid = True
            self._publish(frame, img_msg)
            return

        # 找到目标标签的角点 (没找到时传 None, 由稳定器走丢检保持)
        corners, margin = self._find_target_corners(det_msg)
        now = time.monotonic()
        filtered = self._stabilizer.update(corners, margin, cam_k, dist, now)

        if filtered is not None:
            rvec, tvec = filtered.rvec, filtered.tvec
            if self._draw_axes:
                cv2.drawFrameAxes(frame, cam_k, dist, rvec, tvec,
                                  self._tag_size * 0.5, 2)
            if self._show_offset:
                draw_offset_indicator(frame, self._offset, rvec, tvec,
                                      cam_k, dist)
            drawn = draw_mesh(frame, self._mesh, rvec, tvec, cam_k, dist,
                              fill_alpha=self._fill_alpha,
                              edge_thickness=self._edge_thickness,
                              cull_backfaces=self._cull)
            if drawn:
                self._frames_drawn += 1
                if filtered.held:
                    self._frames_held += 1
                self._annotate(frame, filtered)
            self._publish_marker(img_msg)

        self._publish(frame, img_msg)

    # ------------------------------------------------------------------
    def _find_target_corners(self, det_msg: AprilTagDetectionArray):
        """返回目标标签的 (corners(4,2), decision_margin).

        找不到目标标签时返回 (None, None) —— 交给稳定器决定是保持还是放弃。
        质量筛选(margin / 重投影误差 / 二义性)统一在 PoseStabilizer 里做,
        避免同一套阈值散落在两处。
        """
        for det in det_msg.detections:
            if self._target_id >= 0 and det.id != self._target_id:
                continue
            corners = np.array([[c.x, c.y] for c in det.corners],
                               dtype=np.float64)
            if corners.shape != (4, 2):
                continue
            return corners, float(det.decision_margin)
        return None, None

    # ------------------------------------------------------------------
    def _annotate(self, frame: np.ndarray, filtered) -> None:
        lines = [
            f'AR: {self._object_type} size={self._object_size * 1000:.0f}mm',
            f'offset in TAG frame = [{self._offset[0] * 1000:+.0f}, '
            f'{self._offset[1] * 1000:+.0f}, {self._offset[2] * 1000:+.0f}] mm',
            'object is STATIC in the AprilTag world frame',
            f'filter: {filtered.samples} samples, {filtered.rejected} rejected',
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 46 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                        cv2.LINE_AA)
        if filtered.held:
            cv2.putText(frame, 'POSE HELD (detection dropped)',
                        (10, 46 + len(lines) * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
                        cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _publish_marker(self, img_msg: Image) -> None:
        """在 tag frame 中发布等价的 RViz marker.

        frame_id = tag_N, 所以 RViz 会用 TF 把它摆到正确位置 ——
        物体相对 tag 静止这件事由 TF 自动体现, 无需我们每帧重算。
        """
        arr = MarkerArray()
        m = Marker()
        m.header.stamp = img_msg.header.stamp
        m.header.frame_id = self._tag_frame
        m.ns = 'ar_object'
        m.id = 0
        m.action = Marker.ADD
        m.pose.position.x = self._offset[0]
        m.pose.position.y = self._offset[1]
        m.pose.position.z = self._offset[2]
        m.pose.orientation.w = 1.0
        m.color.a = 0.65
        if self._object_type == 'cube':
            m.type = Marker.CUBE
            m.scale.x = m.scale.y = m.scale.z = self._object_size
            m.color.r, m.color.g, m.color.b = 0.1, 0.8, 0.3
        elif self._object_type == 'arrow':
            m.type = Marker.ARROW
            # ARROW 用起点/终点表示, 沿 tag +z
            m.pose.position.x = m.pose.position.y = m.pose.position.z = 0.0
            m.points = [
                Point(x=self._offset[0], y=self._offset[1], z=self._offset[2]),
                Point(x=self._offset[0], y=self._offset[1],
                      z=self._offset[2] + self._object_size),
            ]
            m.scale.x = self._object_size * 0.12      # shaft diameter
            m.scale.y = self._object_size * 0.24      # head diameter
            m.scale.z = self._object_size * 0.3       # head length
            m.color.r, m.color.g, m.color.b = 0.9, 0.4, 0.1
        else:
            # pyramid 在 RViz 没有原生类型, 用 LINE_LIST 画棱
            m.type = Marker.LINE_LIST
            m.scale.x = 0.002
            m.color.r, m.color.g, m.color.b = 0.2, 0.7, 0.9
            m.pose.position.x = m.pose.position.y = m.pose.position.z = 0.0
            v = self._mesh.vertices
            pts: List[Point] = []
            for a, b in self._mesh.edges:
                pts.append(Point(x=float(v[a][0]), y=float(v[a][1]),
                                 z=float(v[a][2])))
                pts.append(Point(x=float(v[b][0]), y=float(v[b][1]),
                                 z=float(v[b][2])))
            m.points = pts
        m.lifetime.nanosec = 300_000_000
        arr.markers.append(m)
        self._pub_marker.publish(arr)

    # ------------------------------------------------------------------
    def _publish(self, frame: np.ndarray, img_msg: Image) -> None:
        try:
            out = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            out.header = img_msg.header
            self._pub_img.publish(out)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'cv_bridge publish failed: {exc}')


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = ArObjectNode()
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
