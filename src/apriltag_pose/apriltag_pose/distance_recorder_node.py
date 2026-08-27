"""distance_recorder_node.

Interactive distance-error data collector.

It subscribes to AprilTag detections + CameraInfo, computes the tag pose with
the SAME solve_tag_pose() used by the visualizer (so we get rvec, tvec, the
PnP method and the decision margin for free), and -- when a batch is running
-- appends one row per accepted sample to a CSV.

Workflow (terminal-driven, no custom messages/services needed):

    # 1. place the tag at a measured distance, e.g. 0.30 m, face-on
    ros2 param set /distance_recorder_node true_distance_m 0.30
    ros2 param set /distance_recorder_node sample_group front
    ros2 service call /distance_recorder_node/start_batch std_srvs/srv/Trigger

    # 2. wait for "batch complete" log + per-batch summary, then move the tag
    # 3. repeat for 0.50 / 0.70 / 1.00 m, and a "tilted" group each
    # 4. run scripts/analyze_distance_results.py

Filtering rule (logged, never silent): a sample is rejected when
  - no target tag is detected in the frame, or
  - decision_margin < min_decision_margin, or
  - solvePnP returns None (bad CameraInfo / degenerate geometry).
Rejected frames are counted and reported per batch.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import Trigger

from apriltag_pose.pose_math import (
    absolute_error, camera_matrix_from_camera_info, compute_distance_stats,
    euclidean_distance, is_camera_info_valid, meters_to_mm,
    relative_error_percent, solve_tag_pose,
)


def _best_effort_qos(depth: int = 10) -> QoSProfile:
    """Subscriber QoS that receives from any publisher reliability."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.VOLATILE,
    )

CSV_FIELDS = [
    'timestamp', 'camera_device', 'image_width', 'image_height',
    'tag_family', 'tag_id', 'tag_size_m', 'true_distance_m',
    'estimated_tx_m', 'estimated_ty_m', 'estimated_tz_m', 'estimated_norm_m',
    'absolute_error_m', 'absolute_error_mm', 'relative_error_percent',
    'rvec_x', 'rvec_y', 'rvec_z', 'decision_margin', 'pose_method',
    'sample_group',
]


def _default_output_dir() -> str:
    return str(Path.home() / 'apriltag_pose_ws' / 'results')


class DistanceRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__('distance_recorder_node')

        self.declare_parameter('detections_topic', '/apriltag/detections')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('tag_size', 0.080)
        self.declare_parameter('target_tag_id', 0)
        self.declare_parameter('camera_device', 'usb_camera')
        self.declare_parameter('use_rectified_image', True)
        # batch params (set per batch via `ros2 param set` before start_batch)
        self.declare_parameter('true_distance_m', 0.500)
        self.declare_parameter('sample_group', 'front')
        self.declare_parameter('sample_count', 20)
        self.declare_parameter('min_decision_margin', 20.0)
        # output
        self.declare_parameter('output_directory', _default_output_dir())
        self.declare_parameter('csv_filename', 'distance_measurements.csv')

        self._det_topic = str(self.get_parameter('detections_topic').value)
        self._info_topic = str(self.get_parameter('camera_info_topic').value)
        self._tag_size = float(self.get_parameter('tag_size').value)
        self._target_id = int(self.get_parameter('target_tag_id').value)
        self._camera_device = str(self.get_parameter('camera_device').value)
        self._use_rect = bool(self.get_parameter('use_rectified_image').value)
        self._min_margin = float(self.get_parameter('min_decision_margin').value)
        self._out_dir = str(self.get_parameter('output_directory').value)
        self._csv_name = str(self.get_parameter('csv_filename').value)

        if self._tag_size <= 0:
            raise ValueError(f'tag_size must be > 0, got {self._tag_size}')
        os.makedirs(self._out_dir, exist_ok=True)
        self._csv_path = os.path.join(self._out_dir, self._csv_name)

        # state
        self._latest_info: Optional[CameraInfo] = None
        self._batch_active = False
        self._batch_truth = 0.0
        self._batch_group = ''
        self._batch_target_n = 0
        self._batch_samples: List[dict] = []
        self._batch_rejected = 0
        self._batch_reject_reasons = {'no_tag': 0, 'low_margin': 0, 'pnp_fail': 0}
        self._batch_t0 = 0.0
        self._last_warn = 0.0

        self.create_subscription(CameraInfo, self._info_topic,
                                 self._on_info, _best_effort_qos())
        self.create_subscription(AprilTagDetectionArray, self._det_topic,
                                 self._on_detections, _best_effort_qos())
        self.create_service(Trigger, 'start_batch', self._on_start_batch)

        self.get_logger().info(
            f'distance_recorder_node up:\n'
            f'  detections: {self._det_topic}\n'
            f'  camera_info:{self._info_topic}\n'
            f'  tag_size:   {self._tag_size:.4f} m  target_id={self._target_id}\n'
            f'  csv:        {self._csv_path}\n'
            f'  --> set true_distance_m & sample_group, then call '
            f'ros2 service call /distance_recorder_node/start_batch '
            f'std_srvs/srv/Trigger')

    # ------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        self._latest_info = msg

    # ------------------------------------------------------------------
    def _on_start_batch(self, _req, resp: Trigger.Response) -> Trigger.Response:
        # Read the (possibly just-set) params at trigger time.
        truth = float(self.get_parameter('true_distance_m').value)
        group = str(self.get_parameter('sample_group').value)
        n = int(self.get_parameter('sample_count').value)
        self._min_margin = float(self.get_parameter('min_decision_margin').value)
        if truth <= 0:
            resp.success = False
            resp.message = 'true_distance_m must be > 0; set it first.'
            return resp
        if n <= 0:
            resp.success = False
            resp.message = 'sample_count must be > 0.'
            return resp
        if self._latest_info is None:
            resp.success = False
            resp.message = 'no CameraInfo yet; start the camera first.'
            return resp

        self._batch_truth = truth
        self._batch_group = group
        self._batch_target_n = n
        self._batch_samples = []
        self._batch_rejected = 0
        self._batch_reject_reasons = {'no_tag': 0, 'low_margin': 0, 'pnp_fail': 0}
        self._batch_active = True
        self._batch_t0 = time.monotonic()
        self.get_logger().info(
            f'==== BATCH START: truth={truth:.3f} m  group="{group}"  '
            f'target_n={n}  min_margin={self._min_margin:.1f} ====')
        resp.success = True
        resp.message = (f'collecting {n} samples for truth={truth:.3f}m '
                        f'group={group}; see log for progress.')
        return resp

    # ------------------------------------------------------------------
    def _on_detections(self, msg: AprilTagDetectionArray) -> None:
        if not self._batch_active:
            return
        info = self._latest_info
        if info is None:
            return

        # pick the target detection
        det = None
        for d in msg.detections:
            if d.id == self._target_id:
                det = d
                break
        if det is None:
            self._reject('no_tag')
            return

        margin = float(det.decision_margin)
        if margin < self._min_margin:
            self._reject('low_margin')
            return

        # camera matrix (rectified -> P, else K+D)
        try:
            if self._use_rect:
                K = camera_matrix_from_camera_info(info.p, use_projection_matrix=True)
                dist = np.zeros((5,), dtype=np.float64)
            else:
                K = camera_matrix_from_camera_info(info.k, use_projection_matrix=False)
                dist = np.asarray(info.d, dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'bad CameraInfo: {exc}')
            self._reject('pnp_fail')
            return

        if not is_camera_info_valid(K):
            self._reject('pnp_fail')
            return

        corners = np.array([[c.x, c.y] for c in det.corners], dtype=np.float64)
        pose = solve_tag_pose(corners, self._tag_size, K, dist)
        if pose is None:
            self._reject('pnp_fail')
            return

        tx, ty, tz = float(pose.tvec[0]), float(pose.tvec[1]), float(pose.tvec[2])
        est = euclidean_distance((tx, ty, tz))
        abs_err = absolute_error(est, self._batch_truth)
        rel = relative_error_percent(est, self._batch_truth)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self._batch_samples.append({
            'timestamp': f'{stamp:.6f}',
            'camera_device': self._camera_device,
            'image_width': int(info.width),
            'image_height': int(info.height),
            'tag_family': str(det.family),
            'tag_id': int(det.id),
            'tag_size_m': f'{self._tag_size:.6f}',
            'true_distance_m': f'{self._batch_truth:.6f}',
            'estimated_tx_m': f'{tx:.6f}',
            'estimated_ty_m': f'{ty:.6f}',
            'estimated_tz_m': f'{tz:.6f}',
            'estimated_norm_m': f'{est:.6f}',
            'absolute_error_m': f'{abs_err:.6f}',
            'absolute_error_mm': f'{meters_to_mm(abs_err):.3f}',
            'relative_error_percent': f'{rel:.4f}',
            'rvec_x': f'{float(pose.rvec[0]):.6f}',
            'rvec_y': f'{float(pose.rvec[1]):.6f}',
            'rvec_z': f'{float(pose.rvec[2]):.6f}',
            'decision_margin': f'{margin:.4f}',
            'pose_method': pose.method,
            'sample_group': self._batch_group,
        })

        n = len(self._batch_samples)
        if n == 1 or n % 5 == 0:
            self.get_logger().info(
                f'  [{n}/{self._batch_target_n}] est={est:.4f} m  '
                f'abs_err={meters_to_mm(abs_err):.1f} mm  '
                f'rel={rel:.2f}%  margin={margin:.1f}')
        if n >= self._batch_target_n:
            self._finalize_batch()

    # ------------------------------------------------------------------
    def _reject(self, reason: str) -> None:
        self._batch_rejected += 1
        self._batch_reject_reasons[reason] = \
            self._batch_reject_reasons.get(reason, 0) + 1
        now = time.monotonic()
        if now - self._last_warn > 2.0:
            self._last_warn = now
            self.get_logger().warn(
                f'rejected {self._batch_rejected} frames so far '
                f'({self._batch_reject_reasons}); keeping tag steady and '
                f'well-lit, check min_decision_margin.')

    # ------------------------------------------------------------------
    def _finalize_batch(self) -> None:
        self._batch_active = False
        rows = self._batch_samples
        write_header = not os.path.isfile(self._csv_path) \
            or os.path.getsize(self._csv_path) == 0
        with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                w.writeheader()
            for row in rows:
                w.writerow(row)

        est = [float(r['estimated_norm_m']) for r in rows]
        stats = compute_distance_stats(est, self._batch_truth)
        abs_max = max(abs(stats.maximum - self._batch_truth),
                      abs(stats.minimum - self._batch_truth))
        dur = time.monotonic() - self._batch_t0
        self.get_logger().info(
            f'==== BATCH COMPLETE: group="{self._batch_group}" '
            f'truth={self._batch_truth:.3f} m ====\n'
            f'  accepted {stats.n} / {stats.n + self._batch_rejected} frames '
            f'({dur:.1f}s); rejected {self._batch_rejected} '
            f'{self._batch_reject_reasons}\n'
            f'  mean={stats.mean:.4f} m  median={stats.median:.4f} m  '
            f'MAE={meters_to_mm(stats.mae):.1f} mm  '
            f'RMSE={meters_to_mm(stats.rmse):.1f} mm  '
            f'max|err|={meters_to_mm(abs_max):.1f} mm  '
            f'bias={meters_to_mm(stats.bias):+.1f} mm\n'
            f'  appended to {self._csv_path}')
        self.get_logger().info(
            '  -> move the tag and set true_distance_m / sample_group, '
            'then call start_batch again.')


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = DistanceRecorderNode()
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
