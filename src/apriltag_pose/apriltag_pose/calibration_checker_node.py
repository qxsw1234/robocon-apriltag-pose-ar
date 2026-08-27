"""calibration_checker_node.

Validates the camera intrinsics from two sources and reports PASS/WARN/FAIL:

  1. The calibration YAML file on disk (camera_calibration output format):
       image_width, image_height, camera_name, camera_matrix{rows,cols,data},
       distortion_model, distortion_coefficients{...},
       rectification_matrix{...}, projection_matrix{...}
  2. The live /camera/camera_info topic (K, D, R, P).

Run after calibration and after launching the camera to confirm the file is
loadable and that v4l2_camera actually published the intrinsics (a common
failure is CameraInfo full of zeros because camera_info_url was not set).

The node does a one-shot check (file on startup, live topic on first message)
then shuts itself down so it can be used in a launch/CI step.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

from apriltag_pose.calibration_io import (
    load_calibration_yaml, validate_calibration_file)


class Check:
    def __init__(self, name: str, level: str, detail: str) -> None:
        self.name, self.level, self.detail = name, level, detail

    def __str__(self) -> str:
        return f'[{self.level}] {self.name}: {self.detail}'


class CalibrationCheckerNode(Node):
    def __init__(self) -> None:
        super().__init__('calibration_checker_node')

        self.declare_parameter('calibration_file', '')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('expected_width', 0)    # 0 = don't check
        self.declare_parameter('expected_height', 0)
        self.declare_parameter('timeout_sec', 5.0)

        self._path = str(self.get_parameter('calibration_file').value)
        self._info_topic = str(self.get_parameter('camera_info_topic').value)
        self._exp_w = int(self.get_parameter('expected_width').value)
        self._exp_h = int(self.get_parameter('expected_height').value)
        self._timeout = float(self.get_parameter('timeout_sec').value)

        self._checks: List[Check] = []
        self._file_K: Optional[List[float]] = None
        self._file_wh: Tuple[int, int] = (0, 0)
        self._got_info = False

        # 1. file checks (immediate)
        self._check_file()

        # 2. live camera_info check (on first message)
        self.create_subscription(CameraInfo, self._info_topic,
                                 self._on_info, 10)
        self._t0 = self.get_clock().now()
        self._timer = self.create_timer(0.2, self._on_tick)

    # ------------------------------------------------------------------
    def _check_file(self) -> None:
        # Delegate the file-format validation to the ROS-free calibration_io.
        for name, level, detail in validate_calibration_file(self._path):
            self._checks.append(Check(name, level, detail))

        # Also stash the file K + resolution for the live-vs-file comparison.
        if self._path and os.path.isfile(self._path):
            try:
                cfg = load_calibration_yaml(self._path)
                self._file_wh = (int(cfg.get('image_width', 0)),
                                 int(cfg.get('image_height', 0)))
                cm = cfg.get('camera_matrix')
                if isinstance(cm, dict) and len(cm.get('data', [])) == 9:
                    self._file_K = [float(v) for v in cm['data']]
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        if self._got_info:
            return
        self._got_info = True

        def nz(arr, n):
            return arr is not None and len(arr) >= n and any(v != 0.0 for v in arr[:n])

        if nz(list(msg.k), 9):
            fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
            self._checks.append(Check('live_K', 'PASS',
                                      f'fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}'))
            if self._file_K is not None and list(msg.k) != self._file_K:
                self._checks.append(Check('live_vs_file_K', 'WARN',
                                          'live K differs from file K '
                                          '(different resolution/republish?)'))
            else:
                self._checks.append(Check('live_vs_file_K', 'PASS', 'matches file'))
        else:
            self._checks.append(Check('live_K', 'FAIL',
                                      'CameraInfo K is all zeros -- '
                                      'camera_info_url not loaded by v4l2_camera?'))
        if nz(list(msg.d), 5):
            self._checks.append(Check('live_D', 'PASS', f'{len(msg.d)} coeffs'))
        else:
            self._checks.append(Check('live_D', 'WARN', 'D empty/zero on live topic'))
        if nz(list(msg.r), 9):
            self._checks.append(Check('live_R', 'PASS', '9 elements'))
        else:
            self._checks.append(Check('live_R', 'WARN', 'R empty/zero'))
        if nz(list(msg.p), 12):
            self._checks.append(Check('live_P', 'PASS', '12 elements'))
        else:
            self._checks.append(Check('live_P', 'WARN', 'P empty/zero'))

        if msg.width and msg.height:
            self._checks.append(Check('live_resolution', 'PASS',
                                      f'{msg.width}x{msg.height}'))
            if self._exp_w and (msg.width != self._exp_w or msg.height != self._exp_h):
                self._checks.append(Check('resolution_match', 'FAIL',
                                          f'live {msg.width}x{msg.height} != '
                                          f'expected {self._exp_w}x{self._exp_h}'))
            else:
                self._checks.append(Check('resolution_match', 'PASS',
                                          f'{msg.width}x{msg.height}'))
        self._finish()

    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        if self._got_info:
            return
        if (self.get_clock().now() - self._t0).nanoseconds * 1e-9 > self._timeout:
            self._checks.append(Check('live_topic', 'FAIL',
                                      f'no CameraInfo on {self._info_topic} within '
                                      f'{self._timeout}s'))
            self._finish()

    # ------------------------------------------------------------------
    def _finish(self) -> None:
        self._timer.cancel()
        self.get_logger().info('===== calibration check report =====')
        for c in self._checks:
            self.get_logger().info(str(c))
        fails = [c for c in self._checks if c.level == 'FAIL']
        warns = [c for c in self._checks if c.level == 'WARN']
        verdict = 'PASS' if not fails else 'FAIL'
        self.get_logger().info(
            f'===== verdict: {verdict} '
            f'({len(fails)} fail, {len(warns)} warn) =====')
        # non-zero exit if failed, so it can gate a pipeline
        raise SystemExit(1 if fails else 0)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = CalibrationCheckerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
