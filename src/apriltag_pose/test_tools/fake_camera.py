#!/usr/bin/env python3
"""Publish a synthetic AprilTag scene so the whole ROS pipeline can be tested
without a camera or a printed tag.

Publishes:
    /camera/image_rect   (rgb8)  - a rendered tag36h11 ID0 view, orbiting slowly
    /camera/camera_info          - matching intrinsics (already "rectified")

Together with apriltag_ros + tag_visualizer_node + ar_object_node this exercises
the real topic/QoS/TF plumbing, not just the math.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, str(Path('/home/czm/apriltag_pose_ws/src/apriltag_pose')))
from apriltag_pose.ar_geometry import project_points  # noqa: E402

W, H = 640, 480
FX = FY = 600.0
CX, CY = 320.0, 240.0
TAG = 0.08


class FakeCam(Node):
    def __init__(self):
        super().__init__('fake_camera')
        # image_transport (used by apriltag_ros) subscribes RELIABLE, so the
        # publisher must be RELIABLE too or no images are delivered at all.
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=5,
                         durability=DurabilityPolicy.VOLATILE)
        # small warmup so late subscribers (image_transport) don't miss the start
        self._warmup = 0
        self.pub_img = self.create_publisher(Image, '/camera/image_rect', qos)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', qos)
        self.bridge = CvBridge()
        d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36H11)
        cell, quiet = 50, 2
        side, pad = 8 * cell, quiet * cell
        sheet = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
        sheet[pad:pad + side, pad:pad + side] = cv2.aruco.drawMarker(d, 0, side)
        self.sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
        self.sheet_m = (side + 2 * pad) / side * TAG
        self.k = 0
        self.create_timer(1.0 / 15.0, self.tick)

    def tick(self):
        # orbit the camera around the tag so the AR object must track it
        t = self.k / 15.0
        yaw = 0.5 * np.sin(t * 0.6)
        rvec = np.array([[np.pi - 0.25 * np.sin(t * 0.4)], [yaw], [0.05 * np.sin(t)]])
        tvec = np.array([[0.03 * np.sin(t * 0.5)], [0.0],
                         [0.45 + 0.12 * np.sin(t * 0.3)]])
        s = self.sheet_m / 2
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
        K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
        dst = project_points(obj, rvec, tvec, K, None).astype(np.float32)
        n = self.sheet.shape[0]
        src = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)
        Mh = cv2.getPerspectiveTransform(src, dst)
        canvas = np.full((H, W, 3), 200, np.uint8)
        img = cv2.warpPerspective(self.sheet, Mh, (W, H), dst=canvas,
                                  borderMode=cv2.BORDER_TRANSPARENT)

        stamp = self.get_clock().now().to_msg()
        m = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        m.header.stamp = stamp
        m.header.frame_id = 'camera_optical_frame'
        self.pub_img.publish(m)

        ci = CameraInfo()
        ci.header.stamp = stamp
        ci.header.frame_id = 'camera_optical_frame'
        ci.width, ci.height = W, H
        ci.distortion_model = 'plumb_bob'
        ci.d = [0.0] * 5
        ci.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(ci)
        self.k += 1


def main():
    rclpy.init()
    n = FakeCam()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()


if __name__ == '__main__':
    main()
