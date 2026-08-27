#!/usr/bin/env python3
"""Directly drive ar_object_node's ROS interface: publish Image + CameraInfo +
AprilTagDetectionArray ourselves (no apriltag_ros), then check /apriltag/image_ar
and /apriltag/ar_marker come back with correct content.

This isolates the node under test from apriltag_ros / image_transport plumbing.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import MarkerArray

sys.path.insert(0, '/home/czm/apriltag_pose_ws/src/apriltag_pose')
from apriltag_pose.ar_geometry import project_points  # noqa: E402

W, H = 640, 480
FX = FY = 600.0
CX, CY = 320.0, 240.0
TAG = 0.08


def qos(reliable=True, depth=5):
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable
        else ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST, depth=depth,
        durability=DurabilityPolicy.VOLATILE)


class Driver(Node):
    def __init__(self):
        super().__init__('ar_driver')
        self.pub_img = self.create_publisher(Image, '/camera/image_rect', qos())
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', qos())
        self.pub_det = self.create_publisher(AprilTagDetectionArray,
                                             '/apriltag/detections', qos())
        self.sub_ar = self.create_subscription(Image, '/apriltag/image_ar',
                                              self.on_ar, qos())
        self.sub_mk = self.create_subscription(MarkerArray, '/apriltag/ar_marker',
                                               self.on_mk, qos())
        self.bridge = CvBridge()
        self.got_img = 0
        self.got_mk = 0
        self.marker = None
        self.ar_frames = []

        d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36H11)
        cell, quiet = 50, 2
        side, pad = 8 * cell, quiet * cell
        sheet = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
        sheet[pad:pad + side, pad:pad + side] = cv2.aruco.drawMarker(d, 0, side)
        self.sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
        self.sheet_m = (side + 2 * pad) / side * TAG
        self.dict = d

    def on_ar(self, msg):
        self.got_img += 1
        self.ar_frames.append(self.bridge.imgmsg_to_cv2(msg, 'bgr8'))

    def on_mk(self, msg):
        self.got_mk += 1
        if msg.markers:
            self.marker = msg.markers[0]

    def render(self, rvec, tvec):
        s = self.sheet_m / 2
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
        K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
        dst = project_points(obj, rvec, tvec, K, None).astype(np.float32)
        n = self.sheet.shape[0]
        src = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)
        Mh = cv2.getPerspectiveTransform(src, dst)
        canvas = np.full((H, W, 3), 200, np.uint8)
        return cv2.warpPerspective(self.sheet, Mh, (W, H), dst=canvas,
                                   borderMode=cv2.BORDER_TRANSPARENT)

    def send(self, rvec, tvec):
        img = self.render(rvec, tvec)
        corners, ids, _ = cv2.aruco.detectMarkers(img, self.dict)
        if ids is None or 0 not in ids.ravel():
            return False, None
        quad = corners[int(np.where(ids.ravel() == 0)[0][0])].reshape(4, 2)
        apr = quad[[3, 2, 1, 0], :]          # aruco TL,TR,BR,BL -> apriltag BL,BR,TR,TL

        stamp = self.get_clock().now().to_msg()
        m = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        m.header.stamp = stamp
        m.header.frame_id = 'camera_optical_frame'

        ci = CameraInfo()
        ci.header.stamp = stamp
        ci.header.frame_id = 'camera_optical_frame'
        ci.width, ci.height = W, H
        ci.distortion_model = 'plumb_bob'
        ci.d = [0.0] * 5
        ci.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]

        da = AprilTagDetectionArray()
        da.header = m.header
        det = AprilTagDetection()
        det.family = '36h11'
        det.id = 0
        det.hamming = 0
        det.decision_margin = 80.0
        det.centre.x = float(apr[:, 0].mean())
        det.centre.y = float(apr[:, 1].mean())
        for i in range(4):
            det.corners[i].x = float(apr[i, 0])
            det.corners[i].y = float(apr[i, 1])
        da.detections.append(det)

        self.pub_info.publish(ci)
        self.pub_img.publish(m)
        self.pub_det.publish(da)
        return True, img


def main():
    rclpy.init()
    n = Driver()
    views = [
        ('front',   np.array([[np.pi], [0.0], [0.0]]),  np.array([[0.], [0.], [0.40]])),
        ('tilt-R',  np.array([[2.95], [0.6], [0.05]]),  np.array([[0.02], [0.], [0.45]])),
        ('tilt-L',  np.array([[2.90], [-0.6], [-.05]]), np.array([[-.02], [0.], [0.45]])),
        ('far',     np.array([[np.pi], [0.1], [0.0]]),  np.array([[0.], [0.], [0.85]])),
    ]
    # let discovery settle
    deadline = time.time() + 5
    while time.time() < deadline:
        rclpy.spin_once(n, timeout_sec=0.05)

    results = []
    for name, rv, tv in views:
        before = n.got_img
        for _ in range(6):                     # a few frames per view
            n.send(rv, tv)
            end = time.time() + 0.4
            while time.time() < end:
                rclpy.spin_once(n, timeout_sec=0.02)
        results.append((name, n.got_img - before))

    print(f'/apriltag/image_ar   frames received : {n.got_img}')
    print(f'/apriltag/ar_marker  msgs   received : {n.got_mk}')
    for name, cnt in results:
        print(f'  view {name:<8} -> {cnt} annotated frames')

    ok = True
    if n.got_img == 0:
        print('[FAIL] no AR images received'); ok = False
    if n.marker is None:
        print('[FAIL] no AR marker received'); ok = False
    else:
        mk = n.marker
        print(f'\nmarker: frame_id={mk.header.frame_id!r} type={mk.type} '
              f'pos=({mk.pose.position.x:.3f},{mk.pose.position.y:.3f},'
              f'{mk.pose.position.z:.3f}) scale=({mk.scale.x:.3f},'
              f'{mk.scale.y:.3f},{mk.scale.z:.3f})')
        if mk.header.frame_id != 'tag_0':
            print(f'[FAIL] marker frame_id must be tag_0, got {mk.header.frame_id}')
            ok = False
        expect = (0.0, 0.06, 0.04)
        got = (mk.pose.position.x, mk.pose.position.y, mk.pose.position.z)
        if not np.allclose(got, expect, atol=1e-6):
            print(f'[FAIL] marker offset {got} != expected {expect}'); ok = False
        if not np.isclose(mk.scale.x, 0.04, atol=1e-6):
            print(f'[FAIL] marker scale {mk.scale.x} != 0.04'); ok = False

    if n.ar_frames:
        # the AR overlay must actually change the image vs the plain render
        _, plain = n.send(*views[0][1:])
        rendered = n.ar_frames[-1]
        diff = int(np.abs(rendered.astype(int) - plain.astype(int)).sum())
        print(f'overlay pixel-difference vs plain render: {diff} (must be > 0)')
        if diff == 0:
            print('[FAIL] AR image identical to input - nothing was drawn'); ok = False
        out = Path('/tmp/ar_verify/ros_ar_output.png')
        cv2.imwrite(str(out), np.hstack(n.ar_frames[-4:]))
        print(f'saved {out}')

    print('\n[PASS] ar_object_node ROS interface verified' if ok else '\n[FAIL]')
    n.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
