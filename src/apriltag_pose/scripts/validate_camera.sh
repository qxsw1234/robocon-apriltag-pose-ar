#!/usr/bin/env bash
# validate_camera.sh - 启动 camera_only, 验证图像话题/帧率/CameraInfo 是否正常.
# 用法: bash validate_camera.sh [video_device]
set -u

VIDEO="${1:-/dev/v4l/by-id/usb-HD_Camera_Manufacturer_USB_2.0_Camera-video-index0}"
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/apriltag_pose_ws/install/setup.bash 2>/dev/null

echo ">>> 启动 v4l2_camera (设备: $VIDEO) ..."
ros2 launch apriltag_pose camera_only.launch.py video_device:="$VIDEO" \
    > /tmp/validate_camera.log 2>&1 &
LAUNCH_PID=$!
trap 'kill $LAUNCH_PID 2>/dev/null; wait $LAUNCH_PID 2>/dev/null' EXIT

echo ">>> 等待 5 秒让相机启动 ..."
sleep 5

echo
echo "===== topic list ====="
ros2 topic list 2>/dev/null | grep -E "camera|image" || true
echo
echo "===== /camera/image_raw 帧率 (3秒) ====="
timeout 3 ros2 topic hz /camera/image_raw 2>/dev/null || echo "(未收到 /camera/image_raw)"
echo
echo "===== /camera/camera_info (一次) ====="
timeout 3 ros2 topic echo /camera/camera_info --once 2>/dev/null \
    | grep -E "frame_id|height|width|K:|D:|P:" || echo "(未收到 camera_info)"

echo
echo ">>> 完成, 关闭相机."
