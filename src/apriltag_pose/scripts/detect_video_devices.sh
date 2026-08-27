#!/usr/bin/env bash
# detect_video_devices.sh - 枚举所有 V4L2 视频设备, 用于定位电脑内置摄像头 / K230.
# 用法: bash detect_video_devices.sh
set -u

echo "===== lsusb ====="
lsusb 2>/dev/null || true
echo
echo "===== /dev/video* ====="
ls -l /dev/video* 2>/dev/null || echo "(无 /dev/video*)"
echo
echo "===== v4l2-ctl --list-devices ====="
v4l2-ctl --list-devices 2>/dev/null || echo "(v4l2-ctl 不可用, sudo apt install v4l-utils)"
echo
echo "===== /dev/v4l/by-id (稳定路径, 推荐在 launch 里使用) ====="
ls -l /dev/v4l/by-id/ 2>/dev/null || echo "(无 by-id)"
echo
echo "===== /dev/v4l/by-path ====="
ls -l /dev/v4l/by-path/ 2>/dev/null || echo "(无 by-path)"
echo
echo "===== 每个视频节点支持的格式/分辨率/帧率 ====="
for dev in /dev/video*; do
  [ -e "$dev" ] || continue
  echo "----- $dev -----"
  v4l2-ctl -d "$dev" --list-formats-ext 2>/dev/null || true
done
echo
echo "提示: K230 (CanMV 固件) 默认枚举为 CDC 串口 + MTP, 不产生 /dev/videoX."
echo "      若 lsusb 看到 'Kendryte / CanMV' 但没有新 video 节点, 属情况B, 见 README."
