# ROS 2 AprilTag 相机位姿估计与 AR

本工作空间对应视觉组正式考核方向一，基于 ROS 2 Humble、OpenCV 和 AprilTag 实现相机图像发布、标签检测、角点/中心/ID 标注、PnP 三维位姿估计、TF、RViz2 坐标系显示和世界坐标系固定 AR 物体。

完整使用说明、功能清单和当前完成状态见 [`src/apriltag_pose/README.md`](src/apriltag_pose/README.md)，稳定性实验见 [`src/apriltag_pose/STABILITY.md`](src/apriltag_pose/STABILITY.md)，开发过程与 AI 使用说明见 [`DEVELOPMENT_RECORD.md`](DEVELOPMENT_RECORD.md)。

## 当前提交状态

- 代码、构建与单元测试完成，最终 82/82 测试通过；最新构建与测试日志见 `docs/test_logs/`；
- 普通题和 AR 拓展题的算法、节点与离线闭环验证完成；
- 已生成可打印棋盘格和 AprilTag；
- 本次提交未完成真实摄像头内参标定、实机测距和实机 AR 视频，`camera.yaml` 保留为零内参占位文件；
- 提交内容按“当前代码进度 + 离线结果 + 问题分析”如实呈现，不将离线验证描述为实机结果。

## 构建

```bash
source /opt/ros/humble/setup.bash
cd ~/apriltag_pose_ws
colcon build --packages-select apriltag_pose --symlink-install
source install/setup.bash
```
