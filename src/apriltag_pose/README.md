# apriltag_pose — ROS 2 + OpenCV + AprilTag 相机位姿估计与测距系统

基于 ROS 2 Humble、OpenCV、apriltag_ros 的 AprilTag 相机位姿估计系统：读取 V4L2 摄像头（外置 USB 摄像头或电脑内置摄像头），发布图像，棋盘格标定内参，`image_proc` 去畸变，apriltag 检测，自定义节点用 `cv2.solvePnP` 计算 `rvec`/`tvec`，建立 TF，在图像上标注 ID/角点/中心/坐标轴/tvec/距离，在 RViz2 显示坐标系，并完成距离误差实验与统计报告。

**拓展题 1.2**：以 AprilTag 坐标系为世界参考系，在其中放置固定的虚拟三维物体（立方体/三棱锥/三维箭头）。相机移动或旋转时，物体按真实透视关系改变大小、方向和图像位置，但相对 AprilTag 世界系保持静止。

> 工作空间内原先有一个功能重叠的旧包 `apriltag_camera_pose`，已删除（删除前备份为工作空间根目录的 `apriltag_camera_pose_backup_*.tar.gz`），避免两个包造成混淆。

> **提交状态说明**：本次按“代码与离线验证成果”提交。真实摄像头内参标定、实机距离实验和实机 AR 视频尚未完成，`config/camera.yaml` 仍为零内参占位；离线结果不会表述为实机结果。开发过程与 AI 使用说明见工作空间根目录 `DEVELOPMENT_RECORD.md`。

---

## 1. 项目功能

### 普通题 1.1

1. 读取摄像头图像（V4L2，外置 USB 摄像头或电脑内置摄像头）
2. 通过 ROS 2 发布相机图像（`/camera/image_raw`、`/camera/camera_info`）
3. 检测画面中的 AprilTag（apriltag_ros，tag36h11）
4. 在图像中标注 AprilTag ID
5. 标注四个角点（编号 0/1/2/3）
6. 标注 AprilTag 中心
7. 完成摄像头内参标定（`camera_calibration` 棋盘格）
8. 保存并加载相机内参矩阵和畸变参数（`config/camera.yaml`）
9. 根据 AprilTag 实际边长和相机内参估计三维位姿（`solvePnP`，IPPE_SQUARE + ITERATIVE 后备）
10. 输出旋转向量 `rvec`
11. 输出平移向量 `tvec`（`/apriltag/pose`）
12. 建立相机坐标系与 AprilTag 坐标系之间的 TF（`camera_link → camera_optical_frame → tag_0`）
13. 在图像上绘制 AprilTag 三维坐标轴（`cv2.drawFrameAxes`）
14. 在 RViz2 中显示相机坐标系、AprilTag 坐标系和 TF
15. 测量估计距离与真实距离之间的误差
16. 保存距离误差实验数据（`results/distance_measurements.csv`）
17. 自动生成误差统计报告（`distance_summary.csv` / `distance_error_plot.png` / `distance_report.md`）
18. 完整 README、启动命令和故障排查说明

### 拓展题 1.2（AR 虚拟物体）

19. 以 AprilTag 坐标系为世界参考系，放置固定虚拟三维物体（`ar_object_node`）
20. 物体与标签有明确三维偏移（默认沿标签 +y 抬升 60 mm、沿法线 +z 外移 40 mm），不与标签重合
21. 透视投影 + 背面剔除 + 画家算法深度排序，半透明填充并描边（`/apriltag/image_ar`）
22. 同步在 `tag_0` frame 发布等价 RViz Marker（`/apriltag/ar_marker`），位置由 TF 自动跟随
23. 变换正确性的三重验证：离线闭环脚本 + 单元测试 + 节点启动自检（见 §21）

### 稳定性改进

24. 位姿滑动窗口滤波（平移中值/均值 + 四元数**符号对齐后**平均）
25. 平面 PnP 二义性（姿态翻转）抑制：枚举候选解 + 基于时序连续性择优
26. 异常帧剔除（`decision_margin` / 重投影误差 / 位姿突变）
27. 丢检保持防闪烁；摄像头帧率从 8 Hz 提升到 30 Hz

> 全部改进的原因、做法和**实测数据**见 [`STABILITY.md`](STABILITY.md)。

### 当前进度状态

| 项目 | 状态 | 说明 |
|---|---|---|
| 代码 / `colcon build` / 单元测试 | ✅ 完成 | pytest **82/82** 通过 |
| 拓展题 1.2 AR 虚拟物体 | ✅ 完成 | 24 项单测 + ROS 端到端验证通过 |
| 稳定性改进 | ✅ 完成 | 抖动降 3.5×、PnP 翻转 2→0、帧率 8→30 Hz（实测） |
| 变换关系正确性（离线闭环） | ✅ 完成 | 8 视角：位姿误差 ≤5.3 mm，tag 系漂移 **0.000000 mm** |
| 可打印标定物料 | ✅ 完成 | `print_assets/` 两个 A4 PDF，栅格化自校验精度 <0.08 mm |
| 棋盘格 / AprilTag 实物 | ⏳ **待打印 + 用尺子实测** | 见 §9.0 |
| `config/camera.yaml`（内参） | ⏳ 当前是零占位 | 需先打印棋盘格，再按 §9 标定生成 |
| 距离误差实验（4 距离 × 2 姿态 × ≥20 帧） | ⏳ 待实机采集 | 依赖上面两项 |
| `results/` 实机产物 | ⏳ 待实机运行 | 离线验证图已有：`results/ar_transform_verification.png` |

**未标定也能先跑**：传 `use_rectified_image:=false` 做 2D 冒烟测试（标注 ID/角点/中心，但无三维位姿）。

## 2. 系统架构图

```text
外置 USB 摄像头 或 电脑内置摄像头
        ↓
Linux V4L2 设备 /dev/videoX  (推荐用 /dev/v4l/by-id/... 稳定路径)
        ↓
v4l2_camera
        ↓
/camera/image_raw   /camera/camera_info
        ↓
image_proc::RectifyNode  (图像校正)
        ↓
/camera/image_rect
        ↓
apriltag_ros  (检测 + PnP + TF)
        ↓
/apriltag/detections   /tf  (camera_optical_frame → tag_0)
        ↓
        ├── tag_visualizer_node  (自算 solvePnP，测量路径，不滤波)
        │       ↓
        │   /apriltag/image_annotated  /apriltag/pose
        │   /apriltag/distance         /apriltag/markers
        │
        └── ar_object_node       (拓展题 1.2，显示路径，带滤波)
                ↓
            /apriltag/image_ar   /apriltag/ar_marker
        ↓
RViz2、rqt_image_view、distance_recorder_node → CSV → analyze_distance_results.py
```

## 3. 硬件清单

- Linux x86_64 笔记本（Ubuntu 22.04），ROS 2 Humble
- 摄像头（二选一，默认用第一个）：
  - **外置 USB 摄像头**：`USB 2.0 Camera: HD USB Camera`，稳定路径
    `/dev/v4l/by-id/usb-HD_Camera_Manufacturer_USB_2.0_Camera-video-index0`
    （实测支持 YUYV 640x480@30、MJPG 640x480@120）
  - 笔记本内置摄像头：`/dev/v4l/by-id/usb-SunplusIT_Inc_Integrated_RGB_Camera_01.00.00-video-index0`
- 打印的 AprilTag（tag36h11，ID 0；**黑色方块边长需实测**）
- 打印的 9×6 内部角点棋盘格（**方格边长需实测**，标称 25 mm）
- 卷尺/钢尺（测真实距离）+ 硬质背板（贴标签，防翘曲）

## 4. 软件环境

- Ubuntu 22.04.5 LTS (jammy)，内核 6.5
- ROS 2 Humble (`$ROS_DISTRO=humble`)
- Python 3.10、OpenCV 4.5.4、numpy、matplotlib、PyYAML、reportlab
  - `pandas` 仅 `analyze_distance_results.py` 需要：`sudo apt install python3-pandas`
- ROS 包：`v4l2_camera`、`image_proc`、`camera_calibration`、`cv_bridge`、`apriltag_ros`(3.4.0)、`apriltag_msgs`、`tf2_ros`、`rviz2`、`rqt_image_view`
- 工具：`v4l-utils`、`poppler-utils`（PDF 自校验用 `pdftoppm`）、`usbutils`

## 5. 摄像头设备确认

```bash
bash $(ros2 pkg prefix apriltag_pose)/share/apriltag_pose/scripts/detect_video_devices.sh
# 列出支持的格式/分辨率/帧率
v4l2-ctl -d /dev/video4 --list-formats-ext
```

**优先用 `/dev/v4l/by-id/...` 稳定路径**，因为 `/dev/videoN` 的编号会随插拔顺序变化。
本机一个摄像头占两个节点（如 video4/video5），`-video-index0` 才是取图那个。

## 6. 切换摄像头

launch 默认用外置 USB 摄像头。要换成内置摄像头：

```bash
ros2 launch apriltag_pose apriltag_pose.launch.py \
  video_device:=/dev/v4l/by-id/usb-SunplusIT_Inc_Integrated_RGB_Camera_01.00.00-video-index0
```

注意：**每个摄像头的内参不同**，换摄像头必须重新标定，并用
`camera_info_file:=<对应的 yaml>` 指定对应内参文件。

## 7. 依赖安装

核心 ROS 包通常已随 ROS 2 Humble 安装。如缺失：

```bash
sudo apt update
sudo apt install -y \
  ros-humble-v4l2-camera ros-humble-camera-calibration ros-humble-image-pipeline \
  ros-humble-cv-bridge ros-humble-image-transport ros-humble-apriltag-ros \
  ros-humble-apriltag-msgs ros-humble-tf2-ros ros-humble-tf2-tools \
  ros-humble-rviz2 ros-humble-rqt-image-view \
  python3-opencv python3-numpy python3-pandas python3-matplotlib python3-yaml \
  python3-reportlab v4l-utils poppler-utils usbutils
```

## 8. 编译命令

```bash
source /opt/ros/humble/setup.bash
cd ~/apriltag_pose_ws
colcon build --packages-select apriltag_pose --symlink-install
source install/setup.bash
```

## 9. 摄像头标定

### 9.0 先生成并打印标定物料

```bash
python3 ~/apriltag_pose_ws/src/apriltag_pose/scripts/make_print_assets.py
```

生成到 `~/apriltag_pose_ws/print_assets/`：

| 文件 | 内容 |
|---|---|
| `calibration_checkerboard_A4.pdf` | 9×6 内部角点棋盘格，方格标称 25 mm |
| `apriltag_36h11_id0_A4.pdf` | tag36h11 ID 0，黑方块标称 80 mm，含白色静默区 |
| `print_assets_README.txt` | 打印与实测说明 |

两个 PDF 都是**纯矢量**（每个黑格都是独立矢量矩形），打印边缘不发虚。
脚本每次运行都会自动做**几何自校验**：把 PDF 用 `pdftoppm` 栅格化到 300 dpi，
再用 `findChessboardCorners` / `cv2.aruco.detectMarkers` 重新检测，
验证格子数、解码 ID 和实际尺寸。本机自校验结果：

```
page size: 210.1 x 297.0 mm (A4 = 210.0 x 297.0)
[OK] checkerboard: 9x6 corners found, square = 24.998 / 24.994 mm (err 0.006 mm)
[OK] apriltag: decoded ID 0, black square edge = 79.925 mm (err 0.075 mm)
```

**打印三条铁律**：
1. 打印对话框里选**「实际大小 / 100%」**，取消勾选「适应页面 / Fit to page」；
2. 用普通 A4 **哑光纸**，不要相纸/铜版纸（反光会导致检测失败）；
3. 整面贴在硬纸板上，**不要只贴四角**（弯曲会让标定和位姿都错掉）。

**实测方法**（决定精度的关键）：
- 棋盘格：沿边量 **10 个方格的总长**（标称 250 mm）再 ÷10 → 单格边长。
  量总长再除，比直接量一格误差小 10 倍。
- AprilTag：量**黑色方块外边长**（标称 80 mm），**不含白边**。
  已用 `detectMarkers` 验证过：检测器返回的 4 个角点正好落在黑方块 4 角，
  所以「黑方块外边长」就是要填的 `tag_size`。

### 9.1 跑标定

标定必须在**最终运行时相同分辨率**（默认 640×480）下进行，并锁定对焦。

```bash
ros2 launch apriltag_pose calibration.launch.py \
  size:=<实测单格边长，米> checkerboard:=9x6
```

采集 25~40 个有效姿态：中心、四角、远近、水平/垂直倾斜、轻微旋转，覆盖全画面。
避免运动模糊、反光、棋盘格不平整。标定器显示 `CALIBRATED` 后点 `SAVE`，
生成 `/tmp/calibrationdata.tar.gz`（内含 `ost.yaml`）。
将 `ost.yaml` 内容按 `config/camera.yaml` 格式整理后覆盖它
（`image_width/image_height` 必须与运行分辨率一致）。

**验收线**：重投影误差 < 0.5 px。

标定后校验：

```bash
ros2 launch apriltag_pose apriltag_pose.launch.py \
  use_rectified_image:=false use_rviz:=false run_calibration_check:=true
# 或单独跑：
ros2 run apriltag_pose calibration_checker_node \
  --ros-args -p calibration_file:=$PWD/config/camera.yaml
```

相机内参矩阵：

```
K = [fx  0  cx
      0 fy  cy
      0  0   1]
```

- `fx`/`fy`：以像素为单位的焦距
- `cx`/`cy`：主点坐标
- `D`：镜头畸变参数
- `P`：校正图像使用的投影矩阵（去畸变后用 `P[:3,:3]` 作为新内参）

## 10. AprilTag 尺寸设置

1. 用 §9.0 生成的 `apriltag_36h11_id0_A4.pdf`（tag36h11 ID 0，已自校验）。
2. 100% 打印，**关闭「适应页面」缩放**。
3. 用游标卡尺/钢尺测量**黑色方块外边长**：
   - 不要测整张纸；
   - 不要把外围白色静默区算入 `tag_size`（静默区是检测必需的，但不属于 `tag_size`）。
4. 单位用**米**（如 79.4 mm → `0.0794`）。
5. 三处必须一致：
   - `config/apriltag.yaml` 的 `size` 与 `tag.sizes[0]`
   - `config/system.yaml` 的 `tag_size_m`
   - launch 参数 `tag_size`（或每次命令行传）

## 11. 修改配置

- `config/camera.yaml`：标定内参（标定后覆盖）
- `config/apriltag.yaml`：标签族、ID、`tag_size`（米）、检测器参数
- `config/system.yaml`：距离实验参数（真实距离、样本数、过滤阈值）
- `config/apriltag_pose.rviz`：RViz2 显示配置

## 12. 启动系统

```bash
source /opt/ros/humble/setup.bash
source ~/apriltag_pose_ws/install/setup.bash

# 完整系统（标定完成后），默认已开 AR 和 RViz
ros2 launch apriltag_pose apriltag_pose.launch.py tag_size:=<实测值>

# 指定设备/标签尺寸
ros2 launch apriltag_pose apriltag_pose.launch.py \
  video_device:=/dev/video0 tag_size:=0.0794 use_rviz:=true

# 标定前冒烟测试（无位姿，仅 2D 标注）
ros2 launch apriltag_pose apriltag_pose.launch.py use_rectified_image:=false

# 换 AR 物体形状 / 调偏移
ros2 launch apriltag_pose apriltag_pose.launch.py \
  ar_object_type:=pyramid ar_offset_y_m:=0.08 ar_offset_z_m:=0.05

# 关掉 AR（只做普通题 1.1）
ros2 launch apriltag_pose apriltag_pose.launch.py enable_ar:=false
```

主要 launch 参数：

| 分类 | 参数 |
|---|---|
| 相机 | `video_device camera_name camera_frame_id image_width image_height pixel_format output_encoding frame_rate camera_info_file` |
| 标签 | `tag_family tag_id tag_size tag_frame pose_method` |
| 可视化 | `axis_length use_rectified_image use_rviz` |
| AR (1.2) | `enable_ar ar_object_type ar_object_size_m ar_object_size_ratio ar_offset_x_m ar_offset_y_m ar_offset_z_m ar_fill_alpha` |
| 稳定性 | `filter_window use_median_translation pose_hold_sec min_decision_margin max_reproj_error_px` |
| 可选节点 | `enable_distance_recorder run_calibration_check` |

> `output_encoding` 默认 `mono8`（实测 30 Hz）。改成 `rgb8` 会掉到约 8 Hz。
> **不要**用 `pixel_format:=MJPG`，Humble 的 v4l2_camera 会崩溃。详见 `STABILITY.md`。

## 13. 查看图像

```bash
ros2 run rqt_image_view rqt_image_view    # 选 /apriltag/image_annotated 或 /apriltag/image_ar
ros2 topic hz /camera/image_rect
ros2 topic echo /camera/camera_info --once
```

## 14. 查看 TF

```bash
ros2 run tf2_ros tf2_echo camera_optical_frame tag_0
ros2 run tf2_tools view_frames               # 生成 TF 树 PDF
```

## 15. 打开 RViz2

launch 默认已带 RViz2（`use_rviz:=true`）。手动打开：

```bash
rviz2 -d $(ros2 pkg prefix apriltag_pose)/share/apriltag_pose/config/apriltag_pose.rviz
```

Fixed Frame = `camera_link`。显示：TF、Annotated Image(`/apriltag/image_annotated`)、AR Image(`/apriltag/image_ar`)、PoseArray(`/apriltag/pose`)、Markers(`/apriltag/markers`)、AR Object(`/apriltag/ar_marker`，挂在 `tag_0` frame)、tag_0 Axes、camera_link Axes、Grid。

## 16. 记录真实距离

```bash
# 终端 1：启动系统并启用记录器
ros2 launch apriltag_pose apriltag_pose.launch.py \
  enable_distance_recorder:=true use_rviz:=false

# 终端 2：引导式采集（按提示摆放标签、回车）
bash ~/apriltag_pose_ws/src/apriltag_pose/scripts/run_distance_test.sh
```

或手动控制单组：

```bash
ros2 param set /distance_recorder_node true_distance_m 0.30
ros2 param set /distance_recorder_node sample_group front
ros2 param set /distance_recorder_node sample_count 20
ros2 service call /distance_recorder_node/start_batch std_srvs/srv/Trigger
```

实验距离建议：0.30 / 0.50 / 0.70 / 1.00 m；每组姿态 `front`（正对）与 `tilted`（轻微倾斜）；每组合 ≥20 样本。要求：相机固定、标签贴硬质平面、卷尺测光心到标签中心、光照稳定、不手持相机、稳定后再记录。`decision_margin < min_decision_margin` 的样本被剔除（规则见节点日志）。

## 17. 生成误差报告

```bash
python3 ~/apriltag_pose_ws/src/apriltag_pose/scripts/analyze_distance_results.py
# 输出: results/distance_summary.csv, distance_error_plot.png, distance_report.md
```

报告含：全局 MAE/RMSE/最大误差/标准差/Bias；分组统计；4 张误差图；误差来源分析。

**`tz` 与 `norm(tvec)` 的区别**：
- `tz`：标签中心沿相机光轴方向的深度；
- `norm(tvec)=sqrt(tx²+ty²+tz²)`：相机光心到标签中心的空间直线距离（测距实验用此值）。

## 18. 常见错误

- **K230 不适用于本题**：本项目用普通 USB / 内置摄像头。（历史背景：曾试过亚博 K230，其 CanMV 固件枚举为 CDC+MTP 而非 UVC，不产生 `/dev/videoX`，无法被 `v4l2_camera` 读取，故未采用。）
- **USB 线只能供电不能传输数据**：换支持数据传输的 USB 线。
- **摄像头权限不足**：`sudo usermod -aG video $USER` 后重新登录；或临时 `sudo chmod 666 /dev/video0`。
- **`Failed getting value for control ...: Permission denied`**：v4l2_camera 读某个厂商私有控制项失败，**无害**，图像照常出。可忽略。
- **`Failed mapping device memory` / `Device or resource busy`**：设备已被别的进程占用。先 `pkill -9 -f v4l2_camera_node`，或用 `fuser -v /dev/video4` 找出占用者。
- **`/dev/videoX` 编号变化**：用 `/dev/v4l/by-id/...` 稳定路径（`detect_video_devices.sh` 列出）。一个摄像头通常占两个节点，取图的是 `-video-index0`。
- **帧率只有 8 Hz**：`output_encoding:=rgb8` 时 YUYV→RGB 转换是瓶颈。改用默认的 `mono8` 可达 30 Hz（实测）。详见 `STABILITY.md`。
- **`pixel_format:=MJPG` 直接崩溃**：Humble 的 v4l2_camera 0.6.2 不支持 MJPG，会报 `Current pixel format is not supported yet: MJPG` 然后在 cv_bridge 抛异常退出。**必须用 YUYV**，即使摄像头硬件标称 MJPG 帧率更高。
- **节点名出现重复 / 混进无关节点**：同一 ROS 域里跑了别的项目（如 Gazebo/Nav2）。用 `export ROS_DOMAIN_ID=42` 隔离，或先停掉其他项目。
- **CameraInfo 始终为零**：`camera_info_url` 未加载。确认 `config/camera.yaml` 已用真实标定覆盖，且路径正确；`run_calibration_check:=true` 排查。
- **标定文件没有加载**：检查 `camera_info_file` 参数（相对名在包 `config/` 下查找，或给绝对路径）。
- **图像分辨率与标定文件不一致**：标定时与运行时必须同分辨率；改分辨率必须重新标定。
- **apriltag_ros 收不到图像**：检查 `image_rect`/`camera_info` remap 与 QoS。`ros2 topic info /apriltag/image_rect -v` 看 pub/sub QoS 是否匹配；本包订阅用 BEST_EFFORT、发布用 RELIABLE 以规避不匹配。
- **image 和 CameraInfo 时间戳不匹配**：`tag_visualizer_node` 用 `message_filters` ApproximateTimeSynchronizer（slop 0.15s）；若帧率过低或抖动大，增大 `sync_slop_sec`。
- **检测不到标签**：确认 `family`(36h11)、`tag.ids`、光照、标签平整度、打印未缩放、距离适中。
- **tag family 设置错误**：`config/apriltag.yaml` 的 `family` 必须与打印的标签族一致。
- **tag_size 单位填成毫米**：`size`/`tag.sizes`/`tag_size` 单位是**米**（80 mm 写 0.080）。填错会导致距离整体成比例偏大/偏小。
- **TF 坐标方向错误 / 相机光学坐标方向错误**：`camera_link→camera_optical_frame` 用四元数 `(-0.5,0.5,-0.5,0.5)`（REP-103 body→optical：optical +X右 +Y下 +Z前）。方向错就检查该静态 TF。
- **RViz Fixed Frame 错误**：Fixed Frame 设 `camera_link`；只有光学坐标系时设 `camera_optical_frame`。
- **solvePnP 输出跳动**：检查标签平整度、光照、`decision_margin`、是否用了校正图、tag_size 是否正确；角点顺序由 `pose_math.apriltag_corners_to_opencv_ippe_order` 处理（`[3,2,1,0]`），勿改。
- **距离整体成比例偏大或偏小**：优先复核 `tag_size`（米）与实测值；其次标定 `fx/fy`；最后真实距离测量基准点。
- **标签靠近画面边缘时误差增大**：畸变校正残留 + 边缘解析力下降；优先在画面中心采集。
- **自动对焦导致内参变化**：标定与运行前锁定对焦（`v4l2-ctl -d /dev/videoX -c focus_auto=0`）。
- **重复 TF 发布**：apriltag_ros 发 `tag_0` TF，本包的可视化节点**不**重复发同名 TF（只发 PoseStamped/Marker），避免冲突。
- **OpenCV 角点顺序不一致**：apriltag_msgs `corners` 为 [BL,BR,TR,TL]（图像坐标），`pose_math` 把它重排为 IPPE_SQUARE 要求的 [TL,TR,BR,BL]；若换其他检测器需重新核对顺序。

## 19. 完整卸载或清理方法

```bash
# 停止所有相关进程
pkill -f "apriltag_pose.launch" ; pkill -f v4l2_camera_node ; pkill -f apriltag_node
pkill -f ar_object_node ; pkill -f tag_visualizer_node

# 删除本包
rm -rf ~/apriltag_pose_ws/src/apriltag_pose
rm -rf ~/apriltag_pose_ws/build/apriltag_pose ~/apriltag_pose_ws/install/apriltag_pose

# 卸载 apt 包（可选，按需）
sudo apt remove ros-humble-apriltag-ros ros-humble-v4l2-camera   # 谨慎，会影响其他项目
```

## 20. 考核演示流程

**普通题 1.1**

1. `ros2 launch apriltag_pose apriltag_pose.launch.py tag_size:=<实测值> use_rviz:=true`
2. 用 rqt_image_view 看 `/apriltag/image_annotated`：展示 ID、四角点(0/1/2/3)、中心十字、三维坐标轴、tvec、距离、重投影误差
3. `ros2 run tf2_ros tf2_echo camera_optical_frame tag_0`：展示 TF
4. `ros2 run tf2_tools view_frames`：生成 TF 树 PDF
5. RViz2：展示 camera_link / tag_0 坐标系与连线
6. 移动标签：展示距离实时变化、靠近变大/远离变小
7. `enable_distance_recorder:=true` + `run_distance_test.sh`：采集多距离多姿态
8. `analyze_distance_results.py`：展示 CSV/图/报告与 MAE/RMSE

**拓展题 1.2（AR）**

9. 看 `/apriltag/image_ar`：虚拟立方体悬浮在标签上方，与标签有明显三维偏移
10. **绕着标签移动相机**：物体近大远小、朝向随视角改变、可见面数变化，但始终"钉"在标签同一位置 → 这就是"相对世界系静止"
11. RViz2 看 `AR Object (tag frame)` marker：它与 `tag_0` 坐标系的相对位置恒定不动
12. 节点启动日志里的自检：`max delta in TAG frame = 0.000000000 m`
13. `python3 scripts/verify_ar_transform.py -o /tmp/ar.png`：离线 8 视角闭环验证
14. `ros2 launch ... ar_object_type:=pyramid` / `arrow`：换物体形状

**稳定性**

15. 展示 `STABILITY.md` 的实测数据表
16. 用手短暂遮挡标签：画面显示 `POSE HELD`，物体不闪烁消失
17. `colcon test --packages-select apriltag_pose` → 82/82 通过

---

## 21. 拓展题 1.2：AR 虚拟物体的原理与验证

### 数学原理

`solvePnP` 给出的 `(rvec, tvec)` 描述的正是 **tag 系 → 相机光学系** 的刚体变换：

```
X_cam = R(rvec) · X_tag + tvec
```

所以只要把虚拟物体的顶点写成 **tag 系坐标**，直接喂给
`cv2.projectPoints(rvec, tvec, K, D)` 就得到正确的透视投影。
物体因此天然"钉死"在 AprilTag 建立的世界系里 —— 相机怎么动，
物体在世界系中的坐标都不变，只有投影结果变。

代码上的体现：`ar_object_node` 里 `self._mesh` **只在构造函数里建一次**，
每帧都用同一份 tag 系顶点，从不修改。

### 渲染

- **背面剔除**：面法线与视线夹角 > 90° 的面不画（`is_face_visible`）
- **画家算法**：剩下的面按面心深度从远到近绘制，近处自然覆盖远处（`sort_faces_back_to_front`）
- 两者都在**相机系**中判断，因为可见性取决于相机而非世界系
- 半透明填充（`fillConvexPoly` + `addWeighted`）+ 描边
- 顶点跑到相机背后时整体跳过渲染，避免 `projectPoints` 输出发散坐标画出乱线

### 物体定义（参数化，非硬编码）

| 参数 | 默认 | 说明 |
|---|---|---|
| `ar_object_type` | `cube` | `cube` / `pyramid` / `arrow` |
| `ar_object_size_ratio` | `0.5` | 物体尺寸 = 该比例 × `tag_size` |
| `ar_offset_x/y/z_m` | `0 / 0.06 / 0.04` | tag 系偏移（米），**非零**以保证不与标签重合 |

默认物体是边长 = 0.5×tag_size 的立方体，沿标签 +y 抬升 60 mm、
沿法线 +z 外移 40 mm，整体完全脱离标签平面。

### 变换正确性的三重验证

**① 离线闭环脚本**（不需要摄像头和打印件）

```bash
python3 ~/apriltag_pose_ws/src/apriltag_pose/scripts/verify_ar_transform.py \
    -o ~/apriltag_pose_ws/results/ar_transform_verification.png
```

做法：用 `cv2.aruco` 渲染真实 tag36h11 ID0 图 → 按**已知真值位姿**做透视变换
合成"相机画面" → 重新检测并 `solvePnP` 恢复位姿 → 比较恢复值与真值 →
把物体顶点变换到相机系再**反算回 tag 系**，验证坐标不变。

本机实测（8 个视角）：

```
max position error      : 5.309 mm   (容差 10 mm)
max drift in TAG frame  : 0.000000000 mm   ← 物体在世界系中完全静止
perspective check: near(0.25m)=19786 > front(0.40m)=5507 > far(0.80m)=1173 px²
[PASS] all views
```

这个验证能抓出坐标系约定类的静默错误（角点顺序、`tag_size` 定义、
旋转方向、tag系↔相机系搞反等），且因为真值是自己指定的，误差可以定量衡量。

**② 单元测试**（`test/test_ar_geometry.py`，24 项）

关键几项：
- `test_tag_frame_coords_invariant_camera_frame_coords_change`：tag 系坐标严格不变，相机系坐标必须变
- `test_object_gets_smaller_when_camera_moves_away`：距离翻倍 → 投影面积约 1/4
- `test_projection_matches_manual_pinhole_math`：手算 `u=fx·X/Z+cx` 与 `projectPoints` 一致
- `test_only_front_face_visible_when_facing_camera`：正对时 6 面只有 1 面可见
- `test_rejects_geometry_behind_camera`：相机背后的几何拒绝渲染

**③ 节点启动自检**

`ar_object_node` 启动时对 3 个不同位姿跑 `verify_static_in_world`，日志打印：

```
self-check "static in tag/world frame":
  max delta in TAG frame    = 0.000000000 m  (must be 0)
  max delta in CAMERA frame = 0.199077 m  (must be > 0)
```

**④ RViz 交叉验证**

`/apriltag/ar_marker` 的 `frame_id` 是 `tag_0`，位置就是那个固定偏移。
RViz 用 TF 把它摆到正确位置 —— 物体相对 tag 静止这件事由 TF 自动体现，
无需每帧重算。图像侧和 RViz 侧两条独立路径显示一致，即互为验证。

---

## 角点顺序与坐标系约定（核心原理）

相机光学坐标系（ROS/REP-103，与 OpenCV 图像一致）：+X 图像右、+Y 图像下、+Z 朝场景前方。`solvePnP` 的 `tvec` 直接落在该系，`tz>0` 为标签深度。

AprilTag 角点（libapriltag）：标签系 +X 右、+Y 上、+Z 出标签面（朝观察者）；`corners[0..3]` = [BL, BR, TR, TL]（标签系 CCW-from-BL；图像坐标下顺时针）。`cv2.SOLVEPNP_IPPE_SQUARE` 要求物点 [TL,TR,BR,BL] = `[(-s,+s),(+s,+s),(+s,-s),(-s,-s)]`，故对图像角点取 `[3,2,1,0]` 重排。重投影误差超阈值则回退 `SOLVEPNP_ITERATIVE`，所用方法记录在 `PoseResult.method` 与 CSV 的 `pose_method` 列。

距离：`norm(tvec)=sqrt(tx²+ty²+tz²)`（光心到标签中心直线距离）；`tz`（光轴方向深度）。两者在报告中明确区分。

## TF 树

```
camera_link  (x-fwd, y-left, z-up)        ← 静态 TF (quat -0.5,0.5,-0.5,0.5)
  └── camera_optical_frame  (x-right, y-down, z-fwd)   ← v4l2_camera image frame_id
        └── tag_0                          ← apriltag_ros 动态 TF (pose_estimation_method: pnp)
```

`tag_visualizer_node` 另发 `/apriltag/pose`(PoseArray)、`/apriltag/distance`(Float64)、`/apriltag/markers`(MarkerArray) 作为 apriltag_ros TF 的独立交叉校验。

`ar_object_node` 发 `/apriltag/ar_marker`，其 `frame_id` = `tag_0`，
所以虚拟物体在 RViz 里由 TF 自动跟随标签。

## 测量路径 vs 显示路径

两个消费节点职责不同，**滤波策略也不同**，这是有意的设计：

| | `tag_visualizer_node` | `ar_object_node` |
|---|---|---|
| 角色 | **测量**路径 | **显示**路径 |
| 输出 | `/apriltag/image_annotated`、`/apriltag/pose`、`/apriltag/distance` | `/apriltag/image_ar`、`/apriltag/ar_marker` |
| 滤波 | **不滤波** | 滤波（`PoseStabilizer`） |
| 原因 | 距离误差实验读 `/apriltag/distance`，平滑会掩盖真实逐帧精度，让报告误差虚假变好 | 抖动影响观感，需要平滑 |

## 单元测试

```bash
colcon test --packages-select apriltag_pose
colcon test-result --verbose
# -> 82 tests, 0 errors, 0 failures, 0 skipped
```

也可以直接跑 pytest（不需要先 build）：

```bash
cd ~/apriltag_pose_ws/src/apriltag_pose && python3 -m pytest test/ -q
```

| 测试文件 | 项数 | 覆盖内容 |
|---|---|---|
| `test_pose_math.py` | 22 | 角点尺寸/重排、CameraInfo 解析与有效性、solvePnP 恢复已知位姿与无效输入不崩溃、欧氏距离、m↔mm、绝对/相对误差、距离统计、Rodrigues、四元数归一化与一致性 |
| `test_calibration_file.py` | 9 | 标定 YAML 校验（K/D/R/P 维度、fx/fy 有效性、分辨率、文件存在性） |
| `test_ar_geometry.py` | 24 | **拓展题 1.2**：物体构造、tag 系坐标不变性、近大远小、手算投影一致性、背面剔除、画家算法排序、相机背后拒绝渲染、绘制 |
| `test_pose_filter.py` | 27 | **稳定性**：四元数符号对齐与平均、PnP 二义性候选枚举与连续性择优、滤波降抖、异常帧剔除、丢检保持 |
| **合计** | **82** | |

## 文件清单

```
src/apriltag_pose/
├── apriltag_pose/
│   ├── pose_math.py               # 纯数学: solvePnP / 距离 / 统计 / 旋转
│   ├── ar_geometry.py             # 拓展题 1.2: 物体模型 + 投影 + 可见性 + 绘制
│   ├── pose_filter.py             # 稳定性: 滤波 + 二义性 + 异常剔除 + 保持
│   ├── calibration_io.py          # 标定 YAML 读取与校验
│   ├── tag_visualizer_node.py     # 普通题主节点(测量路径, 不滤波)
│   ├── ar_object_node.py          # 拓展题 1.2 节点(显示路径, 带滤波)
│   ├── distance_recorder_node.py  # 距离实验采集
│   └── calibration_checker_node.py
├── launch/
│   ├── apriltag_pose.launch.py    # 一键全系统
│   ├── calibration.launch.py      # 内参标定
│   └── camera_only.launch.py      # 仅相机(排查用)
├── config/
│   ├── camera.yaml                # 内参(待标定覆盖)
│   ├── apriltag.yaml              # 标签族/ID/尺寸/检测器
│   ├── system.yaml                # 距离实验参数
│   └── apriltag_pose.rviz
├── scripts/
│   ├── make_print_assets.py       # 生成可打印 PDF + 几何自校验
│   ├── verify_ar_transform.py     # 1.2 离线闭环验证
│   ├── analyze_distance_results.py
│   ├── detect_video_devices.sh / validate_camera.sh / run_distance_test.sh
├── test_tools/                    # 无硬件调试辅助(非单元测试)
│   ├── fake_camera.py             # 合成 tag 画面发布器
│   └── drive_ar_node.py           # 直接驱动 ar_object_node 的 ROS 接口
├── test/                          # 82 项单元测试
├── README.md
└── STABILITY.md                   # 稳定性改进 + 全部实测数据
```
