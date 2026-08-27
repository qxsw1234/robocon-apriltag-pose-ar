# 稳定性改进记录 (阶段 5)

本文件记录针对"过程中遇到的问题"所做的改进、原因、以及**实测**数据。
所有数字都是在本机 (Ubuntu 22.04 + ROS 2 Humble, 外置 USB 2.0 Camera,
640x480) 实际测出来的, 不是估计值。

---

## 1. 摄像头帧率: rgb8 -> mono8, 8 Hz -> 30 Hz

### 问题
`v4l2_camera` 默认按 `output_encoding:=rgb8` 输出时, 日志里有:
```
Image encoding not the same as requested output, performing possibly slow
conversion: yuv422_yuy2 => rgb8
```
`/camera/image_raw` 实测只有 **8.2 Hz**, 远低于摄像头的 30 fps 能力。

### 排查
`v4l2-ctl --list-formats-ext` 显示该摄像头支持:
- `YUYV`  640x480 @ 30 fps
- `MJPG`  640x480 @ 120 fps

试过三种组合 (每次单独占用设备, 避免互相干扰):

| pixel_format | output_encoding | 实测帧率 | 结论 |
|---|---|---|---|
| YUYV | rgb8 | **8.2 Hz** | 色彩转换是瓶颈 |
| MJPG | rgb8 | **崩溃** | 见下 |
| YUYV | mono8 | **29.8 Hz** | 采纳为默认 |

MJPG 在 Humble 的 `v4l2_camera` 0.6.2 上不可用:
```
Current pixel format is not supported yet: MJPG 1196444237
terminate called after throwing an instance of 'cv_bridge::Exception'
  what():  Unrecognized image encoding []
[ros2run]: Aborted
```
所以**不要**切到 MJPG, 尽管摄像头硬件支持且标称帧率更高。

### 改动
三个 launch 都新增 `output_encoding` 参数:
- `apriltag_pose.launch.py` / `camera_only.launch.py` 默认 **mono8**
- `calibration.launch.py` 默认保持 **rgb8** (标定是一次性流程, 帧率无关,
  保持彩色以最大化 `cameracalibrator` 兼容性)

### 为什么 mono8 不损失任何东西
1. AprilTag 检测本身就是在灰度图上做的, 彩色信息用不到;
2. `cv_bridge` 的 `imgmsg_to_cv2(desired_encoding='bgr8')` 会把 mono8
   自动提升为 3 通道 BGR (已实测: `mono8 -> shape=(480,640,3)`),
   所以图像标注仍然可以用彩色画角点/坐标轴/文字。

需要真彩色图时传 `output_encoding:=rgb8` 即可。

### 实测结果 (隔离 ROS_DOMAIN_ID, 排除其他项目干扰)
```
/camera/image_raw            29.829 Hz
/apriltag/detections         29.820 Hz   <- 检测跑满帧率
/apriltag/image_annotated    11.650 Hz
/apriltag/image_ar           12.535 Hz
```
**检测链路满帧 30 Hz** (位姿精度只取决于这个);
两个标注图像发布节点各约 12 Hz, 是 Python 侧 `cv2_to_imgmsg` + 发布
640x480 图像的开销, 对肉眼观看完全够用。

---

## 2. 位姿抖动: 滑动窗口滤波

### 问题
逐帧独立解算的位姿会因角点亚像素噪声而抖动, 深度方向 (tz) 最明显,
表现为 AR 物体轻微"呼吸"。

### 改动
新增 `apriltag_pose/pose_filter.py` 的 `PoseStabilizer`:
- 平移: 滑动窗口**中值** (默认, 抗离群) 或均值 (更平滑), `filter_window` 可调
- 旋转: **不能直接平均四元数** —— q 与 -q 表示同一旋转, 直接平均会互相抵消。
  必须先做符号对齐 (`align_quaternion_sign`) 再平均并归一化。
  单元测试 `test_average_of_q_and_negative_q_is_not_zero` 专门守这个陷阱。

### 实测效果
同一条含噪观测序列 (角点噪声 σ=0.4 px, 80 帧, window=7):
```
raw      tvec 标准差 (mm): [0.160, 0.178, 1.521]
filtered tvec 标准差 (mm): [0.068, 0.072, 0.432]
抖动下降倍数            : [2.35x, 2.47x, 3.52x]
均值误差 raw  0.067 mm -> filtered 0.040 mm   (滤波没有引入偏置)
```
深度方向抖动降到 1/3.5, 且精度反而略好。

---

## 3. 平面 PnP 二义性 (姿态翻转)

### 问题
标签**正对相机**时, 由平面 4 点解出的姿态存在一个近似镜像解
(绕标签平面内轴翻转)。两个解的重投影误差非常接近, 所以
"永远取重投影误差最小的解"会导致姿态在两解之间来回跳变,
AR 物体会突然翻到另一侧。

### 实测证据
用 `solvePnPGeneric(SOLVEPNP_IPPE_SQUARE)` 枚举候选解 (噪声 σ=0.3 px):

| 视角 | 候选数 | 两解重投影误差 | 两解与真值的角度差 |
|---|---|---|---|
| 正对 | 2 | 0.40 / 0.96 px | 8.28° / 8.29° | <- 几乎无法区分
| 近似正对 | 2 | 0.27 / 0.80 px | 5.34° / 6.72° |
| 轻微倾斜 | 2 | 0.08 / 1.54 px | 1.77° / 17.53° |
| 明显倾斜 | 2 | 0.13 / 1.90 px | 1.99° / 21.75° | <- 容易区分

可见越正对越难区分, 这正是翻转高发区。

### 改动
`solve_tag_pose_candidates` 枚举全部候选; `choose_candidate` 在
"误差与最优解相当"的候选中, 选**姿态与上一帧最接近**的那个。

容差同时用相对和绝对两个判据:
```
err <= best * err_ratio_tol   或   err <= best + err_abs_tol_px
```
**只用相对容差是不够的** —— 低噪声时 `best` 可能接近 0, 此时
`best * ratio` 也接近 0, 会把真正有竞争力的第二解错误地排除掉。
这个 bug 是被单元测试
`test_absolute_tolerance_matters_when_best_error_is_tiny` 抓出来的。

### 实测效果
近似正对视角, 200 帧含噪序列, 统计相邻帧姿态角变化 > 15° 的次数:
```
永远取误差最小解 :  2 次翻转
基于连续性择优   :  0 次翻转
平均角度误差     :  5.82° vs 5.96°  (代价可忽略)
```

---

## 4. 异常帧剔除

丢弃而不是让它污染滤波结果:
- `decision_margin` < `min_decision_margin` (默认 20.0) —— 通常是误检
- 重投影误差 > `max_reproj_error_px` (默认 4.0 px)
- 相对上次滤波结果位置突变 > `max_jump_m` (默认 0.25 m) 或
  姿态突变 > `max_jump_deg` (默认 60°)

突变判据有一个重要细节: **连续** `reset_after` (默认 5) 帧都被判为突变时,
认为这是**真实的大幅运动**而不是噪声, 于是清空窗口重新收敛。
否则相机快速移动后系统会被旧位姿永久卡死。
单元测试 `test_position_jump_is_rejected_then_accepted_after_persistence`
覆盖了这个行为。

---

## 5. 丢检保持 (防闪烁)

短暂丢检 (标签被手挡一下、运动模糊) 时, 在 `pose_hold_sec`
(默认 0.25 s) 内沿用上次位姿, 避免 AR 物体闪烁消失;
超时后停止绘制。画面上会显示 `POSE HELD (detection dropped)` 提示,
这样"保持"是可见的, 不会伪装成真实检测。

---

## 6. 测量路径故意不滤波

`tag_visualizer_node` **不做**滤波, 因为它是**测量路径**:
距离误差实验 (普通题第 11 项) 读的是 `/apriltag/distance`,
在这里平滑会掩盖系统真实的逐帧精度, 让报告出来的误差比实际更好看。

抖动抑制只做在 `ar_object_node` (**显示路径**)。
两个节点的职责分离已写进各自的模块 docstring。

---

## 7. 其他

- `use_k230` launch 参数已删除: 声明了但从未被读取, 且本题未使用 K230。
- 旧包 `apriltag_camera_pose` 已删除 (删除前打包备份到
  `apriltag_camera_pose_backup_20260803.tar.gz`), 避免两个功能重叠的包
  让考核方困惑。
- 包名 `k230_apriltag_pose` -> `apriltag_pose`, 与实际硬件和题目一致。

---

## 单元测试覆盖

```
82 tests, 0 errors, 0 failures, 0 skipped
```
其中本阶段新增:
- `test_pose_filter.py`  27 项 (四元数符号/平均、PnP 二义性、滤波降抖、
  异常帧剔除、丢检保持)
- `test_ar_geometry.py`  24 项 (见拓展题 1.2)
