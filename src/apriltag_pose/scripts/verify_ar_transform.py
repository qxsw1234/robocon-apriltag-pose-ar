#!/usr/bin/env python3
"""verify_ar_transform.py - 离线验证拓展题 1.2 的变换关系是否正确 (不需要摄像头).

做法 (闭环, 端到端):
    1. 用 cv2.aruco 渲染一张真实的 tag36h11 ID 0 图 (带白色静默区),
       按【已知的相机位姿】做透视变换, 合成出"相机拍到的画面";
    2. 用检测器在合成图上重新检测标签, 再用 solvePnP 恢复位姿;
    3. 比较【恢复的位姿】和【真值位姿】-> 验证位姿链路精度;
    4. 把虚拟物体顶点从 tag 系变换到相机系再反算回 tag 系,
       验证在所有视角下物体在 tag(世界)系中的坐标【完全不变】;
    5. 渲染多视角对比图, 肉眼确认近大远小 / 朝向随视角变化 / 相对标签静止。

为什么这个验证有意义:
    它不依赖真实摄像头和打印件, 所以能在实机之前就抓出坐标系约定错误
    (角点顺序、tag_size 定义、旋转方向、tag系->相机系 的方向搞反等)。
    真值位姿是自己指定的, 所以误差可以定量衡量。

用法:
    python3 verify_ar_transform.py                 # 跑验证, 打印表格
    python3 verify_ar_transform.py -o out.png      # 同时存多视角对比图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# 允许直接从源码树运行 (无需先 colcon build / source)
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from apriltag_pose.ar_geometry import (                       # noqa: E402
    build_object, draw_mesh, draw_offset_indicator, is_face_visible,
    project_points, transform_points_to_camera,
)
from apriltag_pose.pose_math import solve_tag_pose            # noqa: E402

IMG_W, IMG_H = 640, 480
# 一个典型的 640x480 内参 (fx=fy=600, 主点在图像中心)
K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0],
              [0.0, 0.0, 1.0]])
ZERO_DIST = np.zeros(5)
TAG_SIZE = 0.08
OBJECT_OFFSET = (0.0, 0.06, 0.04)      # tag 系: 上移 60 mm, 沿法线外移 40 mm

# (名称, rvec, tvec) —— tag 系 -> 相机系 的真值位姿.
# rvec 绕 x 轴 ~pi 表示"标签正对相机"(tag 的 +z 指向相机).
VIEWS: List[Tuple[str, np.ndarray, np.ndarray]] = [
    ('front',   np.array([[np.pi], [0.0], [0.0]]),    np.array([[0.0], [0.0], [0.40]])),
    ('near',    np.array([[np.pi], [0.05], [0.0]]),   np.array([[0.0], [0.0], [0.25]])),
    ('far',     np.array([[np.pi], [0.05], [0.0]]),   np.array([[0.0], [0.0], [0.80]])),
    ('tilt-R',  np.array([[2.95], [0.60], [0.05]]),   np.array([[0.02], [0.0], [0.45]])),
    ('tilt-L',  np.array([[2.90], [-0.60], [-0.05]]), np.array([[-0.02], [0.0], [0.45]])),
    ('tilt-up', np.array([[2.55], [0.10], [0.0]]),    np.array([[0.0], [0.02], [0.42]])),
    ('shift-R', np.array([[np.pi], [0.0], [0.0]]),    np.array([[0.08], [0.0], [0.50]])),
    ('roll',    np.array([[2.85], [0.15], [0.70]]),   np.array([[0.01], [0.01], [0.48]])),
]


def _make_tag_sheet(cell_px: int = 50, quiet_cells: int = 2
                    ) -> Tuple[np.ndarray, float]:
    """渲染一张 tag36h11 ID0 "纸": 黑方块 + 白色静默区.

    返回 (BGR 图, 整张纸的物理边长/米). 静默区必须一起做透视变换,
    否则检测器在合成图上找不到标签。
    """
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36H11)
    side = 8 * cell_px                       # 36h11 = 8x8 单元格
    pad = quiet_cells * cell_px
    sheet = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
    sheet[pad:pad + side, pad:pad + side] = cv2.aruco.drawMarker(
        dictionary, 0, side)
    sheet_m = (side + 2 * pad) / side * TAG_SIZE
    return cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR), sheet_m


def render_view(sheet: np.ndarray, sheet_m: float,
                rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """按给定位姿把 tag 纸投影成"相机画面"."""
    s = sheet_m / 2.0
    # tag 系 4 角 (TL,TR,BR,BL), 与 sheet 图像 4 角一一对应
    obj = np.array([[-s, +s, 0.0], [+s, +s, 0.0],
                    [+s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64)
    dst = project_points(obj, rvec, tvec, K, None).astype(np.float32)
    n = sheet.shape[0]
    src = np.array([[0, 0], [n, 0], [n, n], [0, n]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((IMG_H, IMG_W, 3), 200, np.uint8)   # 灰色背景
    return cv2.warpPerspective(sheet, homography, (IMG_W, IMG_H), dst=canvas,
                               borderMode=cv2.BORDER_TRANSPARENT)


def detect_and_solve(image: np.ndarray):
    """在图上检测 tag36h11 ID0 并解算位姿. 返回 PoseResult 或 None."""
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36H11)
    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
    if ids is None or 0 not in ids.ravel():
        return None
    quad = corners[int(np.where(ids.ravel() == 0)[0][0])].reshape(4, 2)
    # cv2.aruco 返回 TL,TR,BR,BL; pose_math 期望 apriltag 顺序 BL,BR,TR,TL
    apriltag_order = quad[[3, 2, 1, 0], :]
    return solve_tag_pose(apriltag_order, TAG_SIZE, K, ZERO_DIST)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('-o', '--output-image', type=Path, default=None,
                    help='save a multi-view comparison image here')
    ap.add_argument('--object-type', default='cube',
                    choices=('cube', 'pyramid', 'arrow'))
    ap.add_argument('--pos-tol-mm', type=float, default=10.0,
                    help='max allowed position error (mm)')
    ap.add_argument('--static-tol-mm', type=float, default=1e-6,
                    help='max allowed drift of the object in the TAG frame (mm)')
    args = ap.parse_args()

    sheet, sheet_m = _make_tag_sheet()
    mesh = build_object(args.object_type, TAG_SIZE * 0.5, OBJECT_OFFSET)
    expected_centre = np.asarray(OBJECT_OFFSET, dtype=np.float64)

    print(f'AR transform verification  (object={args.object_type}, '
          f'tag_size={TAG_SIZE * 1000:.0f} mm, '
          f'offset in tag frame={tuple(np.round(np.array(OBJECT_OFFSET) * 1000, 1))} mm)')
    print()
    header = (f'{"view":<9}{"true tvec (m)":<26}{"solved tvec (m)":<26}'
              f'{"pos err":<10}{"reproj":<9}{"faces":<7}'
              f'{"object centre in TAG frame (m)"}')
    print(header)
    print('-' * len(header))

    images = []
    failures: List[str] = []
    max_pos_err_mm = 0.0
    max_static_err_mm = 0.0

    for name, rvec_true, tvec_true in VIEWS:
        img = render_view(sheet, sheet_m, rvec_true, tvec_true)
        pose = detect_and_solve(img)
        if pose is None:
            failures.append(f'{name}: tag not detected / PnP failed')
            print(f'{name:<9}{"-":<26}{"DETECT FAILED":<26}')
            continue

        # ---- 位姿精度
        pos_err_mm = float(np.linalg.norm(
            pose.tvec.ravel() - tvec_true.ravel()) * 1000.0)
        max_pos_err_mm = max(max_pos_err_mm, pos_err_mm)
        if pos_err_mm > args.pos_tol_mm:
            failures.append(f'{name}: position error {pos_err_mm:.2f} mm '
                            f'> {args.pos_tol_mm} mm')

        # ---- 核心: 物体在 tag(世界)系中的坐标必须不变
        # 把顶点送到相机系, 再用恢复出的位姿反变换回 tag 系.
        R, _ = cv2.Rodrigues(pose.rvec)
        verts_cam = transform_points_to_camera(mesh.vertices, pose.rvec, pose.tvec)
        centre_cam = verts_cam.mean(axis=0)
        centre_tag = R.T @ (centre_cam - pose.tvec.ravel())
        static_err_mm = float(np.abs(centre_tag - expected_centre).max() * 1000.0)
        max_static_err_mm = max(max_static_err_mm, static_err_mm)
        if static_err_mm > args.static_tol_mm:
            failures.append(f'{name}: object drifted {static_err_mm:.6f} mm in '
                            f'the TAG frame (must be ~0)')

        n_visible = sum(1 for f in mesh.faces if is_face_visible(verts_cam, f))

        print(f'{name:<9}'
              f'{str(np.round(tvec_true.ravel(), 4)):<26}'
              f'{str(np.round(pose.tvec.ravel(), 4)):<26}'
              f'{pos_err_mm:>6.2f} mm  '
              f'{pose.reprojection_error_px:>6.2f}px  '
              f'{n_visible:<7}'
              f'{np.round(centre_tag, 6)}')

        # ---- 渲染叠加图
        cv2.drawFrameAxes(img, K, ZERO_DIST, pose.rvec, pose.tvec,
                          TAG_SIZE * 0.5, 2)
        draw_offset_indicator(img, OBJECT_OFFSET, pose.rvec, pose.tvec, K,
                              ZERO_DIST)
        draw_mesh(img, mesh, pose.rvec, pose.tvec, K, ZERO_DIST)
        cv2.putText(img, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 200), 2, cv2.LINE_AA)
        images.append(img)

    # ---- 近大远小定量检查
    print()
    areas = {}
    for name in ('near', 'front', 'far'):
        rvec, tvec = next((r, t) for n, r, t in VIEWS if n == name)
        p = project_points(mesh.vertices, rvec, tvec, K, None)
        areas[name] = ((p[:, 0].max() - p[:, 0].min()) *
                       (p[:, 1].max() - p[:, 1].min()))
    print(f'perspective check (projected bbox area, px^2): '
          f'near(0.25m)={areas["near"]:.0f} > front(0.40m)={areas["front"]:.0f} '
          f'> far(0.80m)={areas["far"]:.0f}')
    if not (areas['near'] > areas['front'] > areas['far']):
        failures.append('perspective check failed: object does not shrink with '
                        'distance')
    else:
        print('  -> object shrinks as the camera moves away: OK')

    if args.output_image and images:
        cols = 4
        rows = [np.hstack(images[i:i + cols])
                for i in range(0, len(images), cols)]
        width = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)),
                       constant_values=200) for r in rows]
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output_image), np.vstack(rows))
        print(f'\nsaved multi-view comparison -> {args.output_image}')

    print()
    print(f'max position error         : {max_pos_err_mm:.3f} mm '
          f'(tolerance {args.pos_tol_mm} mm)')
    print(f'max drift in TAG frame     : {max_static_err_mm:.9f} mm '
          f'(must be ~0 -> object is STATIC in the AprilTag world frame)')

    if failures:
        print(f'\n[FAIL] {len(failures)} problem(s):')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('\n[PASS] all views: pose recovered within tolerance, virtual object '
          'stays fixed in the AprilTag world frame, perspective is correct.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
