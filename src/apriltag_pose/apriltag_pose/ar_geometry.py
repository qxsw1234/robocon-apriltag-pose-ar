"""AR geometry: 在 AprilTag 坐标系(世界系)中定义固定的虚拟三维物体并投影到图像.

拓展题 1.2 的核心数学. 和 pose_math 一样, 这里不碰 ROS, 便于单元测试.

坐标系约定
----------
AprilTag 自身坐标系 (= 本题的"世界坐标系", 与 pose_math 一致):
    +x -> 标签右, +y -> 标签上, +z -> 垂直标签表面射出(朝向观察者)
solvePnP 给出的 (rvec, tvec) 描述的正是 tag 系 -> 相机光学系 的刚体变换:
    X_cam = R(rvec) @ X_tag + tvec
所以只要把物体顶点写成 tag 系坐标, 直接喂给 cv2.projectPoints(rvec, tvec)
就能得到正确的透视投影 —— 物体自然"钉死"在 tag 建立的世界系里,
相机怎么动, 物体在世界系中的坐标都不变, 只有投影结果变。

为什么物体要有 z 方向偏移
------------------------
题目要求虚拟物体不与标签重合. 默认把物体沿 tag 法线 +z 抬升,
使它"悬浮"在标签前方, 这样透视关系(近大远小、遮挡顺序)最直观。

渲染
----
* 背面剔除 (backface culling): 面法线在相机系中指向背离相机的面不画。
* 画家算法 (painter's algorithm): 剩下的面按"面心到相机的深度"从远到近
  依次绘制, 近处的面自然覆盖远处的面, 得到正确的遮挡关系。
这两步都在相机坐标系中做, 因为可见性取决于相机而不是世界系。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 物体模型 (顶点在 tag 系, 单位米)
# ---------------------------------------------------------------------------


@dataclass
class Mesh:
    """一个简单的多面体.

    vertices : (N,3) tag 系顶点坐标, 单位米
    faces    : 每个面的顶点索引, 逆时针(从面外侧看)排列, 用于法线朝外
    edges    : 需要描边的顶点索引对
    face_colors : 每个面的 BGR 颜色
    """
    vertices: np.ndarray
    faces: List[Tuple[int, ...]]
    edges: List[Tuple[int, int]]
    face_colors: List[Tuple[int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f'vertices must be (N,3), got {self.vertices.shape}')
        n = len(self.vertices)
        for f in self.faces:
            if len(f) < 3:
                raise ValueError(f'face needs >= 3 indices, got {f}')
            if any(i < 0 or i >= n for i in f):
                raise ValueError(f'face index out of range in {f} (N={n})')
        for a, b in self.edges:
            if not (0 <= a < n and 0 <= b < n):
                raise ValueError(f'edge index out of range: ({a},{b})')
        if not self.face_colors:
            self.face_colors = [(0, 200, 255)] * len(self.faces)
        elif len(self.face_colors) != len(self.faces):
            raise ValueError('face_colors length must match faces length')


def make_cube(size_m: float,
              offset_m: Sequence[float] = (0.0, 0.0, 0.0)) -> Mesh:
    """轴对齐立方体, 中心位于 offset_m (tag 系), 边长 size_m.

    faces 的顶点顺序保证法线朝立方体外部。
    """
    if size_m <= 0:
        raise ValueError(f'size_m must be > 0, got {size_m}')
    h = size_m / 2.0
    ox, oy, oz = (float(v) for v in offset_m)
    # 0..3 = z-  (后面), 4..7 = z+ (前面); 每层逆时针(俯视)
    v = np.array([
        [-h, -h, -h], [+h, -h, -h], [+h, +h, -h], [-h, +h, -h],   # z-
        [-h, -h, +h], [+h, -h, +h], [+h, +h, +h], [-h, +h, +h],   # z+
    ], dtype=np.float64)
    v += np.array([ox, oy, oz], dtype=np.float64)

    # 每个面从外侧看逆时针 -> 叉乘得到朝外法线
    faces = [
        (0, 3, 2, 1),   # z- 后面
        (4, 5, 6, 7),   # z+ 前面
        (0, 1, 5, 4),   # y- 下面
        (2, 3, 7, 6),   # y+ 上面
        (0, 4, 7, 3),   # x- 左面
        (1, 2, 6, 5),   # x+ 右面
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    colors = [
        (60, 60, 200),     # z- 红
        (60, 200, 60),     # z+ 绿
        (200, 160, 60),    # y- 蓝
        (60, 200, 200),    # y+ 黄
        (200, 60, 200),    # x- 品红
        (200, 200, 80),    # x+ 青
    ]
    return Mesh(vertices=v, faces=faces, edges=edges, face_colors=colors)


def make_pyramid(size_m: float,
                 offset_m: Sequence[float] = (0.0, 0.0, 0.0)) -> Mesh:
    """正四面体形状的三棱锥: 正方形底面 + 顶点, 底面中心在 offset_m."""
    if size_m <= 0:
        raise ValueError(f'size_m must be > 0, got {size_m}')
    h = size_m / 2.0
    ox, oy, oz = (float(v) for v in offset_m)
    v = np.array([
        [-h, -h, 0.0], [+h, -h, 0.0], [+h, +h, 0.0], [-h, +h, 0.0],  # 底面
        [0.0, 0.0, size_m],                                           # 顶点
    ], dtype=np.float64)
    v += np.array([ox, oy, oz], dtype=np.float64)
    faces = [
        (0, 3, 2, 1),      # 底面, 法线朝 -z (朝外)
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (0, 4), (1, 4), (2, 4), (3, 4)]
    colors = [(80, 80, 80), (60, 60, 220), (60, 220, 60),
              (220, 160, 60), (60, 220, 220)]
    return Mesh(vertices=v, faces=faces, edges=edges, face_colors=colors)


def make_arrow(length_m: float,
               offset_m: Sequence[float] = (0.0, 0.0, 0.0),
               shaft_ratio: float = 0.12) -> Mesh:
    """沿 tag 系 +z 指向的三维箭头 (方柱杆 + 四棱锥头)."""
    if length_m <= 0:
        raise ValueError(f'length_m must be > 0, got {length_m}')
    ox, oy, oz = (float(v) for v in offset_m)
    r = max(1e-6, length_m * shaft_ratio)       # 杆半宽
    shaft_len = length_m * 0.7
    head_r = r * 2.2
    v = np.array([
        # 杆: 底面 0..3, 顶面 4..7
        [-r, -r, 0.0], [+r, -r, 0.0], [+r, +r, 0.0], [-r, +r, 0.0],
        [-r, -r, shaft_len], [+r, -r, shaft_len],
        [+r, +r, shaft_len], [-r, +r, shaft_len],
        # 箭头底面 8..11 + 尖端 12
        [-head_r, -head_r, shaft_len], [+head_r, -head_r, shaft_len],
        [+head_r, +head_r, shaft_len], [-head_r, +head_r, shaft_len],
        [0.0, 0.0, length_m],
    ], dtype=np.float64)
    v += np.array([ox, oy, oz], dtype=np.float64)
    faces = [
        (0, 3, 2, 1),                              # 杆底
        (0, 1, 5, 4), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 0, 4, 7),                # 杆侧
        (8, 11, 10, 9),                            # 箭头底
        (8, 9, 12), (9, 10, 12), (10, 11, 12), (11, 8, 12),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7),
             (8, 9), (9, 10), (10, 11), (11, 8),
             (8, 12), (9, 12), (10, 12), (11, 12)]
    colors = [(70, 70, 200)] * 5 + [(60, 170, 240)] * 5
    return Mesh(vertices=v, faces=faces, edges=edges, face_colors=colors)


_BUILDERS = {
    'cube': make_cube,
    'pyramid': make_pyramid,
    'arrow': make_arrow,
}


def build_object(object_type: str, size_m: float,
                 offset_m: Sequence[float]) -> Mesh:
    """按名字构造物体. object_type in {'cube','pyramid','arrow'}."""
    key = str(object_type).strip().lower()
    if key not in _BUILDERS:
        raise ValueError(f'unknown object_type {object_type!r}, '
                         f'expected one of {sorted(_BUILDERS)}')
    return _BUILDERS[key](size_m, offset_m)


# ---------------------------------------------------------------------------
# 变换与投影
# ---------------------------------------------------------------------------

def transform_points_to_camera(points_tag: np.ndarray,
                               rvec: np.ndarray,
                               tvec: np.ndarray) -> np.ndarray:
    """X_cam = R(rvec) @ X_tag + tvec.  返回 (N,3)."""
    pts = np.asarray(points_tag, dtype=np.float64).reshape(-1, 3)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    return (R @ pts.T).T + t


def project_points(points_tag: np.ndarray,
                   rvec: np.ndarray,
                   tvec: np.ndarray,
                   camera_matrix: np.ndarray,
                   dist_coeffs: Optional[np.ndarray]) -> np.ndarray:
    """把 tag 系点投影到像素平面. 返回 (N,2) float64."""
    pts = np.asarray(points_tag, dtype=np.float64).reshape(-1, 1, 3)
    dist = np.zeros((5,), dtype=np.float64) if dist_coeffs is None \
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    proj, _ = cv2.projectPoints(
        pts,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        dist)
    return proj.reshape(-1, 2)


def face_depth(vertices_cam: np.ndarray, face: Sequence[int]) -> float:
    """面心在相机系中的 z 深度 (画家算法的排序键)."""
    idx = list(face)
    return float(np.mean(vertices_cam[idx, 2]))


def is_face_visible(vertices_cam: np.ndarray, face: Sequence[int]) -> bool:
    """背面剔除: 面法线与视线方向夹角 > 90 度才可见.

    面在相机系中的法线 n = (v1-v0) x (v2-v0). 顶点按"从外侧看逆时针"排列时
    n 指向物体外部. 相机在原点, 面心到相机的方向是 -centroid.
    可见条件: dot(n, -centroid) > 0, 即 dot(n, centroid) < 0.
    """
    idx = list(face)
    v0, v1, v2 = vertices_cam[idx[0]], vertices_cam[idx[1]], vertices_cam[idx[2]]
    n = np.cross(v1 - v0, v2 - v0)
    centroid = vertices_cam[idx].mean(axis=0)
    return bool(np.dot(n, centroid) < 0.0)


def sort_faces_back_to_front(vertices_cam: np.ndarray,
                             faces: Sequence[Sequence[int]]) -> List[int]:
    """返回面索引, 按深度从远到近 (画家算法的绘制顺序)."""
    order = sorted(range(len(faces)),
                   key=lambda i: face_depth(vertices_cam, faces[i]),
                   reverse=True)
    return order


def is_in_front_of_camera(vertices_cam: np.ndarray,
                          min_depth_m: float = 1e-3) -> bool:
    """所有顶点都在相机前方才渲染.

    有顶点跑到相机背后 (z <= 0) 时 projectPoints 会给出翻转/发散的像素坐标,
    画出来是乱线, 所以整体跳过更安全。
    """
    return bool(np.all(vertices_cam[:, 2] > float(min_depth_m)))


# ---------------------------------------------------------------------------
# 绘制
# ---------------------------------------------------------------------------

def draw_mesh(image: np.ndarray,
              mesh: Mesh,
              rvec: np.ndarray,
              tvec: np.ndarray,
              camera_matrix: np.ndarray,
              dist_coeffs: Optional[np.ndarray] = None,
              fill_alpha: float = 0.45,
              edge_color: Tuple[int, int, int] = (255, 255, 255),
              edge_thickness: int = 2,
              cull_backfaces: bool = True) -> bool:
    """把 mesh 画到 image 上 (原地修改). 返回是否真的画了.

    绘制顺序: 半透明填充面(画家算法, 远->近) -> 描边.
    """
    verts_cam = transform_points_to_camera(mesh.vertices, rvec, tvec)
    if not is_in_front_of_camera(verts_cam):
        return False

    pts2d = project_points(mesh.vertices, rvec, tvec, camera_matrix, dist_coeffs)
    if not np.all(np.isfinite(pts2d)):
        return False

    h, w = image.shape[:2]
    # 完全在画面外就不用画了 (留一点余量, 部分可见的仍然画)
    if (pts2d[:, 0].max() < -w or pts2d[:, 0].min() > 2 * w or
            pts2d[:, 1].max() < -h or pts2d[:, 1].min() > 2 * h):
        return False

    pts_i = np.rint(pts2d).astype(np.int32)

    if fill_alpha > 0.0:
        overlay = image.copy()
        for fi in sort_faces_back_to_front(verts_cam, mesh.faces):
            face = mesh.faces[fi]
            if cull_backfaces and not is_face_visible(verts_cam, face):
                continue
            poly = pts_i[list(face)]
            cv2.fillConvexPoly(overlay, poly, mesh.face_colors[fi], cv2.LINE_AA)
        cv2.addWeighted(overlay, float(fill_alpha), image,
                        1.0 - float(fill_alpha), 0.0, dst=image)

    for a, b in mesh.edges:
        cv2.line(image, tuple(pts_i[a]), tuple(pts_i[b]),
                 edge_color, edge_thickness, cv2.LINE_AA)
    return True


def draw_offset_indicator(image: np.ndarray,
                          offset_m: Sequence[float],
                          rvec: np.ndarray,
                          tvec: np.ndarray,
                          camera_matrix: np.ndarray,
                          dist_coeffs: Optional[np.ndarray] = None,
                          color: Tuple[int, int, int] = (0, 255, 255)) -> None:
    """画一条从 tag 原点到物体中心的虚线, 直观展示三维偏移量."""
    pts = np.array([[0.0, 0.0, 0.0], list(offset_m)], dtype=np.float64)
    cam = transform_points_to_camera(pts, rvec, tvec)
    if not is_in_front_of_camera(cam):
        return
    p = project_points(pts, rvec, tvec, camera_matrix, dist_coeffs)
    if not np.all(np.isfinite(p)):
        return
    a, b = np.rint(p).astype(np.int32)
    # 手画虚线, OpenCV 没有内置虚线
    seg = 8
    total = float(np.hypot(*(b - a)))
    if total < 1.0:
        return
    n = max(1, int(total / seg))
    for i in range(0, n, 2):
        t0, t1 = i / n, min(1.0, (i + 1) / n)
        p0 = a + (b - a) * t0
        p1 = a + (b - a) * t1
        cv2.line(image, tuple(np.rint(p0).astype(int)),
                 tuple(np.rint(p1).astype(int)), color, 1, cv2.LINE_AA)


def verify_static_in_world(mesh: Mesh,
                           poses: Sequence[Tuple[np.ndarray, np.ndarray]]
                           ) -> Dict[str, float]:
    """验证"物体相对 tag 世界系静止"这一性质.

    对多个不同相机位姿, 物体顶点在 **tag 系** 中的坐标必须完全不变,
    而在 **相机系** 中必须随位姿改变。返回两者的最大变化量,
    供单元测试和演示时打印 (tag 系应为 0, 相机系应明显 > 0)。
    """
    if len(poses) < 2:
        raise ValueError('need at least 2 poses to compare')
    tag_pts = [mesh.vertices.copy() for _ in poses]
    cam_pts = [transform_points_to_camera(mesh.vertices, rv, tv)
               for rv, tv in poses]
    tag_spread = max(float(np.abs(tag_pts[i] - tag_pts[0]).max())
                     for i in range(1, len(poses)))
    cam_spread = max(float(np.abs(cam_pts[i] - cam_pts[0]).max())
                     for i in range(1, len(poses)))
    return {'max_tag_frame_delta_m': tag_spread,
            'max_camera_frame_delta_m': cam_spread}
