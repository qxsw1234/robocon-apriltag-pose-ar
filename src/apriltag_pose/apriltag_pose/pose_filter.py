"""位姿稳定性: 滑动窗口滤波 + 异常帧剔除 + 平面 PnP 二义性处理.

纯数学, 不依赖 ROS, 便于单元测试.

解决的三个实际问题
------------------
1. **抖动**: 逐帧独立解算的位姿会因角点亚像素噪声而抖动 (尤其是姿态角).
   对平移做滑动窗口均值/中值; 对旋转不能直接平均四元数 ——
   必须先做【符号对齐】(q 和 -q 表示同一旋转), 否则平均会互相抵消.

2. **异常帧**: decision_margin 偏低或重投影误差偏大的帧通常是误检/严重倾斜,
   直接丢弃比让它污染滤波结果更好.

3. **平面 PnP 二义性**: 正对标签时, 由平面 4 点解出的姿态存在一个近似镜像解
   (绕标签平面内轴翻转), 表现为姿态在两个解之间反复跳变。
   cv2.solvePnPGeneric 会返回多个候选解, 我们用重投影误差 + 与上一帧的
   连续性共同择优, 显著减少翻转。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from apriltag_pose.pose_math import (
    apriltag_corners_to_opencv_ippe_order, is_camera_info_valid,
    tag_object_points_opencv_ippe_square,
)


# ---------------------------------------------------------------------------
# 四元数工具 (滤波需要, 与 pose_math 的单次转换互补)
# ---------------------------------------------------------------------------

def quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> 四元数 (x,y,z,w), 返回 np.ndarray."""
    from apriltag_pose.pose_math import rotation_matrix_to_quaternion
    return np.array(rotation_matrix_to_quaternion(R), dtype=np.float64)


def quaternion_from_rvec(rvec: np.ndarray) -> np.ndarray:
    """旋转向量 -> 四元数 (x,y,z,w)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return quaternion_from_matrix(R)


def rvec_from_quaternion(q: Sequence[float]) -> np.ndarray:
    """四元数 (x,y,z,w) -> 旋转向量 (3,1)."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError('cannot convert a zero quaternion')
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    return rvec


def align_quaternion_sign(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """把 q 的符号对齐到 reference.

    q 与 -q 表示同一个旋转. 平均一组四元数前必须统一符号,
    否则 q 和 -q 会互相抵消, 得到接近零的无效结果。
    """
    q = np.asarray(q, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return -q if float(np.dot(q, reference)) < 0.0 else q


def average_quaternions(quats: Sequence[np.ndarray]) -> np.ndarray:
    """一组四元数的平均 (先按第一个的符号对齐, 再归一化均值).

    对小角度差异这是球面均值的良好近似, 且比特征值法便宜得多。
    """
    if len(quats) == 0:
        raise ValueError('no quaternions to average')
    ref = np.asarray(quats[0], dtype=np.float64)
    acc = np.zeros(4, dtype=np.float64)
    for q in quats:
        acc += align_quaternion_sign(q, ref)
    n = np.linalg.norm(acc)
    if n == 0.0:
        # 理论上不该发生 (符号已对齐), 兜底返回参考值
        return ref / np.linalg.norm(ref)
    return acc / n


def quaternion_angle_deg(q1: Sequence[float], q2: Sequence[float]) -> float:
    """两个四元数之间的旋转角(度), 已处理 q/-q 等价."""
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = abs(float(np.dot(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


# ---------------------------------------------------------------------------
# 平面 PnP 二义性: 在多个候选解中择优
# ---------------------------------------------------------------------------

@dataclass
class PoseCandidate:
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error_px: float


def solve_tag_pose_candidates(corners_apriltag_xy: np.ndarray,
                              tag_size_m: float,
                              camera_matrix: np.ndarray,
                              dist_coeffs: Optional[np.ndarray]
                              ) -> List[PoseCandidate]:
    """用 solvePnPGeneric + IPPE_SQUARE 返回所有候选解, 按重投影误差升序.

    平面 4 点问题一般有 2 个解; 正对标签时两解的重投影误差非常接近,
    这正是姿态跳变的根源。返回全部候选交给上层用时序连续性择优。
    """
    if not is_camera_info_valid(camera_matrix):
        return []
    obj = tag_object_points_opencv_ippe_square(tag_size_m)
    img = apriltag_corners_to_opencv_ippe_order(corners_apriltag_xy).astype(np.float64)
    dist = np.zeros((5,), dtype=np.float64) if dist_coeffs is None \
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

    try:
        n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, camera_matrix, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return []
    if not n_sol:
        return []

    out: List[PoseCandidate] = []
    for i in range(n_sol):
        rvec = np.asarray(rvecs[i], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvecs[i], dtype=np.float64).reshape(3, 1)
        if tvec[2, 0] <= 0:            # 标签必须在相机前方
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist)
        d = img.reshape(-1, 2) - proj.reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
        out.append(PoseCandidate(rvec=rvec, tvec=tvec,
                                 reprojection_error_px=err))
    out.sort(key=lambda c: c.reprojection_error_px)
    return out


def choose_candidate(candidates: Sequence[PoseCandidate],
                     previous_quat: Optional[np.ndarray],
                     err_ratio_tol: float = 1.5,
                     err_abs_tol_px: float = 0.75,
                     max_jump_deg: float = 45.0) -> Optional[PoseCandidate]:
    """在候选解中择优.

    规则:
      * 只有一个候选 -> 直接用.
      * 多个候选且没有历史 -> 用重投影误差最小的.
      * 多个候选且有历史 -> 在"误差与最优解相当"的候选里选姿态与上一帧
        最接近的。"相当"同时看相对和绝对容差:
            err <= best * err_ratio_tol   或   err <= best + err_abs_tol_px
        只用相对容差是不够的: 合成/低噪场景下 best 可能接近 0,
        此时 best * ratio 也接近 0, 会把真正有竞争力的第二解排除掉。
        绝对容差保证近似正对(两解误差都很小且接近)时二义性仍能被处理。

    max_jump_deg 目前仅作为记录用的阈值上限传入; 真正的突变抑制在
    PoseStabilizer 里做, 因为那里才有滤波历史。
    """
    if not candidates:
        return None
    if len(candidates) == 1 or previous_quat is None:
        return candidates[0]

    best_err = candidates[0].reprojection_error_px
    limit = max(best_err * err_ratio_tol, best_err + err_abs_tol_px)
    viable = [c for c in candidates if c.reprojection_error_px <= limit]
    if not viable:
        return candidates[0]

    scored = [(quaternion_angle_deg(quaternion_from_rvec(c.rvec), previous_quat), c)
              for c in viable]
    scored.sort(key=lambda t: t[0])
    return scored[0][1]


# ---------------------------------------------------------------------------
# 滑动窗口滤波器
# ---------------------------------------------------------------------------

@dataclass
class FilteredPose:
    rvec: np.ndarray
    tvec: np.ndarray
    quat: np.ndarray                # (x,y,z,w)
    samples: int                    # 参与滤波的样本数
    rejected: int                   # 累计被剔除的帧数
    held: bool                      # 本次是否是"丢检保持"的结果


class PoseStabilizer:
    """位姿稳定器: 剔除异常帧 -> 二义性择优 -> 滑动窗口滤波 -> 丢检保持.

    典型用法 (每帧):
        out = stab.update(corners, margin, K, dist, now_sec)
        if out is not None: 用 out.rvec / out.tvec 去绘制

    参数
    ----
    window            : 滑动窗口长度 (帧). 1 = 不滤波
    use_median_translation : True 用中值(抗离群), False 用均值(更平滑)
    min_decision_margin : 低于此值的检测直接丢弃
    max_reproj_error_px : 重投影误差超过此值的解直接丢弃
    max_jump_m / max_jump_deg : 与上一次滤波结果相比, 突变超过阈值的帧
                        视为异常并丢弃 (连续丢弃达 reset_after 帧后重置窗口,
                        以便真实的大幅运动最终能被跟上)
    hold_sec          : 丢检后沿用上次结果的最长时间
    """

    def __init__(self,
                 tag_size_m: float,
                 window: int = 5,
                 use_median_translation: bool = True,
                 min_decision_margin: float = 20.0,
                 max_reproj_error_px: float = 4.0,
                 max_jump_m: float = 0.25,
                 max_jump_deg: float = 60.0,
                 reset_after: int = 5,
                 hold_sec: float = 0.25) -> None:
        if tag_size_m <= 0:
            raise ValueError(f'tag_size_m must be > 0, got {tag_size_m}')
        if window < 1:
            raise ValueError(f'window must be >= 1, got {window}')
        self.tag_size_m = float(tag_size_m)
        self.window = int(window)
        self.use_median = bool(use_median_translation)
        self.min_margin = float(min_decision_margin)
        self.max_reproj = float(max_reproj_error_px)
        self.max_jump_m = float(max_jump_m)
        self.max_jump_deg = float(max_jump_deg)
        self.reset_after = int(reset_after)
        self.hold_sec = float(hold_sec)

        self._tvecs: Deque[np.ndarray] = deque(maxlen=self.window)
        self._quats: Deque[np.ndarray] = deque(maxlen=self.window)
        self._last: Optional[FilteredPose] = None
        self._last_t = 0.0
        self._consecutive_rejects = 0
        self.rejected_total = 0

    # ------------------------------------------------------------------
    @property
    def last(self) -> Optional[FilteredPose]:
        return self._last

    def reset(self) -> None:
        self._tvecs.clear()
        self._quats.clear()
        self._consecutive_rejects = 0

    # ------------------------------------------------------------------
    def update(self,
               corners_apriltag_xy: Optional[np.ndarray],
               decision_margin: Optional[float],
               camera_matrix: np.ndarray,
               dist_coeffs: Optional[np.ndarray],
               now_sec: float) -> Optional[FilteredPose]:
        """喂一帧. 返回滤波后的位姿, 或 None (无可用位姿).

        corners 为 None 表示这一帧没检测到目标标签 -> 走丢检保持逻辑。
        """
        if corners_apriltag_xy is None:
            return self._hold(now_sec)

        if decision_margin is not None and float(decision_margin) < self.min_margin:
            self.rejected_total += 1
            return self._hold(now_sec)

        candidates = solve_tag_pose_candidates(
            corners_apriltag_xy, self.tag_size_m, camera_matrix, dist_coeffs)
        candidates = [c for c in candidates
                      if c.reprojection_error_px <= self.max_reproj]
        if not candidates:
            self.rejected_total += 1
            return self._hold(now_sec)

        prev_quat = self._last.quat if self._last is not None else None
        chosen = choose_candidate(candidates, prev_quat,
                                  max_jump_deg=self.max_jump_deg)
        if chosen is None:
            self.rejected_total += 1
            return self._hold(now_sec)

        quat = quaternion_from_rvec(chosen.rvec)
        tvec = chosen.tvec.reshape(3)

        # ---- 突变剔除 (相对上一次滤波输出)
        if self._last is not None:
            d_pos = float(np.linalg.norm(tvec - self._last.tvec.reshape(3)))
            d_ang = quaternion_angle_deg(quat, self._last.quat)
            if d_pos > self.max_jump_m or d_ang > self.max_jump_deg:
                self._consecutive_rejects += 1
                self.rejected_total += 1
                if self._consecutive_rejects < self.reset_after:
                    return self._hold(now_sec)
                # 连续异常太多 -> 认为是真实的大幅运动, 清空窗口重新收敛
                self.reset()

        self._consecutive_rejects = 0

        # ---- 滑动窗口滤波
        self._tvecs.append(tvec)
        ref = self._quats[0] if self._quats else quat
        self._quats.append(align_quaternion_sign(quat, ref))

        arr = np.asarray(self._tvecs, dtype=np.float64)
        t_filt = np.median(arr, axis=0) if self.use_median else arr.mean(axis=0)
        q_filt = average_quaternions(list(self._quats))

        out = FilteredPose(
            rvec=rvec_from_quaternion(q_filt),
            tvec=t_filt.reshape(3, 1),
            quat=q_filt,
            samples=len(self._tvecs),
            rejected=self.rejected_total,
            held=False,
        )
        self._last = out
        self._last_t = float(now_sec)
        return out

    # ------------------------------------------------------------------
    def _hold(self, now_sec: float) -> Optional[FilteredPose]:
        """丢检保持: 短时间内沿用上次结果, 避免物体闪烁消失."""
        if self._last is None:
            return None
        if (float(now_sec) - self._last_t) > self.hold_sec:
            return None
        held = FilteredPose(rvec=self._last.rvec, tvec=self._last.tvec,
                            quat=self._last.quat, samples=self._last.samples,
                            rejected=self.rejected_total, held=True)
        return held
