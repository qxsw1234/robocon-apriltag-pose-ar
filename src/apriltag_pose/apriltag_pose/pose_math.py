"""Pure-math helpers: PnP for AprilTag, distance, error statistics, rotations.

Nothing here talks to ROS - this keeps it trivially unit-testable.

Coordinate conventions
----------------------
Camera optical frame (ROS/REP-103, matches OpenCV image convention):
    +x -> image right
    +y -> image down
    +z -> into the scene (camera-forward)
With this convention the solvePnP ``tvec`` lands directly in the optical
frame and ``tz`` is the tag depth (positive, in front of the camera).

AprilTag 3D corners in the tag's own frame (libapriltag convention):
    +x -> tag right, +y -> tag up, +z -> out of the tag surface (toward viewer)
The apriltag_msgs ``corners`` follow the underlying C library order,
counter-clockwise starting from bottom-left *in the tag frame*:
    corners[0] = (-s/2, -s/2, 0)   # bottom-left  (tag frame)
    corners[1] = (+s/2, -s/2, 0)   # bottom-right (tag frame)
    corners[2] = (+s/2, +s/2, 0)   # top-right    (tag frame)
    corners[3] = (-s/2, +s/2, 0)   # top-left     (tag frame)
Because the image y-axis points down, in *image pixel* space these read as
[BL, BR, TR, TL] (clockwise).

OpenCV cv2.SOLVEPNP_IPPE_SQUARE requires a *specific* ordering of the four
model points: TL, TR, BR, BL. We build that ordering by permuting the
apriltag corners ([3,2,1,0]), and always run a reprojection-error check
afterwards. If the check fails we fall back to SOLVEPNP_ITERATIVE which is
ordering-tolerant. The method actually used is recorded on the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 3-D model points
# ---------------------------------------------------------------------------

def tag_object_points_apriltag_order(tag_size_m: float) -> np.ndarray:
    """4 corner points in APRILTAG's native (CCW-from-BL, tag frame) order."""
    if tag_size_m <= 0:
        raise ValueError(f'tag_size_m must be > 0, got {tag_size_m}')
    s = tag_size_m / 2.0
    return np.array([
        [-s, -s, 0.0],   # 0 bottom-left
        [+s, -s, 0.0],   # 1 bottom-right
        [+s, +s, 0.0],   # 2 top-right
        [-s, +s, 0.0],   # 3 top-left
    ], dtype=np.float64)


def tag_object_points_opencv_ippe_square(tag_size_m: float) -> np.ndarray:
    """4 corner points in the order required by cv2.SOLVEPNP_IPPE_SQUARE.

    OpenCV docs require: TL, TR, BR, BL (in image pixel space, y-down).
    """
    if tag_size_m <= 0:
        raise ValueError(f'tag_size_m must be > 0, got {tag_size_m}')
    s = tag_size_m / 2.0
    return np.array([
        [-s, +s, 0.0],   # 0 top-left
        [+s, +s, 0.0],   # 1 top-right
        [+s, -s, 0.0],   # 2 bottom-right
        [-s, -s, 0.0],   # 3 bottom-left
    ], dtype=np.float64)


def apriltag_corners_to_opencv_ippe_order(corners_apriltag: np.ndarray) -> np.ndarray:
    """Permute (4,2) apriltag corners into IPPE_SQUARE order (TL,TR,BR,BL).

    apriltag order = [BL, BR, TR, TL]  ->  opencv order = [TL, TR, BR, BL]
    i.e. take indices [3, 2, 1, 0].
    """
    corners_apriltag = np.asarray(corners_apriltag, dtype=np.float64)
    if corners_apriltag.shape != (4, 2):
        raise ValueError(f'expected (4,2) corners, got {corners_apriltag.shape}')
    return corners_apriltag[[3, 2, 1, 0], :]


# ---------------------------------------------------------------------------
# Camera info helpers
# ---------------------------------------------------------------------------

def camera_matrix_from_camera_info(k_or_p_row_major: Sequence[float],
                                   use_projection_matrix: bool) -> np.ndarray:
    """Return a 3x3 float64 matrix from a 9-length K or 12-length P (row-major)."""
    arr = np.asarray(k_or_p_row_major, dtype=np.float64)
    if use_projection_matrix:
        if arr.size != 12:
            raise ValueError(f'P matrix must have 12 values, got {arr.size}')
        p = arr.reshape(3, 4)
        return p[:3, :3].copy()
    if arr.size != 9:
        raise ValueError(f'K matrix must have 9 values, got {arr.size}')
    return arr.reshape(3, 3).copy()


def is_camera_info_valid(k_matrix: np.ndarray) -> bool:
    """CameraInfo is usable for PnP if fx, fy, cx, cy are all non-zero & finite."""
    if k_matrix is None or k_matrix.shape != (3, 3):
        return False
    fx, fy, cx, cy = k_matrix[0, 0], k_matrix[1, 1], k_matrix[0, 2], k_matrix[1, 2]
    return all(v != 0.0 and math.isfinite(v) for v in (fx, fy, cx, cy))


# ---------------------------------------------------------------------------
# PnP
# ---------------------------------------------------------------------------

@dataclass
class PoseResult:
    rvec: np.ndarray            # (3,1) rotation vector
    tvec: np.ndarray            # (3,1) translation in meters, optical frame
    reprojection_error_px: float
    method: str                 # "IPPE_SQUARE" | "ITERATIVE"


def _reprojection_error(object_pts: np.ndarray, image_pts: np.ndarray,
                        rvec: np.ndarray, tvec: np.ndarray,
                        camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, camera_matrix, dist_coeffs)
    proj = proj.reshape(-1, 2)
    d = image_pts.reshape(-1, 2) - proj
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def solve_tag_pose(corners_apriltag_xy: np.ndarray,
                   tag_size_m: float,
                   camera_matrix: np.ndarray,
                   dist_coeffs: np.ndarray,
                   reproj_err_threshold_px: float = 4.0
                   ) -> Optional[PoseResult]:
    """Estimate tag pose from apriltag-ordered 2D corners.

    Returns None if CameraInfo is invalid or both solvers fail (so callers
    can keep running instead of crashing on a dropped detection).
    """
    if not is_camera_info_valid(camera_matrix):
        return None
    if tag_size_m <= 0:
        raise ValueError(f'tag_size_m must be > 0, got {tag_size_m}')

    dist = np.zeros((5,), dtype=np.float64) if dist_coeffs is None \
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

    # 1) IPPE_SQUARE with the OpenCV-required corner order.
    obj_ippe = tag_object_points_opencv_ippe_square(tag_size_m)
    img_ippe = apriltag_corners_to_opencv_ippe_order(corners_apriltag_xy).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj_ippe, img_ippe, camera_matrix, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if ok:
        err = _reprojection_error(obj_ippe, img_ippe, rvec, tvec, camera_matrix, dist)
        if err <= reproj_err_threshold_px:
            return PoseResult(rvec=rvec, tvec=tvec,
                              reprojection_error_px=err, method='IPPE_SQUARE')

    # 2) Fallback: ITERATIVE with apriltag order (any consistent order works).
    obj_apr = tag_object_points_apriltag_order(tag_size_m)
    img_apr = np.asarray(corners_apriltag_xy, dtype=np.float64)
    ok2, rvec2, tvec2 = cv2.solvePnP(
        obj_apr, img_apr, camera_matrix, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok2:
        return None
    err2 = _reprojection_error(obj_apr, img_apr, rvec2, tvec2, camera_matrix, dist)
    return PoseResult(rvec=rvec2, tvec=tvec2,
                      reprojection_error_px=err2, method='ITERATIVE')


# ---------------------------------------------------------------------------
# Rotations: Rodrigues / quaternion (for TF orientation & unit tests)
# ---------------------------------------------------------------------------

def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector (3,) or (3,1) to a 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    mat, _ = cv2.Rodrigues(rvec)
    return mat


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a unit quaternion (x, y, z, w).

    Uses Shepperd's method; the returned quaternion has w >= 0.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f'expected 3x3 matrix, got {R.shape}')
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0            # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0   # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0   # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0   # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return normalize_quaternion((qx, qy, qz, qw))


def rvec_to_quaternion(rvec: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a rotation vector directly to a unit quaternion (x, y, z, w)."""
    return rotation_matrix_to_quaternion(rodrigues_to_matrix(rvec))


def normalize_quaternion(q: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return the unit quaternion (x, y, z, w). Raises on zero-length input."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError('cannot normalize a zero quaternion')
    return (x / n, y / n, z / n, w / n)


# ---------------------------------------------------------------------------
# Distance & error stats
# ---------------------------------------------------------------------------

def euclidean_distance(tvec: Sequence[float]) -> float:
    """distance = sqrt(tx^2 + ty^2 + tz^2)  -- straight-line camera->tag centre."""
    tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
    return math.sqrt(tx * tx + ty * ty + tz * tz)


def meters_to_mm(value_m: float) -> float:
    """Convert meters -> millimeters."""
    return float(value_m) * 1000.0


def mm_to_meters(value_mm: float) -> float:
    """Convert millimeters -> meters."""
    return float(value_mm) / 1000.0


def absolute_error(estimated: float, truth: float) -> float:
    return abs(float(estimated) - float(truth))


def relative_error_percent(estimated: float, truth: float) -> float:
    if truth == 0:
        return float('inf')
    return absolute_error(estimated, truth) / float(truth) * 100.0


@dataclass
class DistanceStats:
    n: int
    mean: float
    median: float
    minimum: float
    maximum: float
    stddev: float
    mae: float                  # mean absolute error vs truth
    bias: float                 # mean(estimated - truth)
    mean_rel_err_pct: float
    rmse: float

    def as_dict(self) -> dict:
        return {
            'n': self.n,
            'mean': self.mean,
            'median': self.median,
            'min': self.minimum,
            'max': self.maximum,
            'std': self.stddev,
            'mae': self.mae,
            'bias': self.bias,
            'mean_rel_err_pct': self.mean_rel_err_pct,
            'rmse': self.rmse,
        }


def compute_distance_stats(estimated_m: Sequence[float],
                           truth_m: float) -> DistanceStats:
    """Summary statistics of a set of distance estimates against one truth."""
    if len(estimated_m) == 0:
        raise ValueError('estimated_m is empty')
    arr = np.asarray(estimated_m, dtype=np.float64)
    err = arr - float(truth_m)
    abs_err = np.abs(err)
    return DistanceStats(
        n=int(arr.size),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        minimum=float(arr.min()),
        maximum=float(arr.max()),
        stddev=float(arr.std(ddof=0)),
        mae=float(abs_err.mean()),
        bias=float(err.mean()),
        mean_rel_err_pct=float(abs_err.mean() / float(truth_m) * 100.0)
            if truth_m != 0 else float('inf'),
        rmse=float(np.sqrt(np.mean(err * err))),
    )
