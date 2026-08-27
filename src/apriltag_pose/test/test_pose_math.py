"""Unit tests for pose_math helpers (pure numpy/cv2, no ROS).

Covers everything the spec requires:
  - tvec euclidean distance
  - m <-> mm conversion
  - absolute / relative error
  - quaternion normalization
  - Rodrigues rotation-vector -> rotation matrix
  - solvePnP recovers a known pose (and returns None gracefully)
Run: colcon test --packages-select apriltag_pose
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from apriltag_pose.pose_math import (
    absolute_error,
    apriltag_corners_to_opencv_ippe_order,
    camera_matrix_from_camera_info,
    compute_distance_stats,
    euclidean_distance,
    is_camera_info_valid,
    meters_to_mm,
    mm_to_meters,
    normalize_quaternion,
    relative_error_percent,
    rodrigues_to_matrix,
    rotation_matrix_to_quaternion,
    rvec_to_quaternion,
    solve_tag_pose,
    tag_object_points_apriltag_order,
    tag_object_points_opencv_ippe_square,
)


# ---------------------------------------------------------------------------
# 3D corners
# ---------------------------------------------------------------------------
def test_apriltag_corner_size():
    pts = tag_object_points_apriltag_order(0.1)
    assert pts.shape == (4, 3)
    for i in range(4):
        d = np.linalg.norm(pts[(i + 1) % 4] - pts[i])
        assert math.isclose(d, 0.1, abs_tol=1e-9)


def test_ippe_corner_size():
    pts = tag_object_points_opencv_ippe_square(0.2)
    assert pts.shape == (4, 3)
    for i in range(4):
        d = np.linalg.norm(pts[(i + 1) % 4] - pts[i])
        assert math.isclose(d, 0.2, abs_tol=1e-9)


def test_invalid_tag_size_raises():
    for bad in (0.0, -0.05):
        with pytest.raises(ValueError):
            tag_object_points_apriltag_order(bad)
        with pytest.raises(ValueError):
            tag_object_points_opencv_ippe_square(bad)


# ---------------------------------------------------------------------------
# Corner reordering
# ---------------------------------------------------------------------------
def test_corner_reorder():
    # apriltag order [BL, BR, TR, TL] in image pixels
    apr = np.array([[0, 100], [100, 100], [100, 0], [0, 0]], dtype=np.float64)
    ippe = apriltag_corners_to_opencv_ippe_order(apr)
    expected = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    np.testing.assert_allclose(ippe, expected)


def test_corner_reorder_bad_shape():
    with pytest.raises(ValueError):
        apriltag_corners_to_opencv_ippe_order(np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# CameraInfo helpers
# ---------------------------------------------------------------------------
def test_camera_matrix_from_k():
    k = [500, 0, 320, 0, 500, 240, 0, 0, 1]
    m = camera_matrix_from_camera_info(k, use_projection_matrix=False)
    assert m[0, 0] == 500 and m[1, 1] == 500 and m[0, 2] == 320


def test_camera_matrix_from_p():
    p = [500, 0, 320, 0, 0, 500, 240, 0, 0, 0, 1, 0]
    m = camera_matrix_from_camera_info(p, use_projection_matrix=True)
    np.testing.assert_allclose(m, [[500, 0, 320], [0, 500, 240], [0, 0, 1]])


def test_camera_matrix_bad_size():
    with pytest.raises(ValueError):
        camera_matrix_from_camera_info([1, 2, 3], use_projection_matrix=False)
    with pytest.raises(ValueError):
        camera_matrix_from_camera_info([1, 2, 3], use_projection_matrix=True)


def test_is_camera_info_valid():
    k_valid = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    assert is_camera_info_valid(k_valid) is True
    assert is_camera_info_valid(np.zeros((3, 3))) is False
    assert is_camera_info_valid(None) is False


# ---------------------------------------------------------------------------
# solvePnP end-to-end (synthetic projection)
# ---------------------------------------------------------------------------
def _synth_projection(tag_size, tvec_true, rvec_true, K):
    obj = tag_object_points_apriltag_order(tag_size)
    proj, _ = cv2.projectPoints(obj,
                                np.asarray(rvec_true, dtype=np.float64).reshape(3, 1),
                                np.asarray(tvec_true, dtype=np.float64).reshape(3, 1),
                                K, np.zeros(5))
    return proj.reshape(-1, 2)


def test_solve_pose_recovers_tvec():
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float64)
    tvec_true = np.array([0.02, -0.03, 0.75])
    rvec_true = np.array([0.1, -0.05, 0.03])
    corners = _synth_projection(0.10, tvec_true, rvec_true, K)
    result = solve_tag_pose(corners, 0.10, K, np.zeros(5))
    assert result is not None
    np.testing.assert_allclose(result.tvec.reshape(-1), tvec_true, atol=1e-3)
    assert result.reprojection_error_px < 1e-3
    assert result.method in ('IPPE_SQUARE', 'ITERATIVE')


def test_solve_pose_rejects_invalid_camera_info():
    # no detection / invalid K => None, must NOT crash
    K = np.zeros((3, 3))
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    assert solve_tag_pose(corners, 0.1, K, np.zeros(5)) is None


def test_solve_pose_invalid_tag_size():
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float64)
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    with pytest.raises(ValueError):
        solve_tag_pose(corners, 0.0, K, np.zeros(5))


# ---------------------------------------------------------------------------
# Distance & error scalars
# ---------------------------------------------------------------------------
def test_euclidean_distance():
    assert math.isclose(euclidean_distance((3.0, 4.0, 0.0)), 5.0)
    assert math.isclose(euclidean_distance((1.0, 2.0, 2.0)), 3.0)


def test_m_mm_conversion():
    assert math.isclose(meters_to_mm(0.5), 500.0)
    assert math.isclose(mm_to_meters(500.0), 0.5)
    # round-trip
    assert math.isclose(mm_to_meters(meters_to_mm(0.123)), 0.123)


def test_absolute_error():
    assert absolute_error(0.51, 0.5) == pytest.approx(0.01)
    assert absolute_error(0.49, 0.5) == pytest.approx(0.01)


def test_relative_error_percent():
    assert relative_error_percent(0.51, 0.5) == pytest.approx(2.0)
    assert relative_error_percent(0.5, 0) == float('inf')


def test_distance_stats():
    est = [0.50, 0.51, 0.49, 0.52, 0.48]
    s = compute_distance_stats(est, 0.50)
    assert s.n == 5
    assert math.isclose(s.mae, 0.012, abs_tol=1e-6)
    assert math.isclose(s.bias, 0.0, abs_tol=1e-6)
    assert s.rmse >= 0.0


# ---------------------------------------------------------------------------
# Rotations: Rodrigues / quaternion
# ---------------------------------------------------------------------------
def test_rodrigues_to_matrix_is_rotation():
    rvec = np.array([0.0, 0.0, math.pi / 3])
    R = rodrigues_to_matrix(rvec)
    assert R.shape == (3, 3)
    # orthonormal: R R^T = I, det = 1
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-9)


def test_rodrigues_zero_is_identity():
    R = rodrigues_to_matrix(np.zeros(3))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)


def test_rvec_to_quaternion_matches_rodrigues():
    rvec = np.array([0.1, -0.2, 0.3])
    qx, qy, qz, qw = rvec_to_quaternion(rvec)
    # reconstruct matrix from quaternion and compare to Rodrigues
    q = np.array([qx, qy, qz, qw])
    n = np.linalg.norm(q)
    assert math.isclose(n, 1.0, abs_tol=1e-9)   # normalized
    qx, qy, qz, qw = q / n
    Rq = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    np.testing.assert_allclose(Rq, rodrigues_to_matrix(rvec), atol=1e-9)


def test_normalize_quaternion():
    q = normalize_quaternion((0.0, 0.0, 0.0, 2.0))   # pure w
    assert math.isclose(sum(c * c for c in q), 1.0, abs_tol=1e-9)
    assert q[3] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_quaternion((0.0, 0.0, 0.0, 0.0))


def test_rotation_matrix_to_quaternion_identity():
    qx, qy, qz, qw = rotation_matrix_to_quaternion(np.eye(3))
    assert math.isclose(qw, 1.0, abs_tol=1e-9)
    assert all(math.isclose(v, 0.0, abs_tol=1e-9) for v in (qx, qy, qz))
