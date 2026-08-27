"""Unit tests for calibration_io (calibration YAML validation).

Checks the spec's required properties:
  - YAML file exists / loadable
  - K matrix has 9 elements
  - D (distortion) exists
  - fx, fy > 0
  - resolution > 0
  - P matrix has 12 elements (3x4)
Run: colcon test --packages-select apriltag_pose
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from apriltag_pose.calibration_io import (
    validate_calibration_dict, validate_calibration_file, verdict,
)

VALID_CFG = {
    'image_width': 640,
    'image_height': 480,
    'camera_name': 'test',
    'camera_matrix': {'rows': 3, 'cols': 3, 'data': [
        600.0, 0.0, 320.0,
        0.0, 600.0, 240.0,
        0.0, 0.0, 1.0]},
    'distortion_model': 'plumb_bob',
    'distortion_coefficients': {'rows': 1, 'cols': 5,
                                'data': [-0.1, 0.05, 0.0, 0.0, 0.0]},
    'rectification_matrix': {'rows': 3, 'cols': 3, 'data': [
        1, 0, 0, 0, 1, 0, 0, 0, 1]},
    'projection_matrix': {'rows': 3, 'cols': 4, 'data': [
        600, 0, 320, 0, 0, 600, 240, 0, 0, 0, 1, 0]},
}


def _levels(checks):
    return {name: level for name, level, _ in checks}


def test_valid_calibration_passes():
    checks = validate_calibration_dict(VALID_CFG)
    lv = _levels(checks)
    assert lv.get('resolution') == 'PASS'
    assert lv.get('K_matrix') == 'PASS'
    assert lv.get('fx_fy') == 'PASS'
    assert lv.get('cx_cy') == 'PASS'
    assert lv.get('distortion') == 'PASS'
    assert lv.get('R_matrix') == 'PASS'
    assert lv.get('P_matrix') == 'PASS'
    assert verdict(checks) == 'PASS'


def test_fx_fy_zero_fails():
    cfg = dict(VALID_CFG)
    cfg['camera_matrix'] = {'rows': 3, 'cols': 3, 'data': [
        0.0, 0, 320, 0, 0.0, 240, 0, 0, 1]}
    checks = validate_calibration_dict(cfg)
    assert _levels(checks).get('fx_fy') == 'FAIL'
    assert verdict(checks) == 'FAIL'


def test_k_wrong_size_fails():
    cfg = dict(VALID_CFG)
    cfg['camera_matrix'] = {'rows': 3, 'cols': 3, 'data': [1, 2, 3]}
    checks = validate_calibration_dict(cfg)
    assert _levels(checks).get('K_matrix') == 'FAIL'


def test_missing_distortion_warns():
    cfg = dict(VALID_CFG)
    del cfg['distortion_coefficients']
    checks = validate_calibration_dict(cfg)
    assert _levels(checks).get('distortion') == 'WARN'


def test_p_matrix_format():
    cfg = dict(VALID_CFG)
    cfg['projection_matrix'] = {'rows': 3, 'cols': 4, 'data': [1, 2, 3]}  # wrong size
    checks = validate_calibration_dict(cfg)
    assert _levels(checks).get('P_matrix') == 'FAIL'


def test_resolution_zero_fails():
    cfg = dict(VALID_CFG)
    cfg['image_width'] = 0
    checks = validate_calibration_dict(cfg)
    assert _levels(checks).get('resolution') == 'FAIL'


def test_file_not_found():
    checks = validate_calibration_file('/no/such/file.yaml')
    assert _levels(checks).get('file') == 'FAIL'


def test_file_roundtrip(tmp_path: Path):
    import yaml
    p = tmp_path / 'camera_info.yaml'
    p.write_text(yaml.safe_dump(VALID_CFG), encoding='utf-8')
    assert os.path.isfile(p)
    checks = validate_calibration_file(str(p))
    assert _levels(checks).get('file') == 'PASS'
    assert verdict(checks) == 'PASS'


def test_package_placeholder_exists():
    """The shipped config/camera.yaml placeholder must exist & be loadable."""
    here = Path(__file__).resolve().parent
    placeholder = here.parent / 'config' / 'camera.yaml'
    assert placeholder.is_file(), f'missing {placeholder}'
    checks = validate_calibration_file(str(placeholder))
    # placeholder has fx=fy=0 -> fx_fy FAIL (that's the point: must calibrate)
    assert _levels(checks).get('file') == 'PASS'
    assert _levels(checks).get('fx_fy') == 'FAIL'
