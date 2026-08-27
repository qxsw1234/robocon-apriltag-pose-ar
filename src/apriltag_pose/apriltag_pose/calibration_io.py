"""calibration_io - load & validate a camera_calibration YAML (pure, no ROS).

The YAML is the standard camera_calibration / camera_calibration_parsers format::

    image_width: 640
    image_height: 480
    camera_name: ...
    camera_matrix:        {rows: 3, cols: 3, data: [...9...]}
    distortion_model: plumb_bob
    distortion_coefficients: {rows: 1, cols: 5, data: [...5...]}
    rectification_matrix: {rows: 3, cols: 3, data: [...9...]}
    projection_matrix:    {rows: 3, cols: 4, data: [...12...]}

Kept ROS-free so it can be unit-tested directly.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('PyYAML required: sudo apt install python3-yaml') from exc

Check = Tuple[str, str, str]   # (name, level, detail)  level in {PASS, WARN, FAIL}


def _matrix(data, rows: int, cols: int) -> Optional[List[float]]:
    if data is None:
        return None
    d = list(data.get('data', [])) if isinstance(data, dict) else list(data)
    if len(d) != rows * cols:
        return None
    return [float(v) for v in d]


def _field_matrix(cfg: dict, key: str, rows: int, cols: int):
    """Return (present, values).

    present=False  -> the field is absent (caller may WARN).
    present=True, values=None  -> field exists but has the wrong size (FAIL).
    present=True, values=[...] -> well-formed matrix.
    """
    field = cfg.get(key)
    if field is None:
        return False, None
    return True, _matrix(field, rows, cols)


def load_calibration_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_calibration_dict(cfg: dict) -> List[Check]:
    """Return a list of (name, level, detail) checks for a parsed config."""
    checks: List[Check] = []

    w = int(cfg.get('image_width', 0))
    h = int(cfg.get('image_height', 0))
    if w > 0 and h > 0:
        checks.append(('resolution', 'PASS', f'{w}x{h}'))
    else:
        checks.append(('resolution', 'FAIL', 'image_width/height missing or zero'))

    k_present, K = _field_matrix(cfg, 'camera_matrix', 3, 3)
    d_present, D = _field_matrix(cfg, 'distortion_coefficients', 1, 5)
    # distortion can legitimately be 4/5/8/14 coeffs; accept any present list.
    if d_present and D is None:
        raw = cfg.get('distortion_coefficients')
        D = list(raw.get('data', [])) if isinstance(raw, dict) else list(raw)
        D = [float(v) for v in D] or None
    r_present, R = _field_matrix(cfg, 'rectification_matrix', 3, 3)
    p_present, P = _field_matrix(cfg, 'projection_matrix', 3, 4)

    # K: required. Missing or wrong size => FAIL.
    if not k_present:
        checks.append(('K_matrix', 'FAIL', 'camera_matrix missing'))
    elif K is None:
        checks.append(('K_matrix', 'FAIL', 'camera_matrix must have 9 elements'))
    else:
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        checks.append(('K_matrix', 'PASS', f'9 elements; fx={fx:.1f} fy={fy:.1f}'))
        if fx > 0 and fy > 0:
            checks.append(('fx_fy', 'PASS', f'fx={fx:.1f} fy={fy:.1f}'))
        else:
            checks.append(('fx_fy', 'FAIL', f'fx/fy must be > 0, got fx={fx} fy={fy}'))
        if w > 0 and h > 0 and 0 <= cx <= w and 0 <= cy <= h:
            checks.append(('cx_cy', 'PASS', f'cx={cx:.1f} cy={cy:.1f} within {w}x{h}'))
        else:
            checks.append(('cx_cy', 'WARN', f'cx={cx} cy={cy} outside {w}x{h}'))

    # D: only needs to exist; all-zero is suspicious (rectified/ideal).
    if not d_present:
        checks.append(('distortion', 'WARN', 'no distortion_coefficients field'))
    elif D is None or len(D) == 0:
        checks.append(('distortion', 'WARN', 'distortion_coefficients empty'))
    elif all(v == 0.0 for v in D):
        checks.append(('distortion', 'WARN',
                       'all distortion coeffs zero (ok only if rectified/ideal)'))
    else:
        checks.append(('distortion', 'PASS',
                       f'{len(D)} coeffs, model={cfg.get("distortion_model", "?")}'))

    # R: optional; wrong size => FAIL.
    if not r_present:
        checks.append(('R_matrix', 'WARN', 'rectification_matrix missing'))
    elif R is None:
        checks.append(('R_matrix', 'FAIL', 'R must have 9 elements'))
    else:
        checks.append(('R_matrix', 'PASS', '9 elements'))

    # P: optional; wrong size => FAIL.
    if not p_present:
        checks.append(('P_matrix', 'WARN', 'projection_matrix missing'))
    elif P is None:
        checks.append(('P_matrix', 'FAIL', 'P must have 12 elements (3x4)'))
    else:
        checks.append(('P_matrix', 'PASS', '12 elements (3x4)'))

    return checks


def validate_calibration_file(path: str) -> List[Check]:
    """Validate a calibration YAML on disk. Returns checks (incl. file-exists)."""
    if not path:
        return [('file', 'WARN', 'no calibration_file given')]
    if not os.path.isfile(path):
        return [('file', 'FAIL', f'file not found: {path}')]
    try:
        cfg = load_calibration_yaml(path)
    except Exception as exc:  # noqa: BLE001
        return [('file', 'FAIL', f'cannot parse YAML: {exc}')]
    return [('file', 'PASS', path)] + validate_calibration_dict(cfg)


def verdict(checks: List[Check]) -> str:
    return 'FAIL' if any(c[1] == 'FAIL' for c in checks) else 'PASS'
