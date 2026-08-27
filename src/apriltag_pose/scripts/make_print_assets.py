#!/usr/bin/env python3
"""make_print_assets.py - 生成标定与检测用的可打印 A4 PDF (纯矢量, 尺寸精确).

产出 (默认写到 <workspace>/print_assets/):
    calibration_checkerboard_A4.pdf   9x6 内部角点棋盘格, 方格 25 mm
    apriltag_36h11_id0_A4.pdf         tag36h11 ID 0, 黑色方块外边长 80 mm
    print_assets_README.txt           打印与实测说明

为什么用矢量而不是嵌入 PNG:
    位图在打印时会被重采样, 边缘发虚, 影响角点/边缘检测精度。
    这里把棋盘格和 AprilTag 的每个黑格都画成独立矢量矩形, 打印为纯黑色块。

尺寸约定 (关键, 直接影响位姿精度):
    * 棋盘格 --size 参数 = 内部角点数 (列 x 行), 不是方格数。
      9x6 内部角点 => 10x7 个方格。
    * AprilTag 的 tag_size = 【黑色方块的外边长】, 不含白边。
      本脚本把 36h11 的 8x8 单元格黑方块画成 80 mm, 即每格 10 mm。
      检测器返回的 4 个角点正好是这个黑方块的 4 个角 (已验证)。

打印务必:
    选择「实际大小 / 100% / 不缩放」, 关闭「适应页面」, 否则实测尺寸会偏小。
    打印后一定用尺子/卡尺实测, 用实测值而不是标称值配置系统。

用法:
    python3 make_print_assets.py [-o 输出目录]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas

# ---- 棋盘格标称参数 -------------------------------------------------------
CB_COLS_CORNERS = 9      # 内部角点: 长边方向
CB_ROWS_CORNERS = 6      # 内部角点: 短边方向
CB_SQUARE_MM = 25.0      # 方格标称边长

# ---- AprilTag 标称参数 ---------------------------------------------------
TAG_FAMILY_CELLS = 8     # tag36h11: 6x6 数据位 + 1 圈黑边 = 8x8 单元格
TAG_BLACK_MM = 80.0      # 黑色方块外边长 (= tag_size)
TAG_QUIET_CELLS = 2      # 白色静默区宽度 (单元格数), apriltag 至少要 1


def _draw_crosshair(c: rl_canvas.Canvas, x: float, y: float, arm: float = 4 * mm) -> None:
    """在 (x, y) 画一个细十字, 供打印后用尺子核对基准点."""
    c.setLineWidth(0.3)
    c.line(x - arm, y, x + arm, y)
    c.line(x, y - arm, x, y + arm)


def _header(c: rl_canvas.Canvas, title: str, lines: list[str],
            title_y_mm: float = 288.0, font_size: float = 7.0,
            line_step_mm: float = 3.6) -> None:
    """页眉标题 + 说明文字 (英文, 避免 PDF 中文字体依赖).

    返回值不需要, 但调用方需要自己保证正文不会和最后一行重叠;
    最后一行的 y = title_y_mm - 5 - (len(lines)-1) * line_step_mm (单位 mm)。
    """
    c.setFont('Helvetica-Bold', 11)
    c.drawString(15 * mm, title_y_mm * mm, title)
    c.setFont('Helvetica', font_size)
    y = (title_y_mm - 5.0) * mm
    for line in lines:
        c.drawString(15 * mm, y, line)
        y -= line_step_mm * mm


def make_checkerboard(path: Path) -> dict:
    """9x6 内部角点棋盘格. 10x7 个方格, 长边方向放 10 格."""
    w, h = A4
    n_x = CB_ROWS_CORNERS + 1          # 7  格, 沿纸宽 (210 mm)
    n_y = CB_COLS_CORNERS + 1          # 10 格, 沿纸高 (297 mm)
    sq = CB_SQUARE_MM * mm
    board_w, board_h = n_x * sq, n_y * sq   # 175 x 250 mm

    # 垂直预算 (mm), 自上而下. 297 总高:
    #   页眉占 297..~272, 棋盘 250, 底部标尺+文字需要 ~11
    # 显式算出来而不是靠 (h - margin)/2, 避免标尺跑到纸外面.
    header_bottom_mm = 271.0           # 页眉最后一行以下的安全线
    ruler_budget_mm = 12.0             # 底部标尺 + 说明文字
    avail_mm = header_bottom_mm - ruler_budget_mm
    board_h_mm = n_y * CB_SQUARE_MM
    if board_h_mm > avail_mm:
        raise RuntimeError(f'checkerboard {board_h_mm} mm does not fit in '
                           f'{avail_mm} mm of vertical space')
    y0_mm = ruler_budget_mm + (avail_mm - board_h_mm) / 2.0
    y0 = y0_mm * mm
    x0 = (w - board_w) / 2.0
    if x0 < 8 * mm:
        raise RuntimeError('checkerboard too wide for A4 side margins')

    c = rl_canvas.Canvas(str(path), pagesize=A4)
    _header(c, 'Camera Calibration Checkerboard', [
        f'Pattern: {CB_COLS_CORNERS}x{CB_ROWS_CORNERS} INTERIOR CORNERS '
        f'({n_y}x{n_x} squares).  Nominal square = {CB_SQUARE_MM:.1f} mm.',
        'PRINT AT 100% / ACTUAL SIZE. Disable "fit to page" / "shrink oversized pages".',
        'After printing, measure across 10 squares with a ruler and divide by 10 '
        '-> real square size.',
        'Then:  ros2 launch apriltag_pose calibration.launch.py '
        'size:=<measured_m> checkerboard:=9x6',
        'Mount FLAT on a rigid board. Any bend ruins the calibration.',
    ])

    # 黑白格
    for iy in range(n_y):
        for ix in range(n_x):
            if (ix + iy) % 2 == 0:
                continue
            c.setFillGray(0.0)
            # 多画 0.05 mm 重叠, 避免打印时相邻黑格之间出现白缝
            c.rect(x0 + ix * sq, y0 + iy * sq,
                   sq + 0.05 * mm, sq + 0.05 * mm, stroke=0, fill=1)

    # 棋盘外框 (细线, 便于裁剪对齐)
    c.setStrokeGray(0.0)
    c.setLineWidth(0.3)
    c.rect(x0, y0, board_w, board_h, stroke=1, fill=0)

    # 底部标尺: 标出 7 格 (纸宽方向) 的总长
    c.setFont('Helvetica', 7)
    ruler_y = y0 - 5 * mm
    c.line(x0, ruler_y, x0 + board_w, ruler_y)
    _draw_crosshair(c, x0, ruler_y, 1.5 * mm)
    _draw_crosshair(c, x0 + board_w, ruler_y, 1.5 * mm)
    c.drawCentredString(x0 + board_w / 2.0, ruler_y - 3.6 * mm,
                        f'{n_x} squares = {n_x * CB_SQUARE_MM:.0f} mm nominal '
                        f'(MEASURE THIS, divide by {n_x})')

    # 左侧竖直标尺: 10 格总长
    ruler_x = x0 - 5 * mm
    c.line(ruler_x, y0, ruler_x, y0 + board_h)
    _draw_crosshair(c, ruler_x, y0, 1.5 * mm)
    _draw_crosshair(c, ruler_x, y0 + board_h, 1.5 * mm)
    c.saveState()
    c.translate(ruler_x - 1.5 * mm, y0 + board_h / 2.0)
    c.rotate(90)
    c.drawCentredString(0, 0, f'{n_y} squares = {n_y * CB_SQUARE_MM:.0f} mm nominal '
                              f'(MEASURE THIS, divide by {n_y})')
    c.restoreState()

    c.showPage()
    c.save()
    return {'squares': (n_y, n_x), 'nominal_square_mm': CB_SQUARE_MM,
            'span_x_mm': n_x * CB_SQUARE_MM, 'span_y_mm': n_y * CB_SQUARE_MM}


def tag_bit_grid(tag_id: int = 0) -> np.ndarray:
    """返回 tag36h11 的 8x8 bool 网格 (True = 黑格).

    做法: 用 OpenCV 渲染成位图, 再按单元格中心采样, 得到离散 bit 图,
    这样后面可以画成矢量而不是嵌位图。顺带用 detectMarkers 自检 ID。
    """
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36H11)
    px_per_cell = 20
    side = TAG_FAMILY_CELLS * px_per_cell
    img = cv2.aruco.drawMarker(dictionary, tag_id, side)

    # 自检: 加白边后必须能被检测回同一个 ID
    pad = TAG_QUIET_CELLS * px_per_cell
    probe = np.full((side + 2 * pad, side + 2 * pad), 255, np.uint8)
    probe[pad:pad + side, pad:pad + side] = img
    corners, ids, _ = cv2.aruco.detectMarkers(
        cv2.cvtColor(probe, cv2.COLOR_GRAY2BGR), dictionary)
    if ids is None or int(ids.ravel()[0]) != tag_id:
        raise RuntimeError(f'self-check failed: rendered tag did not decode '
                           f'back to id {tag_id} (got {ids})')
    # 自检: 检测到的角点应当就是黑方块外边界
    quad = corners[0].reshape(4, 2)
    lo, hi = quad.min(axis=0), quad.max(axis=0)
    if not (np.allclose(lo, pad, atol=1.5) and
            np.allclose(hi, pad + side - 1, atol=1.5)):
        raise RuntimeError('self-check failed: detected corners do not match the '
                           f'black square bounds (got {lo}..{hi}, '
                           f'expected {pad}..{pad + side - 1})')

    grid = np.zeros((TAG_FAMILY_CELLS, TAG_FAMILY_CELLS), dtype=bool)
    half = px_per_cell // 2
    for r in range(TAG_FAMILY_CELLS):
        for col in range(TAG_FAMILY_CELLS):
            v = img[r * px_per_cell + half, col * px_per_cell + half]
            grid[r, col] = (v < 128)
    return grid


def make_apriltag(path: Path, tag_id: int = 0) -> dict:
    """tag36h11 ID 0, 黑方块外边长 80 mm, 带白色静默区."""
    w, h = A4
    grid = tag_bit_grid(tag_id)
    cell = (TAG_BLACK_MM / TAG_FAMILY_CELLS) * mm      # 10 mm
    black = TAG_BLACK_MM * mm
    quiet = TAG_QUIET_CELLS * cell
    total = black + 2 * quiet                          # 120 mm

    # 垂直预算 (mm): 页眉到 271, 底部标尺 12, tag 含白边 120
    header_bottom_mm = 271.0
    ruler_budget_mm = 12.0
    avail_mm = header_bottom_mm - ruler_budget_mm
    total_mm = TAG_BLACK_MM + 2 * TAG_QUIET_CELLS * (TAG_BLACK_MM / TAG_FAMILY_CELLS)
    if total_mm > avail_mm:
        raise RuntimeError(f'tag block {total_mm} mm does not fit in {avail_mm} mm')
    y0_mm = ruler_budget_mm + (avail_mm - total_mm) / 2.0
    y0 = y0_mm * mm
    x0 = (w - total) / 2.0                             # 白边左下角
    bx, by = x0 + quiet, y0 + quiet                    # 黑方块左下角

    c = rl_canvas.Canvas(str(path), pagesize=A4)
    _header(c, f'AprilTag  tag36h11  ID {tag_id}', [
        f'tag_size = OUTER EDGE OF THE BLACK SQUARE = {TAG_BLACK_MM:.1f} mm nominal '
        f'({TAG_FAMILY_CELLS} cells x {TAG_BLACK_MM / TAG_FAMILY_CELLS:.1f} mm).',
        'PRINT AT 100% / ACTUAL SIZE. Disable "fit to page".',
        'After printing, measure the BLACK SQUARE edge (not the white margin) '
        'with a ruler/caliper.',
        'Use that measured value (in METERS):  ros2 launch apriltag_pose '
        'apriltag_pose.launch.py tag_size:=<measured_m>',
        'Keep the white quiet zone clean - the detector needs it. '
        'Mount FLAT, avoid glossy paper (specular glare breaks detection).',
    ])

    # 白色静默区 (显式画白底, 保证即使贴在深色板上也有白边)
    c.setFillGray(1.0)
    c.rect(x0, y0, total, total, stroke=0, fill=1)

    # 黑格 (注意: 图像行 r 自上而下, PDF 坐标自下而上, 所以要翻转)
    c.setFillGray(0.0)
    n = TAG_FAMILY_CELLS
    for r in range(n):
        for col in range(n):
            if not grid[r, col]:
                continue
            cx = bx + col * cell
            cy = by + (n - 1 - r) * cell
            # 相邻黑格之间多画 0.05 mm 重叠, 避免打印时出现白色缝隙
            c.rect(cx, cy, cell + 0.05 * mm, cell + 0.05 * mm, stroke=0, fill=1)

    # 白边外框 + 黑方块基准十字 (打印后量这两个十字之间的距离)
    c.setStrokeGray(0.6)
    c.setLineWidth(0.3)
    c.rect(x0, y0, total, total, stroke=1, fill=0)

    c.setStrokeGray(0.0)
    c.setFont('Helvetica', 7)
    ruler_y = y0 - 5 * mm
    c.line(bx, ruler_y, bx + black, ruler_y)
    _draw_crosshair(c, bx, ruler_y, 1.5 * mm)
    _draw_crosshair(c, bx + black, ruler_y, 1.5 * mm)
    c.drawCentredString(bx + black / 2.0, ruler_y - 3.6 * mm,
                        f'black square edge = {TAG_BLACK_MM:.0f} mm nominal '
                        f'(MEASURE THIS -> tag_size)')
    c.showPage()
    c.save()
    return {'tag_id': tag_id, 'cells': n, 'nominal_black_mm': TAG_BLACK_MM,
            'cell_mm': TAG_BLACK_MM / n, 'total_with_quiet_mm': TAG_BLACK_MM + 2 * TAG_QUIET_CELLS * (TAG_BLACK_MM / n)}


_README = """打印与实测说明 (print_assets)
=====================================================================
文件:
  calibration_checkerboard_A4.pdf   标定用棋盘格
  apriltag_36h11_id0_A4.pdf         检测用 AprilTag (tag36h11, ID 0)

---------------------------------------------------------------------
1. 打印 (最容易出错的一步)
---------------------------------------------------------------------
两个 PDF 都必须按【实际大小 / 100% / 不缩放】打印:
  * 系统打印对话框: 取消勾选「适应页面」/「Fit to page」/「缩放以适合」
  * 缩放比例选 100% 或「实际大小 / Actual size」
  * 用普通 A4 哑光纸, 不要用相纸/铜版纸 (反光会导致检测失败)

打印完先看页面上标注的基准十字, 用尺子量一下, 确认和标称值一致。
若明显偏小 (常见: 被缩放到 96%), 说明缩放没关掉, 重新打印。

---------------------------------------------------------------------
2. 实测 (决定位姿精度的关键)
---------------------------------------------------------------------
棋盘格:
  沿底边量【10 个方格的总长】, 标称 250 mm, 然后除以 10 得到单格边长。
  量总长再除, 比直接量一格误差小 10 倍。
  例: 实测 248.5 mm -> 单格 24.85 mm -> size:=0.02485

AprilTag:
  量【黑色方块的外边长】, 不要把白边算进去。标称 80 mm。
  同样建议量对边之间, 多量几次取平均。
  例: 实测 79.4 mm -> tag_size:=0.0794

  注意: tag_size 定义为黑色方块外边长。本 PDF 生成时已用
  cv2.aruco.detectMarkers 自检过: 检测器返回的 4 个角点正好落在
  黑方块的 4 个角上, 所以「黑方块外边长」就是要填的 tag_size。

---------------------------------------------------------------------
3. 粘贴
---------------------------------------------------------------------
两张纸都必须【平整】:
  * 用胶棒/双面胶整面贴在硬纸板/泡沫板/写字板上, 不要只贴四个角
  * 棋盘格弯曲会直接让内参标定结果错掉
  * AprilTag 弯曲会让 solvePnP 的平面假设不成立, 位姿抖动变大

---------------------------------------------------------------------
4. 填哪里
---------------------------------------------------------------------
棋盘格实测值 -> 标定命令行:
  ros2 launch apriltag_pose calibration.launch.py \\
      size:=<单格边长, 米> checkerboard:=9x6

AprilTag 实测值 -> 以下位置必须一致:
  config/apriltag.yaml   size 和 tag.sizes[0]
  config/system.yaml     tag_size_m
  launch 默认值          tag_size  (或每次命令行传 tag_size:=)
"""


def verify_pdfs(cb_pdf: Path, tag_pdf: Path, dpi: int = 300) -> list[str]:
    """把 PDF 栅格化, 用检测器自检几何尺寸是否精确.

    这一步很有价值: 它能抓住「页面被缩放」「格子数写错」「tag_size 定义
    弄反 (把含白边的 120 mm 当成 tag_size)」这类静默错误。
    需要 poppler-utils 的 pdftoppm; 没装就跳过 (返回说明)。
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which('pdftoppm') is None:
        return ['[SKIP] pdftoppm (poppler-utils) not found, geometry '
                'self-check skipped.']

    out: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        def raster(pdf: Path, stem: str) -> np.ndarray:
            subprocess.run(['pdftoppm', '-r', str(dpi), '-gray', '-png',
                            str(pdf), str(tmp / stem)],
                           check=True, capture_output=True)
            hits = sorted(tmp.glob(f'{stem}-*.png'))
            if not hits:
                raise RuntimeError(f'pdftoppm produced no output for {pdf}')
            img = cv2.imread(str(hits[0]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f'could not read raster of {pdf}')
            return img

        px2mm = 25.4 / dpi

        # ---- checkerboard: 必须检测到正好 9x6 内部角点, 且格距 = 25 mm
        img = raster(cb_pdf, 'cb')
        page_mm = (img.shape[1] * px2mm, img.shape[0] * px2mm)
        out.append(f'  page size: {page_mm[0]:.1f} x {page_mm[1]:.1f} mm '
                   f'(A4 = 210.0 x 297.0)')
        found, corners = cv2.findChessboardCorners(
            img, (CB_COLS_CORNERS, CB_ROWS_CORNERS), None)
        if not found:
            out.append(f'  [FAIL] checkerboard: findChessboardCorners could not '
                       f'find {CB_COLS_CORNERS}x{CB_ROWS_CORNERS} corners')
        else:
            corners = cv2.cornerSubPix(
                img, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            grid = corners.reshape(CB_ROWS_CORNERS, CB_COLS_CORNERS, 2)
            d_row = np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel() * px2mm
            d_col = np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel() * px2mm
            err = max(abs(d_row.mean() - CB_SQUARE_MM),
                      abs(d_col.mean() - CB_SQUARE_MM))
            tag_ok = 'OK' if err < 0.15 else 'FAIL'
            out.append(f'  [{tag_ok}] checkerboard: {CB_COLS_CORNERS}x'
                       f'{CB_ROWS_CORNERS} corners found, square = '
                       f'{d_row.mean():.3f} / {d_col.mean():.3f} mm '
                       f'(target {CB_SQUARE_MM:.1f}, max err {err:.3f} mm)')

        # ---- AprilTag: 必须解码回 ID 0, 且黑方块边长 = TAG_BLACK_MM
        img = raster(tag_pdf, 'tag')
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36H11)
        quads, ids, _ = cv2.aruco.detectMarkers(
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), dictionary)
        if ids is None or len(ids) != 1 or int(ids.ravel()[0]) != 0:
            out.append(f'  [FAIL] apriltag: expected exactly one tag with ID 0, '
                       f'got {None if ids is None else ids.ravel().tolist()}')
        else:
            quad = quads[0].reshape(4, 2)
            edges = np.array([np.linalg.norm(quad[i] - quad[(i + 1) % 4])
                              for i in range(4)]) * px2mm
            err = abs(edges.mean() - TAG_BLACK_MM)
            tag_ok = 'OK' if err < 0.15 else 'FAIL'
            out.append(f'  [{tag_ok}] apriltag: decoded ID 0, black square edge = '
                       f'{edges.mean():.3f} mm '
                       f'(target {TAG_BLACK_MM:.1f}, err {err:.3f} mm)')
            out.append(f'         -> tag_size to configure = '
                       f'{edges.mean() / 1000.0:.5f} m  (NOMINAL; still measure '
                       f'the real print!)')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_out = Path(__file__).resolve().parents[3] / 'print_assets'
    ap.add_argument('-o', '--output-dir', type=Path, default=default_out)
    ap.add_argument('--no-verify', action='store_true',
                    help='skip the rasterize + re-detect geometry self-check')
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cb = out / 'calibration_checkerboard_A4.pdf'
    tg = out / 'apriltag_36h11_id0_A4.pdf'

    cb_info = make_checkerboard(cb)
    tg_info = make_apriltag(tg, tag_id=0)
    (out / 'print_assets_README.txt').write_text(_README, encoding='utf-8')

    print(f'[OK] {cb}')
    print(f'     {cb_info["squares"][0]}x{cb_info["squares"][1]} squares = '
          f'{CB_COLS_CORNERS}x{CB_ROWS_CORNERS} interior corners, '
          f'square {cb_info["nominal_square_mm"]:.1f} mm nominal')
    print(f'     measure spans: {cb_info["span_y_mm"]:.0f} mm x '
          f'{cb_info["span_x_mm"]:.0f} mm')
    print(f'[OK] {tg}')
    print(f'     tag36h11 id {tg_info["tag_id"]}, black square '
          f'{tg_info["nominal_black_mm"]:.1f} mm nominal '
          f'({tg_info["cells"]} cells x {tg_info["cell_mm"]:.1f} mm), '
          f'with quiet zone {tg_info["total_with_quiet_mm"]:.0f} mm')
    print(f'[OK] {out / "print_assets_README.txt"}')

    if not args.no_verify:
        print('\n--- geometry self-check (rasterize @300dpi + re-detect) ---')
        failed = False
        for line in verify_pdfs(cb, tg):
            print(line)
            if '[FAIL]' in line:
                failed = True
        if failed:
            print('\n[ERROR] geometry self-check FAILED - do not print these files.')
            return 1

    print('\nPRINT AT 100% (no "fit to page"), then MEASURE and use the '
          'measured values.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
