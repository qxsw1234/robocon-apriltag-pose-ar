#!/usr/bin/env python3
"""analyze_distance_results.py - 距离误差统计与报告生成.

读取 results/distance_measurements.csv, 计算:
  - 每个 (true_distance, sample_group) 的 n/mean/median/std/MAE/RMSE/max|err|/bias
  - 全局汇总
输出:
  results/distance_summary.csv
  results/distance_error_plot.png   (4 子图)
  results/distance_report.md

用法:
  python3 analyze_distance_results.py [path/to/distance_measurements.csv]
依赖: pandas, matplotlib  (sudo apt install python3-pandas python3-matplotlib)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("缺少 pandas: sudo apt install python3-pandas")
import matplotlib
matplotlib.use('Agg')  # 无显示环境也能存图
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_CSV = Path.home() / 'apriltag_pose_ws' / 'results' / 'distance_measurements.csv'
DEFAULT_OUT_DIR = Path.home() / 'apriltag_pose_ws' / 'results'

NUM_COLS = [
    'true_distance_m', 'estimated_tx_m', 'estimated_ty_m', 'estimated_tz_m',
    'estimated_norm_m', 'absolute_error_m', 'absolute_error_mm',
    'relative_error_percent', 'rvec_x', 'rvec_y', 'rvec_z', 'decision_margin',
    'tag_size_m',
]


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(f"找不到 CSV: {path}\n先运行 run_distance_test.sh 采集数据。")
    df = pd.read_csv(path)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = df[c].apply(_safe_float)
    # 绝对误差/相对误差 (按 estimated_norm vs true 重新算, 防止 CSV 缺列)
    df['est'] = df['estimated_norm_m']
    df['err'] = df['est'] - df['true_distance_m']
    df['abs_err'] = df['err'].abs()
    df['rel_err_pct'] = df['abs_err'] / df['true_distance_m'] * 100.0
    return df


def per_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (truth, group), g in df.groupby(['true_distance_m', 'sample_group']):
        err = g['err'].values
        abs_err = g['abs_err'].values
        rows.append({
            'true_distance_m': truth,
            'sample_group': group,
            'n': len(g),
            'mean_est_m': g['est'].mean(),
            'median_est_m': np.median(g['est']),
            'std_m': g['est'].std(ddof=0),
            'MAE_mm': abs_err.mean() * 1000,
            'RMSE_mm': np.sqrt((err ** 2).mean()) * 1000,
            'max_abs_err_mm': abs_err.max() * 1000,
            'bias_mm': err.mean() * 1000,
            'mean_rel_err_pct': abs_err.mean() / truth * 100.0,
        })
    out = pd.DataFrame(rows).sort_values(['true_distance_m', 'sample_group'])
    return out


def overall_stats(df: pd.DataFrame) -> dict:
    err = df['err'].values
    abs_err = df['abs_err'].values
    return {
        'n': len(df),
        'MAE_mm': abs_err.mean() * 1000,
        'RMSE_mm': np.sqrt((err ** 2).mean()) * 1000,
        'std_abs_err_mm': abs_err.std(ddof=0) * 1000,
        'max_abs_err_mm': abs_err.max() * 1000,
        'bias_mm': err.mean() * 1000,
        'mean_rel_err_pct': abs_err.mean() / df['true_distance_m'].mean() * 100.0,
    }


def make_plots(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) true vs estimated (scatter, colored by group)
    ax = axes[0, 0]
    for group, g in df.groupby('sample_group'):
        ax.scatter(g['true_distance_m'], g['est'], s=12, alpha=0.6, label=f'est ({group})')
    lim = [0, max(df['true_distance_m'].max(), df['est'].max()) * 1.1]
    ax.plot(lim, lim, 'k--', lw=1, label='ideal')
    ax.set_xlabel('true distance (m)')
    ax.set_ylabel('estimated distance (m)')
    ax.set_title('(1) true vs estimated distance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2) mean abs error per true distance (bar, grouped by sample_group)
    ax = axes[0, 1]
    stats = per_group_stats(df)
    piv = stats.pivot(index='true_distance_m', columns='sample_group',
                      values='MAE_mm')
    piv.plot(kind='bar', ax=ax)
    ax.set_xlabel('true distance (m)')
    ax.set_ylabel('MAE (mm)')
    ax.set_title('(2) mean absolute error per distance')
    ax.legend(title='group', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 3) error vs distance curve (mean +/- std)
    ax = axes[1, 0]
    for group, g in df.groupby('sample_group'):
        gb = g.groupby('true_distance_m')['err'].agg(['mean', 'std'])
        x = gb.index
        ax.errorbar(x, gb['mean'] * 1000, yerr=gb['std'] * 1000, marker='o',
                    capsize=3, label=group)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('true distance (m)')
    ax.set_ylabel('signed error (mm)')
    ax.set_title('(3) error vs distance (mean +/- std)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) front vs tilted MAE comparison
    ax = axes[1, 1]
    if 'front' in stats['sample_group'].values and 'tilted' in stats['sample_group'].values:
        comp = stats.pivot(index='true_distance_m', columns='sample_group',
                           values='MAE_mm')
        comp.plot(kind='bar', ax=ax)
        ax.set_ylabel('MAE (mm)')
        ax.set_title('(4) front vs tilted MAE')
        ax.legend(title='group', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, '需要 front 与 tilted 两组数据', ha='center', va='center')
        ax.set_title('(4) front vs tilted (缺数据)')

    fig.suptitle('AprilTag distance error analysis', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def write_report(df: pd.DataFrame, stats: pd.DataFrame, overall: dict,
                 out_md: Path, out_png: Path) -> None:
    lines = []
    lines.append('# AprilTag 距离误差实验报告\n')
    lines.append(f'- 样本总数: {overall["n"]}')
    lines.append(f'- 标签边长 (CSV): {df["tag_size_m"].iloc[0] if "tag_size_m" in df else "?"} m')
    lines.append(f'- 检测方法: {df["pose_method"].unique().tolist() if "pose_method" in df else "?"}')
    lines.append('')
    lines.append('## 全局统计\n')
    lines.append('| 指标 | 值 |')
    lines.append('|---|---|')
    for k, v in overall.items():
        unit = 'mm' if k != 'n' and k != 'mean_rel_err_pct' else ('%' if 'pct' in k else '')
        lines.append(f'| {k} | {v:.2f} {unit} |' if isinstance(v, float) else f'| {k} | {v} |')
    lines.append('')
    lines.append('## 分组统计 (真实距离 x 姿态)\n')
    lines.append(stats.to_markdown(index=False, floatfmt='.2f'))
    lines.append('')
    lines.append('## 误差图\n')
    lines.append(f'![distance error plot]({out_png.name})\n')
    lines.append('## 误差来源分析\n')
    sources = [
        ('相机标定误差', 'fx/fy/cx/cy 估计偏差直接放大为深度误差; 标定采集姿态不足或棋盘格不平整会引入系统偏差。'),
        ('AprilTag 实际尺寸测量误差 / 打印缩放', 'solvePnP 用 tag_size 把像素缩放到米, tag_size 偏差会使估计距离整体成比例偏大/偏小 (典型症状: 全部距离同向偏差)。'),
        ('标签纸不平整', '角点不在同一平面, 违反 PnP 平面假设, 倾斜时误差显著增大。'),
        ('镜头畸变 / 去畸变残留', 'image_proc 校正不彻底 (尤其画面边缘), 边缘检测误差大。'),
        ('自动对焦变化', '焦距变化导致内参漂移; 应锁定对焦后再标定。'),
        ('图像模糊 / 光照反射', '运动模糊或反光降低角点定位精度与 decision_margin。'),
        ('标签在画面边缘', '畸变残留 + 边缘解析力下降, 误差增大。'),
        ('远距离标签像素不足', '标签成像过小, 角点亚像素精度下降, 远距离误差升高。'),
        ('标签倾斜', '透视形变 + 尺寸测量基准变化, tilted 组误差通常高于 front 组。'),
        ('人工真实距离测量基准不准', '卷尺未对准相机光心与标签中心, 引入基准误差 (尤其近距离)。'),
    ]
    lines.append('| 来源 | 说明 |')
    lines.append('|---|---|')
    for name, desc in sources:
        lines.append(f'| {name} | {desc} |')
    lines.append('')
    lines.append('## 改进建议\n')
    lines.append('- 重新标定并确保覆盖画面四角与不同倾斜; 锁定对焦。')
    lines.append('- 用游标卡尺精确测量 tag_size, 同步更新 apriltag.yaml 与 launch tag_size。')
    lines.append('- 标签贴在硬质平整背板上, 避免反光。')
    lines.append('- 优先在画面中心区域采集, 远距离可换更大标签或更长焦距。')
    lines.append('- 若整体偏差恒定, 优先复核 tag_size 单位 (米 vs 毫米) 与测量基准点。')
    out_md.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    out_dir = csv_path.parent if len(sys.argv) > 1 else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / 'distance_summary.csv'
    out_png = out_dir / 'distance_error_plot.png'
    out_md = out_dir / 'distance_report.md'

    df = load(csv_path)
    if df.empty:
        sys.exit('CSV 为空, 无可分析数据。')
    stats = per_group_stats(df)
    overall = overall_stats(df)
    stats.to_csv(summary_csv, index=False, float_format='%.4f')
    make_plots(df, out_png)
    write_report(df, stats, overall, out_md, out_png)

    print(f'读入 {len(df)} 行: {csv_path}')
    print(f'分组统计 -> {summary_csv}')
    print(f'误差图   -> {out_png}')
    print(f'报告     -> {out_md}')
    print('\n全局: ' + ', '.join(f'{k}={v:.2f}' for k, v in overall.items()))


if __name__ == '__main__':
    main()
