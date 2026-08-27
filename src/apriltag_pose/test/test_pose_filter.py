"""pose_filter 的单元测试 - 稳定性改进的正确性验证.

覆盖:
    1. 四元数符号对齐与平均 (q/-q 等价问题)
    2. 平面 PnP 二义性: 候选解枚举与基于时序连续性的择优
    3. 滑动窗口滤波确实降低抖动
    4. 异常帧剔除 (低 decision_margin / 大重投影误差 / 位姿突变)
    5. 丢检保持与超时
"""

import numpy as np
import pytest

from apriltag_pose.ar_geometry import project_points
from apriltag_pose.pose_filter import (
    PoseStabilizer, align_quaternion_sign, average_quaternions,
    choose_candidate, quaternion_angle_deg, quaternion_from_rvec,
    rvec_from_quaternion, solve_tag_pose_candidates,
)

K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0],
              [0.0, 0.0, 1.0]])
TAG = 0.08


def tag_corners_apriltag_order(size=TAG):
    """apriltag 原生顺序 BL, BR, TR, TL (tag 系)."""
    s = size / 2.0
    return np.array([[-s, -s, 0.0], [+s, -s, 0.0],
                     [+s, +s, 0.0], [-s, +s, 0.0]], dtype=np.float64)


def observe(rvec, tvec, noise_px=0.0, rng=None, size=TAG):
    """合成一次观测: 把 tag 4 角投到像素平面, 可叠加噪声."""
    pts = project_points(tag_corners_apriltag_order(size), rvec, tvec, K, None)
    if noise_px > 0.0:
        rng = rng or np.random.default_rng(0)
        pts = pts + rng.normal(0.0, noise_px, pts.shape)
    return pts


RVEC_TILT = np.array([[2.95], [0.35], [0.05]])
TVEC_TILT = np.array([[0.01], [0.0], [0.45]])
RVEC_FRONT = np.array([[np.pi], [0.0], [0.0]])
TVEC_FRONT = np.array([[0.0], [0.0], [0.45]])


# ---------------------------------------------------------------------------
class TestQuaternionHelpers:
    def test_rvec_quat_roundtrip(self):
        for rvec in (RVEC_TILT, RVEC_FRONT, np.array([[0.1], [-0.2], [0.3]])):
            q = quaternion_from_rvec(rvec)
            back = rvec_from_quaternion(q)
            # 同一旋转: 比较角度差而不是分量 (rvec 表示不唯一)
            assert quaternion_angle_deg(quaternion_from_rvec(back), q) < 1e-6

    def test_align_sign_flips_when_opposed(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        assert np.allclose(align_quaternion_sign(-q, q), q)
        assert np.allclose(align_quaternion_sign(q, q), q)

    def test_average_of_q_and_negative_q_is_not_zero(self):
        """这正是必须做符号对齐的原因: 直接平均 q 和 -q 会得到 0."""
        q = quaternion_from_rvec(RVEC_TILT)
        avg = average_quaternions([q, -q])
        assert np.isclose(np.linalg.norm(avg), 1.0)
        assert quaternion_angle_deg(avg, q) < 1e-6

    def test_average_is_between_inputs(self):
        q1 = quaternion_from_rvec(np.array([[np.pi], [0.0], [0.0]]))
        q2 = quaternion_from_rvec(np.array([[np.pi], [0.2], [0.0]]))
        avg = average_quaternions([q1, q2])
        a1 = quaternion_angle_deg(avg, q1)
        a2 = quaternion_angle_deg(avg, q2)
        full = quaternion_angle_deg(q1, q2)
        assert a1 < full and a2 < full
        assert a1 == pytest.approx(a2, abs=1.0)

    def test_angle_is_zero_for_identical_and_negated(self):
        q = quaternion_from_rvec(RVEC_TILT)
        assert quaternion_angle_deg(q, q) == pytest.approx(0.0, abs=1e-9)
        assert quaternion_angle_deg(q, -q) == pytest.approx(0.0, abs=1e-9)

    def test_average_rejects_empty(self):
        with pytest.raises(ValueError):
            average_quaternions([])


# ---------------------------------------------------------------------------
class TestPnPAmbiguity:
    def test_candidates_recover_the_true_pose(self):
        cands = solve_tag_pose_candidates(observe(RVEC_TILT, TVEC_TILT),
                                          TAG, K, None)
        assert len(cands) >= 1
        best = cands[0]
        assert best.reprojection_error_px < 0.01
        assert np.allclose(best.tvec.ravel(), TVEC_TILT.ravel(), atol=1e-4)

    def test_candidates_sorted_by_reprojection_error(self):
        cands = solve_tag_pose_candidates(observe(RVEC_TILT, TVEC_TILT),
                                          TAG, K, None)
        errs = [c.reprojection_error_px for c in cands]
        assert errs == sorted(errs)

    def test_all_candidates_are_in_front_of_camera(self):
        cands = solve_tag_pose_candidates(observe(RVEC_TILT, TVEC_TILT),
                                          TAG, K, None)
        assert all(c.tvec[2, 0] > 0 for c in cands)

    def test_invalid_intrinsics_yield_no_candidates(self):
        assert solve_tag_pose_candidates(observe(RVEC_TILT, TVEC_TILT), TAG,
                                         np.zeros((3, 3)), None) == []

    def test_choose_prefers_continuity_over_marginal_error(self):
        """两个候选误差相当时, 选与上一帧姿态更接近的那个 (而不是误差最小的)."""
        cands = solve_tag_pose_candidates(observe(RVEC_TILT, TVEC_TILT),
                                          TAG, K, None)
        if len(cands) < 2:
            pytest.skip('this view produced a single candidate')
        # 把"历史"设为第二个候选的姿态; 用足够大的绝对容差让它进入竞争
        prev = quaternion_from_rvec(cands[1].rvec)
        picked = choose_candidate(cands, prev, err_abs_tol_px=10.0)
        assert picked is cands[1], 'continuity should win when errors are comparable'

    def test_frontal_view_ambiguity_is_resolved_by_continuity(self):
        """近似正对是二义性最严重的情形: 两解重投影误差接近, 但姿态差很大.

        这里验证: 跟着历史走能稳定地选到同一侧的解, 从而避免姿态来回翻转。
        """
        rng = np.random.default_rng(3)
        obs = observe(RVEC_FRONT, TVEC_FRONT, noise_px=0.3, rng=rng)
        cands = solve_tag_pose_candidates(obs, TAG, K, None)
        if len(cands) < 2:
            pytest.skip('frontal view produced a single candidate')
        # 两解误差应当是"相当"的量级 (这就是二义性的定义)
        assert cands[1].reprojection_error_px < cands[0].reprojection_error_px + 1.0

        # 以每个候选为历史, 都应当选回它自己 -> 不会无故翻转
        for i in (0, 1):
            prev = quaternion_from_rvec(cands[i].rvec)
            picked = choose_candidate(cands, prev)
            assert quaternion_angle_deg(quaternion_from_rvec(picked.rvec),
                                        prev) < 1e-6

    def test_absolute_tolerance_matters_when_best_error_is_tiny(self):
        """纯相对容差在 best_err≈0 时会失效, 绝对容差必须兜住这种情况."""
        cands = solve_tag_pose_candidates(observe(RVEC_TILT, TVEC_TILT),
                                          TAG, K, None)
        if len(cands) < 2:
            pytest.skip('this view produced a single candidate')
        assert cands[0].reprojection_error_px < 1e-6, 'noise-free best should be ~0'
        prev = quaternion_from_rvec(cands[1].rvec)
        # ratio 容差设成 1.0 (等价于只接受最优解), 绝对容差仍应让第二解可选
        picked = choose_candidate(cands, prev, err_ratio_tol=1.0,
                                  err_abs_tol_px=10.0)
        assert picked is cands[1]

    def test_choose_without_history_returns_lowest_error(self):
        cands = solve_tag_pose_candidates(observe(RVEC_TILT, TVEC_TILT),
                                          TAG, K, None)
        assert choose_candidate(cands, None) is cands[0]

    def test_choose_handles_empty(self):
        assert choose_candidate([], None) is None


# ---------------------------------------------------------------------------
class TestFiltering:
    def test_filtering_reduces_jitter(self):
        """核心: 同一噪声序列, 滤波后的标准差必须明显小于原始解."""
        rng = np.random.default_rng(1234)
        raw, filt = [], []
        stab = PoseStabilizer(TAG, window=7)
        for i in range(80):
            obs = observe(RVEC_TILT, TVEC_TILT, noise_px=0.4, rng=rng)
            cands = solve_tag_pose_candidates(obs, TAG, K, None)
            if cands:
                raw.append(cands[0].tvec.ravel())
            out = stab.update(obs, 80.0, K, None, i / 30.0)
            if out is not None:
                filt.append(out.tvec.ravel())

        assert len(filt) > 60
        raw_std = np.asarray(raw).std(axis=0)
        filt_std = np.asarray(filt).std(axis=0)
        # 深度方向抖动最大, 滤波收益也最明显
        assert filt_std[2] < raw_std[2] * 0.7
        assert np.all(filt_std <= raw_std + 1e-9)

    def test_filtered_pose_stays_accurate(self):
        rng = np.random.default_rng(7)
        stab = PoseStabilizer(TAG, window=5)
        last = None
        for i in range(40):
            last = stab.update(observe(RVEC_TILT, TVEC_TILT, 0.3, rng),
                               80.0, K, None, i / 30.0) or last
        assert last is not None
        # 滤波不应引入偏置
        assert np.allclose(last.tvec.ravel(), TVEC_TILT.ravel(), atol=3e-3)

    def test_window_of_one_is_passthrough(self):
        stab = PoseStabilizer(TAG, window=1)
        out = stab.update(observe(RVEC_TILT, TVEC_TILT), 80.0, K, None, 0.0)
        assert out is not None and out.samples == 1
        assert np.allclose(out.tvec.ravel(), TVEC_TILT.ravel(), atol=1e-4)

    def test_samples_grows_up_to_window(self):
        stab = PoseStabilizer(TAG, window=4)
        seen = []
        for i in range(8):
            out = stab.update(observe(RVEC_TILT, TVEC_TILT), 80.0, K, None,
                              i / 30.0)
            seen.append(out.samples)
        assert seen == [1, 2, 3, 4, 4, 4, 4, 4]

    def test_constructor_validates_args(self):
        with pytest.raises(ValueError):
            PoseStabilizer(0.0)
        with pytest.raises(ValueError):
            PoseStabilizer(TAG, window=0)


# ---------------------------------------------------------------------------
class TestOutlierRejection:
    def test_low_decision_margin_is_rejected(self):
        stab = PoseStabilizer(TAG, min_decision_margin=30.0)
        # 第一帧就低于阈值 -> 没有历史可保持 -> None
        assert stab.update(observe(RVEC_TILT, TVEC_TILT), 5.0, K, None, 0.0) is None
        assert stab.rejected_total == 1

    def test_large_reprojection_error_is_rejected(self):
        """把角点打乱成不可能的四边形, 重投影误差必然很大."""
        stab = PoseStabilizer(TAG, max_reproj_error_px=0.5)
        bad = np.array([[100.0, 100.0], [500.0, 110.0],
                        [120.0, 400.0], [480.0, 380.0]])
        assert stab.update(bad, 80.0, K, None, 0.0) is None
        assert stab.rejected_total >= 1

    def test_position_jump_is_rejected_then_accepted_after_persistence(self):
        """单帧突变被剔除; 若持续存在则认定为真实运动并重新收敛."""
        stab = PoseStabilizer(TAG, window=5, max_jump_m=0.10, reset_after=3,
                              hold_sec=10.0)
        for i in range(6):
            stab.update(observe(RVEC_TILT, TVEC_TILT), 80.0, K, None, i / 30.0)
        baseline = stab.last.tvec.ravel().copy()

        far_tvec = np.array([[0.01], [0.0], [1.20]])       # 跳变 0.75 m
        out = stab.update(observe(RVEC_TILT, far_tvec), 80.0, K, None, 0.3)
        # 第一次被当作异常 -> 输出保持旧值
        assert out is not None and out.held
        assert np.allclose(out.tvec.ravel(), baseline)

        # 持续出现 -> 最终接受并向新位置收敛
        for i in range(8):
            out = stab.update(observe(RVEC_TILT, far_tvec), 80.0, K, None,
                              0.4 + i / 30.0)
        assert out is not None
        assert out.tvec.ravel()[2] > 1.0

    def test_rejected_counter_accumulates(self):
        stab = PoseStabilizer(TAG, min_decision_margin=30.0)
        for i in range(3):
            stab.update(observe(RVEC_TILT, TVEC_TILT), 1.0, K, None, i / 30.0)
        assert stab.rejected_total == 3


# ---------------------------------------------------------------------------
class TestHoldOnDroppedDetection:
    def test_hold_then_expire(self):
        stab = PoseStabilizer(TAG, hold_sec=0.2)
        good = stab.update(observe(RVEC_TILT, TVEC_TILT), 80.0, K, None, 0.0)
        assert good is not None and not good.held

        # 丢检但在保持窗口内 -> 沿用上次结果
        held = stab.update(None, None, K, None, 0.1)
        assert held is not None and held.held
        assert np.allclose(held.tvec, good.tvec)

        # 超过保持时间 -> 不再输出
        assert stab.update(None, None, K, None, 0.5) is None

    def test_hold_without_history_returns_none(self):
        stab = PoseStabilizer(TAG)
        assert stab.update(None, None, K, None, 0.0) is None

    def test_reset_clears_window_but_keeps_last(self):
        stab = PoseStabilizer(TAG, window=5)
        for i in range(5):
            stab.update(observe(RVEC_TILT, TVEC_TILT), 80.0, K, None, i / 30.0)
        assert stab.last is not None
        stab.reset()
        out = stab.update(observe(RVEC_TILT, TVEC_TILT), 80.0, K, None, 1.0)
        assert out.samples == 1
