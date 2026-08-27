"""ar_geometry 的单元测试 - 拓展题 1.2 的数学正确性验证.

不依赖 ROS 和摄像头, 用解析构造的位姿验证:
    1. 物体在 tag(世界)系中的坐标恒定 -> "相对世界系静止"
    2. 相机移动/旋转时投影按真实透视关系变化 -> 近大远小、位置改变
    3. 手算投影 vs cv2.projectPoints 一致 -> 投影链路无误
    4. 背面剔除与画家算法排序正确
    5. 顶点跑到相机背后时拒绝渲染 (避免画出乱线)
"""

import numpy as np
import pytest

import cv2

from apriltag_pose.ar_geometry import (
    Mesh, build_object, draw_mesh, face_depth, is_face_visible,
    is_in_front_of_camera, make_arrow, make_cube, make_pyramid,
    project_points, sort_faces_back_to_front, transform_points_to_camera,
    verify_static_in_world,
)

# 一个典型的 640x480 内参
K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0],
              [0.0, 0.0, 1.0]])

# "标签正对相机": tag 的 +z 指向相机 => 绕 x 轴转 180 度
RVEC_FACING = np.array([[np.pi], [0.0], [0.0]])
TVEC_50CM = np.array([[0.0], [0.0], [0.50]])

TAG_SIZE = 0.08
CUBE_SIZE = TAG_SIZE * 0.5          # 40 mm
OFFSET = (0.0, 0.0, 0.10)           # 沿法线抬升 100 mm


# ---------------------------------------------------------------------------
# 模型构造
# ---------------------------------------------------------------------------

class TestMeshBuilders:
    def test_cube_has_8_vertices_6_faces_12_edges(self):
        m = make_cube(CUBE_SIZE, OFFSET)
        assert m.vertices.shape == (8, 3)
        assert len(m.faces) == 6
        assert len(m.edges) == 12
        assert len(m.face_colors) == 6

    def test_cube_centred_on_offset_with_correct_size(self):
        m = make_cube(CUBE_SIZE, OFFSET)
        assert np.allclose(m.vertices.mean(axis=0), OFFSET)
        # 每个轴的跨度都等于边长
        span = m.vertices.max(axis=0) - m.vertices.min(axis=0)
        assert np.allclose(span, CUBE_SIZE)

    def test_cube_does_not_overlap_the_tag_plane(self):
        """题目要求: 虚拟物体应与标签有三维偏移, 不能完全重合."""
        m = make_cube(CUBE_SIZE, OFFSET)
        # 偏移 100 mm, 半边长 20 mm -> 整体在 z >= 80 mm, 完全脱离 z=0 标签平面
        assert m.vertices[:, 2].min() > 0.0

    def test_pyramid_and_arrow_are_wellformed(self):
        for m in (make_pyramid(CUBE_SIZE, OFFSET), make_arrow(0.10, OFFSET)):
            assert m.vertices.shape[1] == 3
            assert len(m.faces) >= 4
            assert len(m.edges) >= 6
            assert len(m.face_colors) == len(m.faces)

    def test_build_object_dispatch(self):
        for name in ('cube', 'pyramid', 'arrow'):
            assert isinstance(build_object(name, CUBE_SIZE, OFFSET), Mesh)
        # 大小写/空格容错
        assert isinstance(build_object('  CUBE ', CUBE_SIZE, OFFSET), Mesh)

    def test_build_object_rejects_unknown_type(self):
        with pytest.raises(ValueError, match='unknown object_type'):
            build_object('teapot', CUBE_SIZE, OFFSET)

    def test_builders_reject_nonpositive_size(self):
        for fn in (make_cube, make_pyramid, make_arrow):
            with pytest.raises(ValueError):
                fn(0.0, OFFSET)
            with pytest.raises(ValueError):
                fn(-0.01, OFFSET)

    def test_mesh_validates_indices(self):
        v = np.zeros((3, 3))
        with pytest.raises(ValueError, match='out of range'):
            Mesh(vertices=v, faces=[(0, 1, 5)], edges=[])
        with pytest.raises(ValueError, match='out of range'):
            Mesh(vertices=v, faces=[(0, 1, 2)], edges=[(0, 9)])
        with pytest.raises(ValueError, match='must be'):
            Mesh(vertices=np.zeros((3, 2)), faces=[], edges=[])


# ---------------------------------------------------------------------------
# 核心性质: 相对世界系静止
# ---------------------------------------------------------------------------

class TestStaticInWorldFrame:
    def test_tag_frame_coords_invariant_camera_frame_coords_change(self):
        """物体在 tag 系坐标不变, 在相机系坐标随位姿变化 - 这就是 1.2 的要求."""
        m = make_cube(CUBE_SIZE, OFFSET)
        poses = [
            (RVEC_FACING, TVEC_50CM),
            (np.array([[np.pi], [0.35], [0.0]]), np.array([[0.05], [0.0], [0.60]])),
            (np.array([[2.9], [0.0], [0.5]]), np.array([[-0.03], [0.02], [0.45]])),
            (np.array([[np.pi], [-0.5], [0.2]]), np.array([[0.0], [0.04], [0.80]])),
        ]
        out = verify_static_in_world(m, poses)
        # tag 系: 严格为 0
        assert out['max_tag_frame_delta_m'] == 0.0
        # 相机系: 必须真的变了 (否则说明变换没生效)
        assert out['max_camera_frame_delta_m'] > 0.05

    def test_verify_static_needs_two_poses(self):
        m = make_cube(CUBE_SIZE, OFFSET)
        with pytest.raises(ValueError, match='at least 2'):
            verify_static_in_world(m, [(RVEC_FACING, TVEC_50CM)])

    def test_object_gets_smaller_when_camera_moves_away(self):
        """近大远小: 相机后退, 投影面积必须变小."""
        m = make_cube(CUBE_SIZE, OFFSET)

        def area(depth):
            p = project_points(m.vertices, RVEC_FACING,
                               np.array([[0.0], [0.0], [depth]]), K, None)
            return ((p[:, 0].max() - p[:, 0].min()) *
                    (p[:, 1].max() - p[:, 1].min()))

        a_near, a_mid, a_far = area(0.40), area(0.80), area(1.60)
        assert a_near > a_mid > a_far
        # 距离翻倍 -> 线性尺寸减半 -> 面积约 1/4
        assert a_mid / a_far == pytest.approx(4.0, rel=0.25)

    def test_object_shifts_in_image_when_camera_translates_sideways(self):
        """相机横移, 物体在图像中的水平位置必须跟着变."""
        m = make_cube(CUBE_SIZE, OFFSET)
        c0 = project_points(m.vertices, RVEC_FACING, TVEC_50CM, K, None).mean(axis=0)
        c1 = project_points(m.vertices, RVEC_FACING,
                            np.array([[0.10], [0.0], [0.50]]), K, None).mean(axis=0)
        assert c1[0] - c0[0] > 50.0            # tx=+0.1m, f=600, z=0.4 -> ~150 px
        assert abs(c1[1] - c0[1]) < 1.0        # 垂直方向基本不变


# ---------------------------------------------------------------------------
# 投影链路正确性
# ---------------------------------------------------------------------------

class TestProjection:
    def test_transform_matches_manual_R_t(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.01, -0.02, 0.03]])
        rvec = np.array([[0.3], [-0.2], [0.1]])
        tvec = np.array([[0.02], [0.03], [0.5]])
        R, _ = cv2.Rodrigues(rvec)
        expect = (R @ pts.T).T + tvec.reshape(3)
        assert np.allclose(transform_points_to_camera(pts, rvec, tvec), expect)

    def test_projection_matches_manual_pinhole_math(self):
        """手算 u = fx*X/Z + cx 与 projectPoints 必须一致 (零畸变)."""
        m = make_cube(CUBE_SIZE, OFFSET)
        rvec = np.array([[np.pi], [0.2], [-0.1]])
        tvec = np.array([[0.01], [0.02], [0.55]])
        got = project_points(m.vertices, rvec, tvec, K, None)

        cam = transform_points_to_camera(m.vertices, rvec, tvec)
        manual = np.column_stack([
            K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2],
            K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2],
        ])
        assert np.allclose(got, manual, atol=1e-6)

    def test_tag_origin_projects_to_principal_point_when_centred(self):
        """标签正对且居中时, tag 原点必须投到主点."""
        p = project_points(np.zeros((1, 3)), RVEC_FACING, TVEC_50CM, K, None)
        assert p[0] == pytest.approx([K[0, 2], K[1, 2]], abs=1e-6)

    def test_distortion_coefficients_change_the_projection(self):
        m = make_cube(CUBE_SIZE, OFFSET)
        rvec = np.array([[np.pi], [0.3], [0.0]])
        tvec = np.array([[0.06], [0.04], [0.45]])
        undist = project_points(m.vertices, rvec, tvec, K, None)
        dist = project_points(m.vertices, rvec, tvec, K,
                              np.array([-0.3, 0.1, 0.0, 0.0, 0.0]))
        assert not np.allclose(undist, dist)


# ---------------------------------------------------------------------------
# 可见性: 背面剔除 + 画家算法
# ---------------------------------------------------------------------------

class TestVisibility:
    def test_only_front_face_visible_when_facing_camera(self):
        """立方体正对相机时, 6 个面里只有朝向相机的那个可见."""
        m = make_cube(CUBE_SIZE, OFFSET)
        cam = transform_points_to_camera(m.vertices, RVEC_FACING, TVEC_50CM)
        visible = [i for i, f in enumerate(m.faces) if is_face_visible(cam, f)]
        assert len(visible) == 1
        # face 1 = z+ 面; tag +z 朝相机, 所以它是最靠近相机的面
        assert visible == [1]
        assert face_depth(cam, m.faces[1]) < face_depth(cam, m.faces[0])

    def test_oblique_view_shows_three_faces(self):
        """斜视时立方体应看到 2~3 个面 (符合真实透视)."""
        m = make_cube(CUBE_SIZE, OFFSET)
        rvec = np.array([[2.7], [0.6], [0.3]])
        cam = transform_points_to_camera(m.vertices, rvec,
                                         np.array([[0.0], [0.0], [0.5]]))
        visible = [i for i, f in enumerate(m.faces) if is_face_visible(cam, f)]
        assert 2 <= len(visible) <= 3

    def test_painter_order_is_far_to_near(self):
        m = make_cube(CUBE_SIZE, OFFSET)
        cam = transform_points_to_camera(m.vertices, RVEC_FACING, TVEC_50CM)
        order = sort_faces_back_to_front(cam, m.faces)
        depths = [face_depth(cam, m.faces[i]) for i in order]
        assert depths == sorted(depths, reverse=True)
        assert len(order) == len(m.faces)

    def test_rejects_geometry_behind_camera(self):
        m = make_cube(CUBE_SIZE, OFFSET)
        # tag 在相机后方
        cam = transform_points_to_camera(m.vertices, RVEC_FACING,
                                         np.array([[0.0], [0.0], [-0.5]]))
        assert not is_in_front_of_camera(cam)
        # 正常情况必须通过
        ok = transform_points_to_camera(m.vertices, RVEC_FACING, TVEC_50CM)
        assert is_in_front_of_camera(ok)


# ---------------------------------------------------------------------------
# 绘制
# ---------------------------------------------------------------------------

class TestDrawing:
    def test_draw_mesh_modifies_image_and_reports_true(self):
        img = np.zeros((480, 640, 3), np.uint8)
        m = make_cube(CUBE_SIZE, OFFSET)
        assert draw_mesh(img, m, RVEC_FACING, TVEC_50CM, K) is True
        assert img.any(), 'draw_mesh reported success but drew nothing'

    def test_draw_mesh_skips_when_behind_camera(self):
        img = np.zeros((480, 640, 3), np.uint8)
        m = make_cube(CUBE_SIZE, OFFSET)
        assert draw_mesh(img, m, RVEC_FACING,
                         np.array([[0.0], [0.0], [-0.5]]), K) is False
        assert not img.any(), 'nothing should be drawn for geometry behind camera'

    def test_draw_mesh_skips_when_far_outside_frame(self):
        img = np.zeros((480, 640, 3), np.uint8)
        m = make_cube(CUBE_SIZE, OFFSET)
        # 极端横向偏移, 投影远在画面之外
        assert draw_mesh(img, m, RVEC_FACING,
                         np.array([[50.0], [0.0], [0.5]]), K) is False
        assert not img.any()

    def test_fill_alpha_zero_still_draws_edges(self):
        img = np.zeros((480, 640, 3), np.uint8)
        m = make_cube(CUBE_SIZE, OFFSET)
        assert draw_mesh(img, m, RVEC_FACING, TVEC_50CM, K,
                         fill_alpha=0.0) is True
        assert img.any()
