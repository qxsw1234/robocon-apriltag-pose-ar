"""apriltag_pose.launch.py - 一键启动完整 AprilTag 相机位姿系统.

启动顺序:
    0. static TF:  camera_link -> camera_optical_frame  (REP-103 body->optical)
    1. v4l2_camera          -> /camera/image_raw, /camera/camera_info
    2. image_proc (Rectify) -> /camera/image_rect           (use_rectified:=true)
    3. apriltag_ros         -> /apriltag/detections, TF (camera_optical_frame->tag_0)
    4. tag_visualizer_node  -> /apriltag/image_annotated, /apriltag/pose,
                              /apriltag/distance, /apriltag/markers
    5. ar_object_node       -> /apriltag/image_ar, /apriltag/ar_marker
                              (拓展题 1.2, enable_ar:=true, 默认开)
    6. distance_recorder_node (optional, enable_distance_recorder:=true)
    7. calibration_checker_node (optional, run_calibration_check:=true)
    8. rviz2 (optional, use_rviz:=true)

外置 USB 摄像头和内置摄像头用同一个 launch，只改 video_device 和 camera_info_file。

示例:
    ros2 launch apriltag_pose apriltag_pose.launch.py
    ros2 launch apriltag_pose apriltag_pose.launch.py \
        video_device:=/dev/video0 tag_size:=0.080 use_rviz:=true
"""

import os
from pathlib import Path
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


# REP-103: camera_link (x-fwd, y-left, z-up) -> camera_optical_frame
# (x-right, y-down, z-fwd). Quaternion (x,y,z,w) = (-0.5, 0.5, -0.5, 0.5).
_OPTICAL_QX, _OPTICAL_QY, _OPTICAL_QZ, _OPTICAL_QW = '-0.5', '0.5', '-0.5', '0.5'

# Stable by-id path of the external USB camera (avoids /dev/videoN churn across
# replugs). Verified formats: YUYV 640x480@30, MJPG 640x480@120.
# Override with video_device:=... to use another camera; the laptop's built-in one
# is /dev/v4l/by-id/usb-SunplusIT_Inc_Integrated_RGB_Camera_01.00.00-video-index0
_DEFAULT_VIDEO_DEVICE = (
    '/dev/v4l/by-id/usb-HD_Camera_Manufacturer_USB_2.0_Camera-video-index0')


def _declare_args() -> List[DeclareLaunchArgument]:
    return [
        # ---- camera
        DeclareLaunchArgument('video_device', default_value=_DEFAULT_VIDEO_DEVICE),
        DeclareLaunchArgument('camera_name', default_value='usb_camera'),
        DeclareLaunchArgument('camera_frame_id', default_value='camera_optical_frame'),
        DeclareLaunchArgument('image_width', default_value='640'),
        DeclareLaunchArgument('image_height', default_value='480'),
        # v4l2_camera on Humble handles YUYV reliably. MJPG is NOT usable here:
        # the node logs "Current pixel format is not supported yet: MJPG" and
        # then aborts in cv_bridge, so don't switch to it.
        DeclareLaunchArgument('pixel_format', default_value='YUYV'),
        # output_encoding is a real throughput lever, measured on this camera
        # at 640x480: rgb8 -> ~8 Hz, mono8 -> ~30 Hz. The YUYV->rgb8 colour
        # conversion in v4l2_camera is the bottleneck. AprilTag detection is
        # grayscale anyway, and cv_bridge promotes mono8 back to 3-channel BGR
        # for the annotation overlays, so mono8 costs nothing visually.
        # Use rgb8 only if you specifically need a colour image.
        DeclareLaunchArgument('output_encoding', default_value='mono8',
                              description='mono8 (~30 Hz) | rgb8 (~8 Hz)'),
        DeclareLaunchArgument('frame_rate', default_value='30.0'),
        DeclareLaunchArgument('camera_info_file', default_value='camera.yaml',
                              description='Name in config/ or an absolute path '
                                          'to a camera_calibration YAML.'),
        # ---- apriltag
        DeclareLaunchArgument('tag_family', default_value='36h11'),
        DeclareLaunchArgument('tag_id', default_value='0'),
        DeclareLaunchArgument('tag_size', default_value='0.080',
                              description='Tag edge length in METERS (measure it!).'),
        DeclareLaunchArgument('tag_frame', default_value='tag_0'),
        DeclareLaunchArgument('pose_method', default_value='pnp',
                              description='apriltag_ros pose method: pnp |.'),
        # ---- visualizer
        DeclareLaunchArgument('axis_length', default_value='0.040'),
        DeclareLaunchArgument('use_rectified_image', default_value='true'),
        # ---- AR (拓展题 1.2): tag 系作为世界系, 放一个固定的虚拟三维物体
        DeclareLaunchArgument('enable_ar', default_value='true'),
        DeclareLaunchArgument('ar_object_type', default_value='cube',
                              description='cube | pyramid | arrow'),
        DeclareLaunchArgument('ar_object_size_m', default_value='0.0',
                              description='object size in m; <=0 -> '
                                          'ar_object_size_ratio * tag_size'),
        DeclareLaunchArgument('ar_object_size_ratio', default_value='0.5'),
        DeclareLaunchArgument('ar_offset_x_m', default_value='0.0'),
        DeclareLaunchArgument('ar_offset_y_m', default_value='0.06',
                              description='offset along the tag +y (up) axis, m'),
        DeclareLaunchArgument('ar_offset_z_m', default_value='0.04',
                              description='offset along the tag normal (m). '
                                          'Offset must be non-zero so the object '
                                          'does not coincide with the tag.'),
        DeclareLaunchArgument('ar_fill_alpha', default_value='0.45'),
        # ---- 稳定性 (位姿滤波 / 异常帧剔除 / 丢检保持)
        DeclareLaunchArgument('filter_window', default_value='5',
                              description='sliding-window length in frames; '
                                          '1 disables filtering'),
        DeclareLaunchArgument('use_median_translation', default_value='true',
                              description='median (robust) vs mean (smoother)'),
        DeclareLaunchArgument('pose_hold_sec', default_value='0.25',
                              description='keep last pose this long when the '
                                          'detection drops out'),
        DeclareLaunchArgument('min_decision_margin', default_value='20.0'),
        DeclareLaunchArgument('max_reproj_error_px', default_value='4.0'),
        # ---- toggles
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('enable_distance_recorder', default_value='false'),
        DeclareLaunchArgument('run_calibration_check', default_value='false'),
    ]


def _resolve_camera_info(camera_info_file: str, pkg_share: str):
    """Return (url_for_v4l2, abs_path_for_checker).

    ('', '') when the file is not given/found -> uncalibrated.
    """
    if not camera_info_file:
        return '', ''
    p = Path(camera_info_file)
    if not p.is_absolute():
        p = Path(pkg_share) / 'config' / camera_info_file
    if not p.is_file():
        return '', ''
    p = p.resolve()
    return p.as_uri(), str(p)


def _build(context, *_args, **_kwargs):
    pkg_share = get_package_share_directory('apriltag_pose')

    def s(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    def b(name: str) -> bool:
        return s(name).lower() in ('true', '1', 'yes', 'on')

    video_device = s('video_device')
    camera_name = s('camera_name')
    camera_frame = s('camera_frame_id')
    image_w = int(s('image_width'))
    image_h = int(s('image_height'))
    pixel_format = s('pixel_format')
    output_encoding = s('output_encoding')
    frame_rate_f = float(s('frame_rate'))
    camera_info_url, camera_info_path = _resolve_camera_info(
        s('camera_info_file'), pkg_share)
    tag_size = float(s('tag_size'))
    tag_id = int(s('tag_id'))
    tag_frame = s('tag_frame')
    axis_length = float(s('axis_length'))
    use_rect = b('use_rectified_image')
    detection_image_topic = '/camera/image_rect' if use_rect else '/camera/image_raw'

    apriltag_yaml = os.path.join(pkg_share, 'config', 'apriltag.yaml')
    rviz_config = os.path.join(pkg_share, 'config', 'apriltag_pose.rviz')

    actions = []

    # ---- 0. static TF: camera_link -> camera_optical_frame ----
    actions.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_to_optical',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', _OPTICAL_QX, '--qy', _OPTICAL_QY,
                   '--qz', _OPTICAL_QZ, '--qw', _OPTICAL_QW,
                   '--frame-id', 'camera_link',
                   '--child-frame-id', camera_frame],
    ))

    # ---- 1. v4l2_camera ----
    actions.append(Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera_node',
        namespace='camera',
        output='screen',
        parameters=[{
            'video_device': video_device,
            'pixel_format': pixel_format,
            'output_encoding': output_encoding,
            'image_size': [image_w, image_h],
            'time_per_frame': [1, max(1, int(round(frame_rate_f)))],
            'camera_frame_id': camera_frame,
            'camera_name': camera_name,
            'camera_info_url': camera_info_url,
        }],
    ))

    # ---- 2. image_proc::RectifyNode (composable) ----
    if use_rect:
        actions.append(ComposableNodeContainer(
            name='image_proc_container',
            namespace='camera',
            package='rclcpp_components',
            executable='component_container',
            output='screen',
            composable_node_descriptions=[
                ComposableNode(
                    package='image_proc',
                    plugin='image_proc::RectifyNode',
                    name='rectify_node',
                    namespace='camera',
                    remappings=[('image', 'image_raw')],
                ),
            ],
        ))

    # ---- 3. apriltag_ros ----
    # apriltag_node (namespace /apriltag) subscribes to image_rect + camera_info
    # and publishes detections (+ TF when pose_estimation_method is set).
    # Remap both relative and absolute forms so it finds /camera/... regardless
    # of the exact topic-name convention used by the installed apriltag_ros.
    actions.append(Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_node',
        namespace='apriltag',
        output='screen',
        parameters=[apriltag_yaml],
        remappings=[
            ('image_rect', detection_image_topic),
            ('/apriltag/image_rect', detection_image_topic),
            ('camera_info', '/camera/camera_info'),
            ('/apriltag/camera_info', '/camera/camera_info'),
        ],
    ))

    # ---- 4. tag_visualizer_node ----
    actions.append(Node(
        package='apriltag_pose',
        executable='tag_visualizer_node',
        name='tag_visualizer_node',
        output='screen',
        parameters=[{
            'image_topic': detection_image_topic,
            'camera_info_topic': '/camera/camera_info',
            'detections_topic': '/apriltag/detections',
            'tag_size': tag_size,
            'axis_length': axis_length,
            'use_rectified_image': use_rect,
            'target_tag_id': tag_id,
            'optical_frame': camera_frame,
        }],
    ))

    # ---- 5. ar_object_node (拓展题 1.2, optional) ----
    actions.append(Node(
        package='apriltag_pose',
        executable='ar_object_node',
        name='ar_object_node',
        output='screen',
        condition=IfCondition(s('enable_ar')),
        parameters=[{
            'image_topic': detection_image_topic,
            'camera_info_topic': '/camera/camera_info',
            'detections_topic': '/apriltag/detections',
            'tag_size': tag_size,
            'target_tag_id': tag_id,
            'tag_frame': tag_frame,
            'use_rectified_image': use_rect,
            'object_type': s('ar_object_type'),
            'object_size_m': float(s('ar_object_size_m')),
            'object_size_ratio': float(s('ar_object_size_ratio')),
            'offset_xyz_m': [float(s('ar_offset_x_m')),
                             float(s('ar_offset_y_m')),
                             float(s('ar_offset_z_m'))],
            'fill_alpha': float(s('ar_fill_alpha')),
            'filter_window': int(s('filter_window')),
            'use_median_translation': b('use_median_translation'),
            'pose_hold_sec': float(s('pose_hold_sec')),
            'min_decision_margin': float(s('min_decision_margin')),
            'max_reproj_error_px': float(s('max_reproj_error_px')),
        }],
    ))

    # ---- 6. distance_recorder_node (optional) ----
    actions.append(Node(
        package='apriltag_pose',
        executable='distance_recorder_node',
        name='distance_recorder_node',
        output='screen',
        condition=IfCondition(s('enable_distance_recorder')),
        parameters=[{
            'tag_size': tag_size,
            'target_tag_id': tag_id,
            'camera_device': camera_name,
            'use_rectified_image': use_rect,
        }],
    ))

    # ---- 7. calibration_checker_node (optional) ----
    actions.append(Node(
        package='apriltag_pose',
        executable='calibration_checker_node',
        name='calibration_checker_node',
        output='screen',
        condition=IfCondition(s('run_calibration_check')),
        parameters=[{
            'calibration_file': camera_info_path,
            'camera_info_topic': '/camera/camera_info',
            'expected_width': image_w,
            'expected_height': image_h,
        }],
    ))

    # ---- 8. rviz2 (optional) ----
    if os.path.isfile(rviz_config):
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            condition=IfCondition(s('use_rviz')),
            arguments=['-d', rviz_config],
            output='screen',
        ))

    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(_declare_args() + [OpaqueFunction(function=_build)])
