"""camera_only.launch.py - 只启动 v4l2_camera, 用于预览和排查.

发布:
    /camera/image_raw
    /camera/camera_info

附带一个 camera_link -> camera_optical_frame 静态 TF, 方便 RViz 显示.

示例:
    ros2 launch apriltag_pose camera_only.launch.py
    ros2 launch apriltag_pose camera_only.launch.py \
        video_device:=/dev/video0 pixel_format:=YUYV
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

_DEFAULT_VIDEO_DEVICE = (
    '/dev/v4l/by-id/usb-HD_Camera_Manufacturer_USB_2.0_Camera-video-index0')


def _build(context, *_args, **_kwargs):
    def s(n): return LaunchConfiguration(n).perform(context)

    actions = [Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_to_optical',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5',
                   '--frame-id', 'camera_link',
                   '--child-frame-id', s('camera_frame_id')],
    ), Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera_node',
        namespace='camera',
        output='screen',
        parameters=[{
            'video_device': s('video_device'),
            'pixel_format': s('pixel_format'),
            'output_encoding': s('output_encoding'),
            'image_size': [int(s('image_width')), int(s('image_height'))],
            'time_per_frame': [1, max(1, int(round(float(s('frame_rate')))))],
            'camera_frame_id': s('camera_frame_id'),
            'camera_name': s('camera_name'),
        }],
    )]
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('video_device', default_value=_DEFAULT_VIDEO_DEVICE),
        DeclareLaunchArgument('camera_name', default_value='usb_camera'),
        DeclareLaunchArgument('camera_frame_id', default_value='camera_optical_frame'),
        DeclareLaunchArgument('image_width', default_value='640'),
        DeclareLaunchArgument('image_height', default_value='480'),
        DeclareLaunchArgument('pixel_format', default_value='YUYV'),
        DeclareLaunchArgument('output_encoding', default_value='mono8',
                              description='mono8 (~30 Hz) | rgb8 (~8 Hz)'),
        DeclareLaunchArgument('frame_rate', default_value='30.0'),
        OpaqueFunction(function=_build),
    ])
