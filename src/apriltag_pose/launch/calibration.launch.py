"""calibration.launch.py - 启动 v4l2_camera + cameracalibrator 棋盘格标定.

标定前请:
  1. 打印 9x6 内部角点、方格边长已用尺子测量的棋盘格 (默认 0.025 m).
  2. 用最终运行时相同的分辨率 (默认 640x480) 标定.
  3. 锁定对焦 (如可能).

标定完成后, cameracalibrator 会写入 ost.yaml. 把它复制为
config/camera.yaml (camera_calibration 的标准 8 段格式).

示例:
    ros2 launch apriltag_pose calibration.launch.py
    ros2 launch apriltag_pose calibration.launch.py \
        video_device:=/dev/video0 size:=0.025 checkerboard:=9x6
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

_DEFAULT_VIDEO_DEVICE = (
    '/dev/v4l/by-id/usb-HD_Camera_Manufacturer_USB_2.0_Camera-video-index0')


def _build(context, *_args, **_kwargs):
    def s(n): return LaunchConfiguration(n).perform(context)

    size = s('size')                 # e.g. "0.025"
    cb = s('checkerboard')           # e.g. "9x6"  (interior corners: cols x rows)

    actions = [Node(
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
    ), Node(
        package='camera_calibration',
        executable='cameracalibrator',
        name='cameracalibrator',
        output='screen',
        # cameracalibrator subscribes image + camera_info and publishes
        # camera_info on the same namespace. Remap to the /camera topics.
        remappings=[
            ('image', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
        ],
        arguments=[
            '--size', cb,
            '--square', size,
            '--no-service-check',
            'camera:=/camera',
        ],
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
        # 标定是一次性流程, 帧率无关紧要; 保持 rgb8 以最大化
        # cameracalibrator 的兼容性 (它内部按彩色图处理).
        DeclareLaunchArgument('output_encoding', default_value='rgb8'),
        DeclareLaunchArgument('frame_rate', default_value='30.0'),
        DeclareLaunchArgument('size', default_value='0.025',
                              description='square edge length in meters (measure it!)'),
        DeclareLaunchArgument('checkerboard', default_value='9x6',
                              description='interior corners cols x rows, e.g. 9x6'),
        OpaqueFunction(function=_build),
    ])
