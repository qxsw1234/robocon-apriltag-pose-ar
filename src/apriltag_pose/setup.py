import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'apriltag_pose'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.sh')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='czm',
    maintainer_email='czm@example.com',
    description='ROS 2 + OpenCV + AprilTag camera pose estimation and distance validation.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tag_visualizer_node = apriltag_pose.tag_visualizer_node:main',
            'ar_object_node = apriltag_pose.ar_object_node:main',
            'distance_recorder_node = apriltag_pose.distance_recorder_node:main',
            'calibration_checker_node = apriltag_pose.calibration_checker_node:main',
        ],
    },
)
