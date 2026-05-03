"""default.launch.py — starts the node in 'unconfigured' state.

Drive lifecycle manually after launching:
  ros2 lifecycle set /transformers/transformers_node configure
  ros2 lifecycle set /transformers/transformers_node activate

Usage:
  ros2 launch transformers_bridge default.launch.py
  ros2 launch transformers_bridge default.launch.py params_file:=/path/to/yolo_ego.yaml
  ros2 launch transformers_bridge default.launch.py debug:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    params = [LaunchConfiguration('params_file').perform(context)]
    debug  = LaunchConfiguration('debug').perform(context).lower()
    if debug in ('true', 'false'):
        params.append({'debug': debug == 'true'})

    return [
        LifecycleNode(
            package='transformers_bridge',
            namespace='transformers',
            executable='detector',
            name='transformers_node',
            output='both',
            emulate_tty=True,
            parameters=params,
            remappings=[('/camera/image_raw', '/phone/image')],
        )
    ]


def generate_launch_description():
    params_path = os.path.join(
        get_package_share_directory('transformers_bridge'),
        'config', 'default.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=params_path,
            description='Path to the ROS2 parameters YAML file.',
        ),
        DeclareLaunchArgument(
            'debug', default_value='',
            description='Override debug flag (true/false). Empty = use YAML value.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
