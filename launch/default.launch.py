"""default.launch.py — starts the node in 'unconfigured' state.

Drive lifecycle manually after launching:
  ros2 lifecycle set /transformers/transformers_node configure
  ros2 lifecycle set /transformers/transformers_node activate

For venv-based setups, use venv.launch.py or source the helper first:
  source <(ros2 run transformers_bridge venv_setup)
  ros2 launch transformers_bridge default.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LifecycleNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('transformers_bridge'),
        'config', 'rtdetrv2.yaml')

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='False',
        description='Enable debug image publishing with bounding boxes'
    )
    
    debug_eval = PythonExpression(["'", LaunchConfiguration('debug'), "'.lower() == 'true'"])

    return LaunchDescription([
        debug_arg,
        LifecycleNode(
            package='transformers_bridge',
            namespace='transformers',
            executable='detrv2',
            name='transformers_node',
            output='both',
            emulate_tty=True,
            parameters=[params, {'debug': debug_eval}],
            remappings=[
                ('/camera/image_raw', '/camera/image_raw'),
                # ('detections', '/robot/detections'),
                # ('debug_image', '/robot/debug_image'),
            ],
        ),
    ])