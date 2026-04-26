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
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    debug_val = LaunchConfiguration('debug').perform(context).lower()
    params_file = LaunchConfiguration('params_file').perform(context)

    # Base parameters from YAML
    node_params = [params_file]

    # Only override debug if explicitly set to true or false
    if debug_val == 'true':
        node_params.append({'debug': True})
    elif debug_val == 'false':
        node_params.append({'debug': False})
    # If 'none' (default), we don't append anything, so the YAML value wins.

    return [
        LifecycleNode(
            package='transformers_bridge',
            namespace='transformers',
            executable='detector',
            name='transformers_node',
            output='both',
            emulate_tty=True,
            parameters=node_params,
            remappings=[
                ('/camera/image_raw', '/camera/image_raw'),
            ],
        )
    ]


def generate_launch_description():
    params_path = os.path.join(
        get_package_share_directory('transformers_bridge'),
        'config', 'default.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=params_path,
        description='Path to the ROS 2 parameters file to use'
    )

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='none',
        description='Override debug parameter (true/false). If "none", uses value from params_file.'
    )

    return LaunchDescription([
        params_file_arg,
        debug_arg,
        OpaqueFunction(function=launch_setup)
    ])