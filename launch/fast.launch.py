"""fast.launch.py — starts the node and prepares it to be used without manual lifecycle calls.

Sequence:
  1. LifecycleNode starts  (state: unconfigured)
  2. OnProcessStart fires  → TimerAction(0.5 s) → configure
  3. Model loads (warm-up runs inside on_configure)
  4. OnStateTransition(inactive) fires → activate
  5. Node is live (state: active)

Requirements: activate your ML venv before launching so python3 resolves
to the interpreter that has torch/transformers installed. See README.
"""

import os
import launch
from launch import LaunchDescription
from launch.actions import (
    RegisterEventHandler,
    EmitEvent,
    TimerAction,
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import OnProcessStart

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

    # 1. The main node
    node = LifecycleNode(
        package='transformers_bridge',
        namespace='transformers',
        executable='detector',
        name='transformers_node',
        output='both',
        emulate_tty=True,
        parameters=node_params,
        remappings=[
            ('/camera/image_raw', '/camera2/left/image_raw'),
        ],
    )

    # Lifecycle events
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matchers.matches_action(node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )

    on_process_start = RegisterEventHandler(
        OnProcessStart(
            target_action=node,
            on_start=[TimerAction(period=0.5, actions=[configure_event])],
        )
    )

    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch.events.matchers.matches_action(node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        )
    )

    on_inactive = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=node,
            goal_state='inactive',
            entities=[activate_event],
        )
    )

    return [node, on_process_start, on_inactive]


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
