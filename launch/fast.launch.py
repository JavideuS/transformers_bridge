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
)
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import OnProcessStart


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('transformers_bridge'),
        'config', 'rtdetrv2.yaml')

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='False',
        description='Enable debug image publishing'
    )
    debug_eval = PythonExpression(["'", LaunchConfiguration('debug'), "'.lower() == 'true'"])

    # 1. The main node in lifecycle unconfigured state
    node = LifecycleNode(
        package='transformers_bridge',
        namespace='transformers',
        executable='detrv2',
        name='transformers_node',
        output='both',
        emulate_tty=True,
        parameters=[params, {'debug': debug_eval}],
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
        ],
    )

    # Small delay (0.5 s) lets the node finish __init__ and register its
    # lifecycle interface before we send the first transition request.
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

    # Activate as soon as configure succeeds (→ inactive state)
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

    return LaunchDescription([debug_arg, node, on_process_start, on_inactive])
