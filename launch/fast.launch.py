"""fast.launch.py — starts the node and auto-runs configure → activate.

Sequence:
  1. LifecycleNode starts  (unconfigured)
  2. OnProcessStart fires  → TimerAction(0.5 s) → configure
  3. Model loads (warm-up runs inside on_configure)
  4. OnStateTransition(inactive) fires → activate
  5. Node is live

Usage:
  ros2 launch transformers_bridge fast.launch.py
  ros2 launch transformers_bridge fast.launch.py params_file:=/path/to/yolo_ego.yaml
  ros2 launch transformers_bridge fast.launch.py debug:=true
  ros2 launch transformers_bridge fast.launch.py params_file:=/path/to/yolo_ego.yaml debug:=true
"""

import os
import launch
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, RegisterEventHandler, EmitEvent, TimerAction, OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    params = [LaunchConfiguration('params_file').perform(context)]
    debug  = LaunchConfiguration('debug').perform(context).lower()
    if debug in ('true', 'false'):
        params.append({'debug': debug == 'true'})

    node = LifecycleNode(
        package='transformers_bridge',
        namespace='transformers',
        executable='detector',
        name='transformers_node',
        output='both',
        emulate_tty=True,
        parameters=params,
        remappings=[('/camera/image_raw', '/phone/image')],
    )

    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=launch.events.matchers.matches_action(node),
        transition_id=Transition.TRANSITION_CONFIGURE,
    ))
    activate = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=launch.events.matchers.matches_action(node),
        transition_id=Transition.TRANSITION_ACTIVATE,
    ))

    return [
        node,
        RegisterEventHandler(OnProcessStart(
            target_action=node,
            on_start=[TimerAction(period=0.5, actions=[configure])],
        )),
        RegisterEventHandler(OnStateTransition(
            target_lifecycle_node=node,
            goal_state='inactive',
            entities=[activate],
        )),
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
