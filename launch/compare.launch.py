"""compare.launch.py — runs two detector nodes side by side for comparison.

Each node gets its own namespace and loads independently. Useful for comparing
backends (e.g. RT-DETRv2 vs YOLO-World), fine-tuned vs baseline, or any two configs.

Usage:
  ros2 launch transformers_bridge compare.launch.py \\
    params_a:=/path/to/config_a.yaml \\
    params_b:=/path/to/config_b.yaml

  ros2 launch transformers_bridge compare.launch.py \\
    params_a:=src/transformers_bridge/config/yolo_ego.yaml    namespace_a:=ego \\
    params_b:=src/transformers_bridge/config/yolo_baseline.yaml namespace_b:=baseline \\
    camera_topic:=/phone/image

Debug images are published on /<namespace>/debug_image.
Detections are published on /<namespace>/detections.
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


def _make_node(namespace, params_file, camera_topic, delay):
    """Return the node + its lifecycle event handlers."""
    node = LifecycleNode(
        package='transformers_bridge',
        namespace=namespace,
        executable='detector',
        name='detector_node',
        output='both',
        emulate_tty=True,
        parameters=[params_file, {'debug': True}],
        remappings=[('/camera/image_raw', camera_topic)],
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
            on_start=[TimerAction(period=delay, actions=[configure])],
        )),
        RegisterEventHandler(OnStateTransition(
            target_lifecycle_node=node,
            goal_state='inactive',
            entities=[activate],
        )),
    ]


def launch_setup(context, *args, **kwargs):
    params_a     = LaunchConfiguration('params_a').perform(context)
    params_b     = LaunchConfiguration('params_b').perform(context)
    namespace_a  = LaunchConfiguration('namespace_a').perform(context)
    namespace_b  = LaunchConfiguration('namespace_b').perform(context)
    camera_topic = LaunchConfiguration('camera_topic').perform(context)

    return [
        *_make_node(namespace_a, params_a, camera_topic, delay=0.5),
        *_make_node(namespace_b, params_b, camera_topic, delay=1.5),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('transformers_bridge')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_a',
            default_value=os.path.join(pkg_share, 'config', 'yolo_ego.yaml'),
            description='Parameters YAML for model A.',
        ),
        DeclareLaunchArgument(
            'params_b',
            default_value=os.path.join(pkg_share, 'config', 'yolo_baseline.yaml'),
            description='Parameters YAML for model B.',
        ),
        DeclareLaunchArgument(
            'namespace_a', default_value='model_a',
            description='ROS2 namespace for model A.',
        ),
        DeclareLaunchArgument(
            'namespace_b', default_value='model_b',
            description='ROS2 namespace for model B.',
        ),
        DeclareLaunchArgument(
            'camera_topic', default_value='/camera/image_raw',
            description='Input image topic both nodes subscribe to.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
