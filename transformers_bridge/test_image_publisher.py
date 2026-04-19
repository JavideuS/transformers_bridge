"""
Test Image Publisher
====================
Publishes a local image file to /camera/image_raw at a fixed rate.
Use this to test TransformersDetectorNode without a real camera.

Usage:
    ros2 run transformers_bridge test_image_pub --ros-args -p image_path:=/path/to/image.jpg
    ros2 run transformers_bridge test_image_pub  # defaults to madison.jpg in the package
"""

import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TestImagePublisher(Node):
    def __init__(self):
        super().__init__("test_image_publisher")

        self.declare_parameter("image_path", "")
        self.declare_parameter("rate_hz", 1.0)
        self.declare_parameter("topic", "/camera/image_raw")

        image_path = self.get_parameter("image_path").value
        rate_hz    = self.get_parameter("rate_hz").value
        topic      = self.get_parameter("topic").value

        # Default: use madison.jpg that lives next to setup.py
        if not image_path:
            pkg_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            image_path = os.path.join(pkg_dir, "madison.jpg")

        if not os.path.isfile(image_path):
            self.get_logger().error(f"Image not found: {image_path}")
            sys.exit(1)

        # Load once and keep in memory
        bgr = cv2.imread(image_path)
        if bgr is None:
            self.get_logger().error(f"cv2 could not open: {image_path}")
            sys.exit(1)
        self._rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self._bridge    = CvBridge()
        self._publisher = self.create_publisher(Image, topic, 10)
        self._timer     = self.create_timer(1.0 / rate_hz, self._publish)

        self.get_logger().info(
            f"Publishing '{image_path}' → {topic}  @ {rate_hz} Hz"
        )

    def _publish(self):
        msg = self._bridge.cv2_to_imgmsg(self._rgb, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self._publisher.publish(msg)
        self.get_logger().info("Published image frame")


def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
