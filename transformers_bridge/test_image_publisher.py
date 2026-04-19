"""
Test Video Publisher
====================
Publishes frames from a video file to /camera/image_raw, simulating a camera.
Respects the video's native FPS by default, or uses a custom rate.

Usage:
    ros2 run transformers_bridge test_image_pub
        # defaults to SukunaJogo.mp4 next to setup.py

    ros2 run transformers_bridge test_image_pub \\
        --ros-args -p video_path:=/path/to/video.mp4

    ros2 run transformers_bridge test_image_pub \\
        --ros-args -p video_path:=/path/to/video.mp4 -p rate_hz:=10.0 -p loop:=false
"""

import os
import sys

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TestVideoPublisher(Node):
    def __init__(self):
        super().__init__("test_image_publisher")

        self.declare_parameter("video_path", "")
        self.declare_parameter("rate_hz", 0.0)   # 0 = use video's native FPS
        self.declare_parameter("loop", True)       # restart video when it ends
        self.declare_parameter("topic", "/camera/image_raw")

        video_path = self.get_parameter("video_path").value
        rate_hz    = self.get_parameter("rate_hz").value
        self._loop = self.get_parameter("loop").value
        topic      = self.get_parameter("topic").value

        # Default: SukunaJogo_h264.mp4 in the package source directory
        if not video_path:
            # Resolve relative to this script's location (works from source and install)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            video_path = os.path.join(script_dir, "..", "SukunaJogo_h264.mp4")
            video_path = os.path.normpath(video_path)

        if not os.path.isfile(video_path):
            self.get_logger().error(f"Video not found: {video_path}")
            sys.exit(1)

        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            self.get_logger().error(f"cv2 could not open: {video_path}")
            sys.exit(1)

        # Use video's native FPS unless overridden
        native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = rate_hz if rate_hz > 0.0 else native_fps

        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._bridge    = CvBridge()
        self._publisher = self.create_publisher(Image, topic, 10)
        self._timer     = self.create_timer(1.0 / fps, self._publish)

        self.get_logger().info(
            f"Publishing '{video_path}' → {topic}  "
            f"@ {fps:.1f} Hz  ({total_frames} frames, loop={self._loop})"
        )

    def _publish(self):
        ret, frame = self._cap.read()

        if not ret:
            if self._loop:
                self.get_logger().info("Video ended — restarting")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    self.get_logger().error("Could not restart video")
                    return
            else:
                self.get_logger().info("Video ended — shutting down")
                rclpy.shutdown()
                return

        # OpenCV reads BGR → convert to RGB for ROS
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        msg = self._bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self._publisher.publish(msg)

    def destroy_node(self):
        self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TestVideoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
