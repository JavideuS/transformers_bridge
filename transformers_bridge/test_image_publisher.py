"""
Test Video Publisher
====================
Publishes frames from a video file to a ROS2 topic, simulating a camera.
Respects the video's native FPS by default, or uses a custom rate.

Usage (positional args — easiest for testing):
    ros2 run transformers_bridge test_image_pub /path/to/video.mp4
    ros2 run transformers_bridge test_image_pub /path/to/video.mp4 /phone/image

Usage (ROS2 params — overrides positional args):
    ros2 run transformers_bridge test_image_pub \\
        --ros-args -p video_path:=/path/to/video.mp4 -p topic:=/phone/image

Optional params (ROS2 only):
    -p rate_hz:=15.0    # override FPS (default: video's native rate)
    -p loop:=false      # stop after one playthrough (default: true)
"""

import os
import sys

import cv2
import rclpy
import rclpy.utilities
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TestVideoPublisher(Node):
    def __init__(self, cli_video: str = "", cli_topic: str = ""):
        super().__init__("test_image_publisher")

        self.declare_parameter("video_path", cli_video)
        self.declare_parameter("topic",      cli_topic or "/camera/image_raw")
        self.declare_parameter("rate_hz",    0.0)    # 0 = use video's native FPS
        self.declare_parameter("loop",       True)

        video_path = self.get_parameter("video_path").value
        topic      = self.get_parameter("topic").value
        rate_hz    = self.get_parameter("rate_hz").value
        self._loop = self.get_parameter("loop").value

        if not video_path:
            self.get_logger().error(
                "No video specified. Pass a path as a positional argument:\n"
                "  ros2 run transformers_bridge test_image_pub /path/to/video.mp4 [topic]"
            )
            sys.exit(1)

        video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            self.get_logger().error(f"Video not found: {video_path}")
            sys.exit(1)

        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            self.get_logger().error(f"cv2 could not open: {video_path}")
            sys.exit(1)

        native_fps   = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps          = rate_hz if rate_hz > 0.0 else native_fps
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._bridge    = CvBridge()
        self._publisher = self.create_publisher(Image, topic, 10)
        self._timer     = self.create_timer(1.0 / fps, self._publish)

        self.get_logger().info(
            f"Publishing '{video_path}'\n"
            f"  → topic : {topic}\n"
            f"  → rate  : {fps:.1f} Hz  ({total_frames} frames, loop={self._loop})"
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

        msg = self._bridge.cv2_to_imgmsg(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), encoding="rgb8")
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self._publisher.publish(msg)

    def destroy_node(self):
        self._cap.release()
        super().destroy_node()


def main(args=None):
    # Strip ROS2 args to get positional CLI args: [video_path] [topic]
    non_ros = rclpy.utilities.remove_ros_args(args or sys.argv)[1:]
    cli_video = non_ros[0] if len(non_ros) > 0 else ""
    cli_topic = non_ros[1] if len(non_ros) > 1 else ""

    rclpy.init(args=args)
    node = TestVideoPublisher(cli_video=cli_video, cli_topic=cli_topic)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
