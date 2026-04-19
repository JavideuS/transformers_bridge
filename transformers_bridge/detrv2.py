# ML imports
import torch
import numpy as np
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge


class TransformersDetectorNode(Node):
    def __init__(self):
        super().__init__("transformers_node")

        # Parameters - Swapping models
        self.declare_parameter("model_name", "PekingU/rtdetr_v2_r18vd")
        self.declare_parameter("threshold", 0.5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().debug(f"Using device: {self.device}")

        self.model_name = self.get_parameter("model_name").value
        self.threshold = self.get_parameter("threshold").value
        self.get_logger().info(f"Loading model: {self.model_name}")


        # Model load at startup
        self.image_processor = RTDetrImageProcessor.from_pretrained(self.model_name)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval() # To be deterministic

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.publisher = self.create_publisher(Image, "/detections", 10)
        self.get_logger().info("TransformersDetectorNode ready, listening on /camera/image_raw")

    def image_callback(self, msg):
        # ROS Image → numpy array (no need for PIL)
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

        # Inference
        inputs = self.image_processor(images=cv_image, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            # Mixed precision (cuda optimization)
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(cv_image.shape[0], cv_image.shape[1])], device=self.device),
            threshold=self.threshold
        )

        # Log detections
        result = results[0]
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            label = self.model.config.id2label[label_id.item()]
            self.get_logger().info(f"  {label}: {score.item():.2f} {[round(v,1) for v in box.tolist()]}")

        # Re-publish the received image (you can replace with an annotated image later)
        self.publisher.publish(msg)

    def to_detection_msg(self, result, header: Header) -> Image:
        """Placeholder: convert detection results to an Image msg (e.g. annotated frame)."""
        return Image()


def main(args=None):
    rclpy.init(args=args)
    node = TransformersDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    # Stand-alone test (no ROS)
    from PIL import Image as PILImage
    image = PILImage.open("madison.jpg")

    image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")

    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score, label = score.item(), label_id.item()
            box = [round(i, 2) for i in box.tolist()]
            print(f"{model.config.id2label[label]}: {score:.2f} {box}")