# ML imports
import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# ROS imports
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TransformersDetectorNode(LifecycleNode):
    def __init__(self):
        super().__init__("transformers_node")
        # Declare parameters
        self.declare_parameter("model_name", "PekingU/rtdetr_v2_r18vd")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("debug", False)  # toggle bounding-box annotation

        # Model
        self._model = None
        self._processor = None
        self._bridge = CvBridge()

        # Pub/Sub
        self._pub_img = None
        self._pub_debug = None
        self._sub = None

    # ── Lifecycle callbacks ─────────────────────────────────────────────────

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Load model weights. Called once before activation."""
        self.model_name = self.get_parameter("model_name").value
        self.threshold  = self.get_parameter("threshold").value
        self.debug      = self.get_parameter("debug").value

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(
            f"Loading model '{self.model_name}' on {self._device} …")

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model     = AutoModelForObjectDetection.from_pretrained(self.model_name)
        self._model.to(self._device).eval()

        self.get_logger().info("Model loaded ✓")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Create publishers and subscription. Node starts processing images."""
        self._pub_img = self.create_lifecycle_publisher(
            Image, "/detections/image", 10)

        if self.debug:
            self._pub_debug = self.create_lifecycle_publisher(
                Image, "/detections/debug_image", 10)

        self._sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_callback, 10)

        self.get_logger().info(
            f"Active — listening on /camera/image_raw  (debug={self.debug})")
        # Required: activates all lifecycle publishers managed by the base class
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Stop processing. Model stays in memory (on_cleanup frees it)."""
        self.destroy_subscription(self._sub)
        self._sub = None

        self.destroy_lifecycle_publisher(self._pub_img)
        self._pub_img = None

        if self._pub_debug is not None:
            self.destroy_lifecycle_publisher(self._pub_debug)
            self._pub_debug = None

        # Required: deactivates lifecycle publishers managed by the base class
        return super().on_deactivate(state)


    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Free GPU memory."""
        del self._model, self._processor
        torch.cuda.empty_cache()
        self._model = self._processor = None
        self.get_logger().info("Model unloaded, GPU memory freed")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Called on ros2 shutdown regardless of current state."""
        torch.cuda.empty_cache()
        return TransitionCallbackReturn.SUCCESS

    # ── Inference ───────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image) -> None:
        cv_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")

        inputs = self._processor(
            images=cv_image, return_tensors="pt").to(self._device)

        with torch.inference_mode():
            # Mixed precision (cuda optimization)
            if self._device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self._model(**inputs)
            else:
                outputs = self._model(**inputs)

        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor(
                [(cv_image.shape[0], cv_image.shape[1])], device=self._device),
            threshold=self.threshold,
        )[0]

        for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]):
            self.get_logger().info(
                f"  {self._model.config.id2label[label_id.item()]}: "
                f"{score.item():.2f}  box={[round(v,1) for v in box.tolist()]}")

        # Debug annotation — zero cost when debug=False
        if self.debug:
            annotated = self._draw(cv_image.copy(), results)
            debug_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="rgb8")
            debug_msg.header = msg.header
            self._pub_debug.publish(debug_msg)

        # Re-publish original frame (replace with annotated or custom msg later)
        self._pub_img.publish(msg)

    def _draw(self, img: np.ndarray, result) -> np.ndarray:
        """Draw bounding boxes onto img (RGB). Returns the annotated image."""
        for score, label_id, box in zip(
                result["scores"], result["labels"], result["boxes"]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            label = f"{self._model.config.id2label[label_id.item()]} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
        return img


# ── Entry point ─────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TransformersDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# ── Stand-alone test (no ROS) ────────────────────────────────────────────────

if __name__ == "__main__":
    from PIL import Image as PILImage

    image = PILImage.open("madison.jpg")

    image_processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
    model.eval()

    inputs = image_processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=0.5,
    )

    for result in results:
        for score, label_id, box in zip(
                result["scores"], result["labels"], result["boxes"]):
            box = [round(v, 2) for v in box.tolist()]
            print(f"{model.config.id2label[label_id.item()]}: "
                  f"{score.item():.2f}  {box}")