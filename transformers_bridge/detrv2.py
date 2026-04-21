from scripts.venv_setup import auto_inject_venv
auto_inject_venv(packages=['torch', 'transformers'])
# Now safe to import ML packages

# ML imports
import torch
import cv2
import numpy as np
import threading
import time
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers_bridge.model_registry import resolve_model_config

# ROS imports
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
#Optimization
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy



class TransformersDetectorNode(LifecycleNode):
    def __init__(self):
        super().__init__("transformers_node")

        # Declare parameters
        self.declare_parameter("model_name", "PekingU/rtdetr_v2_r18vd")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("debug", False)  # toggle bounding-box annotation
        self.declare_parameter("device",     "auto")  # auto/cpu/cuda
        self.declare_parameter("image_size", 640)     # resize before inference
        self.declare_parameter("compressed", False)   # subscribe to compressed image topics

        # Model (loaded in on_configure)
        self._model = None
        self._processor = None
        self._bridge = CvBridge()
        self._image_size = None

        # Pub/Sub
        self._pub_detections = None
        self._pub_debug = None
        self._sub = None


        self._camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # avoids reliable/best_effor compatibility issues
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # only keep latest — Matches drop policy
        )

        # Inference thread state
        # _latest_frame: written atomically by _image_callback (CPython GIL),
        # pinned to a local variable before use in _infer_loop.
        self._latest_frame: Image | None = None
        self._new_frame_event = threading.Event()   # signals a new frame is ready
        self._running = False
        self._infer_thread: threading.Thread | None = None

        # FPS counting
        self._frame_count = 0
        self._fps_last_time = None

    # ── Lifecycle callbacks ─────────────────────────────────────────────────

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Load model weights. Called once before activation."""
        self.model_name = self.get_parameter("model_name").value
        self.threshold  = self.get_parameter("threshold").value
        self.debug      = self.get_parameter("debug").value
        device_param = self.get_parameter("device").value
        if device_param == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device_param)

        self._image_size = self.get_parameter("image_size").value
        self.compressed  = self.get_parameter("compressed").value

        # Resolve model config from registry
        # This allows us to use custom model names and automatically get the correct processor and model classes
        cfg, matched_key = resolve_model_config(self.model_name)

        if matched_key is None:
            self.get_logger().warn(
                f"Model '{self.model_name}' did not match any registry entry — "
                f"falling back to Auto classes. Results may vary."
            )
        elif not cfg.get("tested", False):
            self.get_logger().warn(
                f"Model type '{matched_key}' is in the registry but has not been "
                f"tested end-to-end. Proceed with caution."
            )

        if cfg.get("notes"):
            self.get_logger().info(f"Registry note for '{matched_key}': {cfg['notes']}")

        for param in cfg.get("extra_params", []):
            try:
                self.declare_parameter(param, "")
            except Exception:
                pass  # already declared

        processor_cls = cfg["processor_cls"]
        model_cls = cfg["model_cls"]

        self.get_logger().info(
            f"Loading model '{self.model_name}' on {self._device} "
            f"[{processor_cls.__name__} / {model_cls.__name__}] …"
        )

        try:
            self._processor = processor_cls.from_pretrained(self.model_name)
            self._model = model_cls.from_pretrained(self.model_name)
        except Exception as e:
            self.get_logger().error(f"Failed to load model '{self.model_name}': {e}")
            return TransitionCallbackReturn.FAILURE
        self._model.to(self._device).eval()

        self._callback_group = ReentrantCallbackGroup()

        self._warm_up()

        self.get_logger().info("Model loaded ✓")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Start inference thread, create publishers and subscription."""
        # Relative names — remappable via --ros-args --remap or launch remappings=[]
        self._pub_detections = self.create_lifecycle_publisher(
            Detection2DArray, "detections", 10)

        if self.debug:
            self._pub_debug = self.create_lifecycle_publisher(
                Image, "debug_image", 10)

        # Subscription just stores the frame and wakes the inference thread
        if self.compressed:
            self._sub = self.create_subscription(
                CompressedImage, "/camera/image_raw/compressed", self._image_callback, self._camera_qos,
                callback_group=self._callback_group)
        else:
            self._sub = self.create_subscription(
                Image, "/camera/image_raw", self._image_callback, self._camera_qos,
                callback_group=self._callback_group)

        # Inference thread: runs at maximum speed, blocked by Event when idle
        self._running = True
        self._infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True, name="infer_loop")
        self._infer_thread.start()

        self.get_logger().info(
            f"Active — inference at max speed  (debug={self.debug})")

        # Required: activates all lifecycle publishers managed by the base class
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Stop inference thread and clean up. Model stays in memory."""
        # Signal the thread to exit, then unblock it if it's waiting
        self._running = False
        self._new_frame_event.set()
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=2.0)
            self._infer_thread = None

        self.destroy_subscription(self._sub)
        self._sub = None
        self._latest_frame = None
        self._new_frame_event.clear()

        self.destroy_lifecycle_publisher(self._pub_detections)
        self._pub_detections = None

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
        self._image_size = None
        self.get_logger().info("Model unloaded, GPU memory freed")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Called on ros2 shutdown regardless of current state."""
        self._running = False
        self._new_frame_event.set()
        torch.cuda.empty_cache()
        return TransitionCallbackReturn.SUCCESS

    # ── Inference ───────────────────────────────────────────────────────────
    def _forward(self, inputs):
        """Run inference on the given inputs."""

        # Mixed precision (cuda optimization)
        if self._device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self._model(**inputs)
        return self._model(**inputs)



    def _warm_up(self, runs: int = 2) -> None:
        """Run a few dummy forward passes so CUDA JIT kernels are compiled
        before the first real frame arrives. Must mirror _infer_and_publish exactly."""
        self.get_logger().info("Running warm-up …")
        dummy = np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
        inputs = self._processor(images=dummy, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            for _ in range(runs):
                self._forward(inputs)
        self.get_logger().info("Warm-up complete ✓")
    
    def _image_callback(self, msg) -> None:
        """Store the latest frame and wake the inference thread. Must stay fast."""
        self._latest_frame = msg    # atomic in CPython (single STORE_ATTR)
        self._new_frame_event.set() # unblock _infer_loop

    def _infer_loop(self) -> None:
        """Inference thread: runs as fast as the model allows.
        Blocks on Event when no new frame is available (no busy-waiting).
        """
        while self._running:
            # Block until _image_callback signals a new frame (or deactivate wakes us)
            got_frame = self._new_frame_event.wait(timeout=0.5)
            self._new_frame_event.clear()

            if not got_frame or not self._running:
                continue

            # Pin to local — safe even if _image_callback fires mid-inference
            # and replaces self._latest_frame with a newer msg.
            frame = self._latest_frame
            if frame is None:
                continue

            self._infer_and_publish(frame)

    def _infer_and_publish(self, frame: Image) -> None:
        now = time.monotonic()
        if self._fps_last_time is not None:
            self._frame_count += 1
            if self._frame_count % 30 == 0:
                fps = 30 / (now - self._fps_last_time)
                self.get_logger().info(f"Inference FPS: {fps:.1f}")
                self._fps_last_time = now
                self._frame_count = 0
        else:
            self._fps_last_time = now

        if isinstance(frame, CompressedImage):
            cv_image = self._bridge.compressed_imgmsg_to_cv2(frame, "rgb8")
        else:
            cv_image = self._bridge.imgmsg_to_cv2(frame, "rgb8")

        inputs = self._processor(
            images=cv_image, return_tensors="pt", size={"height": self._image_size, "width": self._image_size}).to(self._device)

        with torch.inference_mode():
            outputs = self._forward(inputs)

        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor(
                [(cv_image.shape[0], cv_image.shape[1])], device=self._device),
            threshold=self.threshold,
        )[0]

        # Avoid publishing empty detections
        if len(results["scores"]) == 0:
            return

        self._pub_detections.publish(self._to_detection_msg(results, frame.header))

        self.get_logger().debug(f"scores: {results['scores']}, labels: {results['labels']}, boxes: {results['boxes']}")

        # Debug annotation — zero cost when debug=False
        if self.debug:
            annotated = self._draw(cv_image.copy(), results)
            debug_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="rgb8")
            debug_msg.header = frame.header
            self._pub_debug.publish(debug_msg)

    # ── Vision Msgs ───────────────────────────────────────────────────────────
    def _to_detection_msg(self, results, header) -> Detection2DArray:
        array_msg = Detection2DArray()
        array_msg.header = header

        for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]):
            det = Detection2D()
            det.header = header

            # Bounding box center + size (Detection2D uses center format, not x1y1x2y2)
            x1, y1, x2, y2 = box.tolist()
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            # Class + confidence
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self._model.config.id2label[label_id.item()]
            hyp.hypothesis.score = score.item()
            det.results.append(hyp)

            array_msg.detections.append(det)

        return array_msg

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
    # Need to make the node reentrant to handle the inference thread
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
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
                  f"{result['scores'][0].item():.2f}  {box}")