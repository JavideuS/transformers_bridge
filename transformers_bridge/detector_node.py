"""
DetectorNode — base lifecycle node for object detection in ROS2.

Designed to be subclassed. The inference pipeline is a template method with
three override points for downstream packages (e.g. egocentric_ros):

    _post_process(detections, cv_image, frame) -> list[dict]
        Called after backend inference, before drawing and publishing.
        Override to add 3D lifting, filtering, re-scoring, tracking IDs, etc.

    _extra_publish(detections, cv_image, frame)
        Called after the standard Detection2DArray is published.
        Override to publish Detection3DArray, context tokens, depth maps, etc.

    _draw(img, detections) -> np.ndarray
        Override for custom visualization (e.g. overlay depth, track IDs).

Subclass pattern (in egocentric_ros or any dependent package):

    from transformers_bridge.detector_node import DetectorNode

    class EgoDetectorNode(DetectorNode):
        def on_configure(self, state):
            result = super().on_configure(state)       # loads backend, warm-up
            if result != TransitionCallbackReturn.SUCCESS:
                return result
            # load your depth model, read extra params, etc.
            return TransitionCallbackReturn.SUCCESS

        def on_activate(self, state):
            result = super().on_activate(state)        # creates pubs/subs, starts thread
            self._camera_info_sub = self.create_subscription(CameraInfo, ...)
            self._pub_det3d = self.create_lifecycle_publisher(Detection3DArray, ...)
            return result

        def on_deactivate(self, state):
            self.destroy_subscription(self._camera_info_sub)
            self.destroy_lifecycle_publisher(self._pub_det3d)
            return super().on_deactivate(state)        # stops thread, destroys base pubs

        def _post_process(self, detections, cv_image, frame):
            # lift 2D → 3D using self._camera_info
            return detections

        def _extra_publish(self, detections, cv_image, frame):
            self._pub_det3d.publish(self._build_det3d(detections, frame))

Backend selection is done via the 'backend' ROS2 parameter ("transformers" or "yolo").
Switch at runtime: deactivate → cleanup → set new param → configure → activate.
"""

import json
import cv2
import numpy as np
import threading
import time
from pathlib import Path

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

_BACKEND_PACKAGES = {
    "transformers": ["torch", "transformers"],
    "yolo":         ["torch", "ultralytics"],
}


class DetectorNode(LifecycleNode):
    """Base detection lifecycle node. Subclass to extend the pipeline."""

    _TIER_COLOR = {
        "frequent": (0, 220, 0), #GREEN
        "common":   (255, 200, 0), #YELLOW
        "rare":     (255, 120, 0), #ORANGE
    }
    _TIER_BADGE   = {"frequent": "F", "common": "C", "rare": "R"}
    _FALLBACK_COLOR = (0, 200, 255)

    def __init__(self, node_name: str = "detector_node"):
        super().__init__(node_name)

        # ── Common parameters ────────────────────────────────────────────────
        self.declare_parameter("backend",    "transformers")   # "transformers" | "yolo"
        self.declare_parameter("threshold",  0.5)
        self.declare_parameter("debug",      False)
        self.declare_parameter("device",     "auto")           # auto | cpu | cuda
        self.declare_parameter("image_size", 640)
        self.declare_parameter("compressed", False)
        self.declare_parameter("cat_meta_path",     "")        # LVIS-style freq metadata
        self.declare_parameter("camera_info_topic", "")        # optional; enables self._camera_info

        # ── Transformers-specific ────────────────────────────────────────────
        self.declare_parameter("model_name", "PekingU/rtdetr_v2_r18vd")

        # ── YOLO-specific ────────────────────────────────────────────────────
        self.declare_parameter("model_path",       "yolov8s.pt")
        self.declare_parameter("iou_threshold",    0.45)
        self.declare_parameter("class_names_path", "")

        # ── State ────────────────────────────────────────────────────────────
        self._backend       = None
        self._camera_info   = None   # populated if camera_info_topic is set
        self._label_freq    = None
        self._bridge        = CvBridge()
        self._image_size    = None

        # ── Pub/sub placeholders ─────────────────────────────────────────────
        self._pub_detections = None
        self._pub_debug      = None
        self._sub            = None
        self._camera_info_sub = None

        self._camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Inference thread ─────────────────────────────────────────────────
        self._latest_frame  = None
        self._new_frame_event = threading.Event()
        self._running       = False
        self._infer_thread  = None
        self._frame_count   = 0
        self._fps_last_time = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Load backend model. Subclasses: call super() first, check result."""
        backend_name = self.get_parameter("backend").value
        if backend_name not in _BACKEND_PACKAGES:
            self.get_logger().error(
                f"Unknown backend '{backend_name}'. Valid: {list(_BACKEND_PACKAGES)}"
            )
            return TransitionCallbackReturn.FAILURE

        self.threshold   = self.get_parameter("threshold").value
        self.debug       = self.get_parameter("debug").value
        self._image_size = self.get_parameter("image_size").value
        self.compressed  = self.get_parameter("compressed").value

        device_param = self.get_parameter("device").value
        from scripts.venv_setup import auto_inject_venv
        auto_inject_venv(packages=_BACKEND_PACKAGES[backend_name])
        import torch
        self._device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_param == "auto" else torch.device(device_param)
        )

        cat_meta_path = self.get_parameter("cat_meta_path").value
        if cat_meta_path:
            try:
                self._label_freq = _load_label_freq(cat_meta_path)
                self.get_logger().info(
                    f"Adaptive draw: {len(self._label_freq)} categories from '{cat_meta_path}'"
                )
            except Exception as e:
                self.get_logger().warn(f"cat_meta_path load failed: {e} — simple draw")

        if backend_name == "transformers":
            from transformers_bridge.backends.transformers_backend import TransformersDetector
            self._backend = TransformersDetector()
        else:
            from transformers_bridge.backends.yolo_backend import YoloDetector
            self._backend = YoloDetector()

        params = {
            "device":           self._device,
            "threshold":        self.threshold,
            "model_name":       self.get_parameter("model_name").value,
            "model_path":       self.get_parameter("model_path").value,
            "iou_threshold":    self.get_parameter("iou_threshold").value,
            "class_names_path": self.get_parameter("class_names_path").value,
        }
        try:
            self._backend.load(params, self.get_logger())
        except Exception as e:
            self.get_logger().error(f"Backend load failed: {e}")
            return TransitionCallbackReturn.FAILURE

        self._callback_group = ReentrantCallbackGroup()
        self.get_logger().info("Running warm-up …")
        self._backend.warm_up(self._image_size)
        self.get_logger().info("Model loaded ✓")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Create publishers and start inference thread.

        Subclasses: call super() first, then add your publishers/subscribers.
        """
        self._pub_detections = self.create_lifecycle_publisher(
            Detection2DArray, "detections", 10)
        if self.debug:
            self._pub_debug = self.create_lifecycle_publisher(Image, "debug_image", 10)

        topic    = "/camera/image_raw/compressed" if self.compressed else "/camera/image_raw"
        msg_type = CompressedImage if self.compressed else Image
        self._sub = self.create_subscription(
            msg_type, topic, self._image_callback, self._camera_qos,
            callback_group=self._callback_group,
        )

        camera_info_topic = self.get_parameter("camera_info_topic").value
        if camera_info_topic:
            self._camera_info_sub = self.create_subscription(
                CameraInfo, camera_info_topic, self._camera_info_callback,
                self._camera_qos, callback_group=self._callback_group,
            )

        self._running = True
        self._infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True, name="infer_loop")
        self._infer_thread.start()

        self.get_logger().info(f"Active — inference at max speed  (debug={self.debug})")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Stop inference thread and destroy base publishers/subscribers.

        Subclasses: destroy YOUR publishers/subscribers BEFORE calling super().
        """
        self._running = False
        self._new_frame_event.set()
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=2.0)
            self._infer_thread = None

        if self._camera_info_sub is not None:
            self.destroy_subscription(self._camera_info_sub)
            self._camera_info_sub = None

        self.destroy_subscription(self._sub)
        self._sub = None
        self._latest_frame = None
        self._new_frame_event.clear()

        self.destroy_lifecycle_publisher(self._pub_detections)
        self._pub_detections = None
        if self._pub_debug is not None:
            self.destroy_lifecycle_publisher(self._pub_debug)
            self._pub_debug = None

        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Free backend GPU memory. Subclasses: call super() last."""
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        self._label_freq = None
        self._camera_info = None
        self._image_size = None
        self.get_logger().info("Backend unloaded, GPU memory freed")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._running = False
        self._new_frame_event.set()
        if self._backend is not None:
            self._backend.unload()
        return TransitionCallbackReturn.SUCCESS

    # ── Inference pipeline (template method) ─────────────────────────────────

    def _image_callback(self, msg) -> None:
        self._latest_frame = msg
        self._new_frame_event.set()

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _infer_loop(self) -> None:
        while self._running:
            got_frame = self._new_frame_event.wait(timeout=0.5)
            self._new_frame_event.clear()
            if not got_frame or not self._running:
                continue
            frame = self._latest_frame
            if frame is None:
                continue
            self._infer_and_publish(frame)

    def _infer_and_publish(self, frame) -> None:
        now = time.monotonic()
        if self._fps_last_time is not None:
            self._frame_count += 1
            if self._frame_count % 30 == 0:
                self.get_logger().info(
                    f"FPS: {30 / (now - self._fps_last_time):.1f}")
                self._fps_last_time = now
                self._frame_count = 0
        else:
            self._fps_last_time = now

        cv_image = self._to_rgb(frame)

        # 1. Backend inference → normalized list of dicts
        detections = self._backend.infer(cv_image, self._image_size, self.threshold)

        # 2. Post-processing hook — override for 3D lifting, tracking, filtering
        detections = self._post_process(detections, cv_image, frame)

        # 3. Debug visualization
        if self.debug and self._pub_debug is not None:
            debug_msg = self._bridge.cv2_to_imgmsg(
                self._draw(cv_image.copy(), detections), encoding="rgb8")
            debug_msg.header = frame.header
            self._pub_debug.publish(debug_msg)

        # 4. Standard Detection2DArray output
        if detections:
            self._pub_detections.publish(
                self._to_detection_msg(detections, frame.header))

        # 5. Extra publish hook — override for Detection3DArray, context, etc.
        self._extra_publish(detections, cv_image, frame)

        self.get_logger().debug(
            f"detections: {len(detections)}  "
            f"labels: {[d['label'] for d in detections]}"
        )

    # ── Extension hooks ───────────────────────────────────────────────────────

    def _post_process(
        self, detections: list[dict], cv_image: np.ndarray, frame
    ) -> list[dict]:
        """Override to add 3D lifting, tracking IDs, score filtering, etc.

        self._camera_info is available here if camera_info_topic was set.
        Returns the (modified) detections list.
        """
        return detections

    def _extra_publish(
        self, detections: list[dict], cv_image: np.ndarray, frame
    ) -> None:
        """Override to publish additional topics (Detection3DArray, context, depth)."""

    def _draw(self, img: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Override for custom visualization (e.g. tracking IDs, depth overlay)."""
        if self._label_freq is None:
            for d in detections:
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{d['label']} {d['score']:.2f}",
                            (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            return img

        _, w = img.shape[:2]
        text_scale = max(0.35, min(0.7, w / 1280))
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            freq  = self._label_freq.get(d["label"])
            color = self._TIER_COLOR.get(freq, self._FALLBACK_COLOR)
            badge = self._TIER_BADGE.get(freq, "")
            label = (f"{d['label']} [{badge}] {d['score']:.2f}"
                     if badge else f"{d['label']} {d['score']:.2f}")
            thickness = max(1, round(d["score"] * 3))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            (tw, th), bl = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
            ty = max(y1 - 4, th + bl)
            cv2.rectangle(img, (x1, ty - th - bl), (x1 + tw, ty + bl), color, cv2.FILLED)
            cv2.putText(img, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    # ── ROS message helpers ───────────────────────────────────────────────────

    def _to_detection_msg(self, detections: list[dict], header) -> Detection2DArray:
        msg = Detection2DArray()
        msg.header = header
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            det = Detection2D()
            det.header = header
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = d["label"]
            hyp.hypothesis.score    = d["score"]
            det.results.append(hyp)
            msg.detections.append(det)
        return msg

    def _to_rgb(self, msg) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            arr = self._bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
            return (cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    if arr.ndim == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))

        enc = msg.encoding.lower()
        arr = self._bridge.imgmsg_to_cv2(msg, "passthrough")
        if enc in ("rgb8", "rgb16"):
            return arr if arr.dtype == np.uint8 else (arr >> 8).astype(np.uint8)
        if enc in ("bgr8", "8uc3"):
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if enc in ("mono8", "8uc1"):
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if enc == "mono16":
            return cv2.cvtColor((arr >> 8).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if enc == "rgba8":
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        if enc == "bgra8":
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        self.get_logger().warn(
            f"Unknown encoding '{msg.encoding}' — treating as BGR",
            throttle_duration_sec=10.0)
        return (cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                if arr.ndim == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))


# ── Module-level helpers ──────────────────────────────────────────────────────

def _load_label_freq(path: str) -> dict:
    _NORM = {"r": "rare", "c": "common", "f": "frequent",
             "rare": "rare", "common": "common", "frequent": "frequent"}
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and "categories" in raw:
        entries = raw["categories"]
    elif isinstance(raw, dict):
        entries = list(raw.values())
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"Unrecognised cat_meta format in '{path}'")
    return {
        e["name"]: _NORM.get(e["frequency"], e["frequency"])
        for e in entries
        if "name" in e and "frequency" in e
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
