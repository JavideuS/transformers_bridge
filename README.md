# transformers_bridge

A ROS 2 package for object detection using **HuggingFace Transformers** and **Ultralytics YOLO** models. A single `DetectorNode` supports both backends, outputs standard `vision_msgs/Detection2DArray`, and is designed to be subclassed by downstream packages (e.g. an egocentric perception stack).

---

## Architecture

```
/camera/image_raw  ──►  [DetectorNode]  ──►  /transformers/detections   (Detection2DArray)
                                         ──►  /transformers/debug_image  (Image, optional)
```

The node is a **LifecycleNode**. Inference runs on a dedicated background thread so the ROS executor is never blocked. The backend (Transformers or YOLO) is selected by a parameter and loaded at `configure` time.

| Lifecycle phase | What happens |
| --------------- | ------------ |
| `configure` | Backend selected, model loaded, CUDA warm-up runs |
| `activate` | Publishers, subscriber, and inference thread start |
| `deactivate` | Inference thread joins, pubs/sub destroyed — model stays in VRAM |
| `cleanup` | GPU memory freed — backend can be swapped before next `configure` |

The subscriber stores only the **latest frame** (no queue). The inference thread processes at GPU throughput and naturally drops stale frames when the model is slower than the camera.

---

## Package layout

```
transformers_bridge/
├── config/
│   ├── default.yaml              # RT-DETRv2 defaults
│   ├── yolo.yaml                 # YOLO-World defaults
│   ├── yolo_ego.yaml             # YOLO-World + ego-centric category metadata
│   └── yolo_baseline.yaml        # YOLO baseline (COCO classes)
├── launch/
│   ├── default.launch.py         # Start node; drive lifecycle manually
│   ├── fast.launch.py            # Start node + auto configure → activate
│   └── compare.launch.py         # Run two models side by side (any backend)
├── scripts/
│   ├── benchmark.py              # Standalone latency/throughput benchmark
│   └── venv_setup.py             # Venv auto-inject helper
├── transformers_bridge/
│   ├── backends/
│   │   ├── base.py               # BaseDetector ABC — implement to add a backend
│   │   ├── transformers_backend.py
│   │   └── yolo_backend.py
│   ├── detector_node.py          # DetectorNode — base LifecycleNode (subclassable)
│   ├── model_registry.py         # Transformers model factory + list_models entry point
│   └── test_image_publisher.py   # Video → topic publisher for testing
├── requirements.txt
└── package.xml
```

---

## Prerequisites

### ROS 2 dependencies

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### Python / ML dependencies

**Auto-inject:** The node detects any directory named `*venv*` inside the package root and injects it into `sys.path` at runtime — no need to source the venv before launching.

```bash
cd ~/cv_ws/src/transformers_bridge
python3 -m venv venv --system-site-packages
source venv/bin/activate

pip install -r requirements.txt

# CUDA (replace cu124 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

For YOLO support also install Ultralytics inside the same venv:
```bash
pip install ultralytics
```

In a container or Distrobox, a system-wide install also works:
```bash
pip install -r requirements.txt --break-system-packages
```

---

## Build

```bash
cd ~/cv_ws
colcon build --symlink-install --packages-select transformers_bridge
source install/setup.bash
```

---

## Usage

### Quick start

```bash
ros2 launch transformers_bridge fast.launch.py
```

The launch file configures and activates the node automatically.

### Launch arguments

All launch files accept these arguments:

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `params_file` | `config/default.yaml` | Path to a ROS2 parameters YAML file |
| `debug` | `""` (use YAML value) | Override debug flag: `true` or `false` |

```bash
# YOLO backend with ego-centric config
ros2 launch transformers_bridge fast.launch.py \
  params_file:=src/transformers_bridge/config/yolo_ego.yaml

# Override debug without editing the YAML
ros2 launch transformers_bridge fast.launch.py debug:=true

# Both together
ros2 launch transformers_bridge fast.launch.py \
  params_file:=src/transformers_bridge/config/yolo_ego.yaml \
  debug:=true
```

### Side-by-side comparison

```bash
# Compare fine-tuned YOLO vs RT-DETRv2 on the same feed
ros2 launch transformers_bridge compare.launch.py \
  params_a:=src/transformers_bridge/config/yolo_ego.yaml    namespace_a:=yolo \
  params_b:=src/transformers_bridge/config/default.yaml     namespace_b:=rtdetr \
  camera_topic:=/phone/image
```

Both nodes publish on `/<namespace>/detections` and `/<namespace>/debug_image`. Debug is always enabled in compare mode.

### Manual lifecycle (development / debugging)

```bash
# Terminal 1 — start node (stays in 'unconfigured')
ros2 launch transformers_bridge default.launch.py

# Terminal 2
ros2 lifecycle set /transformers/transformers_node configure
ros2 lifecycle set /transformers/transformers_node activate
```

### Switching backends at runtime

No restart needed — use the lifecycle to swap model or backend:

```bash
ros2 lifecycle set /transformers/transformers_node deactivate
ros2 lifecycle set /transformers/transformers_node cleanup
ros2 param set /transformers/transformers_node backend yolo
ros2 param set /transformers/transformers_node model_path /path/to/model.pt
ros2 lifecycle set /transformers/transformers_node configure
ros2 lifecycle set /transformers/transformers_node activate
```

### Test with a video file

```bash
# Terminal 1 — publish video frames
ros2 run transformers_bridge test_image_pub /path/to/video.mp4 /phone/image

# Terminal 2 — run detector on that topic
ros2 launch transformers_bridge fast.launch.py \
  params_file:=src/transformers_bridge/config/yolo_ego.yaml \
  debug:=true
# (remapping to /phone/image is set in the launch file)
```

The publisher loops by default. Optional ROS2 params: `-p rate_hz:=15.0`, `-p loop:=false`.

---

## Parameters

### Common

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `backend` | `transformers` | Inference backend: `transformers` or `yolo` |
| `threshold` | `0.5` | Minimum confidence score for published detections |
| `debug` | `false` | Publish annotated image on `debug_image` |
| `device` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N` |
| `image_size` | `640` | Resize input to this square size before inference |
| `compressed` | `false` | Subscribe to `CompressedImage` instead of `Image` |
| `cat_meta_path` | `""` | Path to a LVIS-style category metadata JSON. Enables tier-colored visualization (frequent/common/rare). Supports LVIS, EgoObjects, and flat-list formats. |
| `camera_info_topic` | `""` | If set, subscribe to `CameraInfo` on this topic and store it as `self._camera_info` — used by subclasses for 3D lifting. |

### Transformers backend (`backend: transformers`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `model_name` | `PekingU/rtdetr_v2_r18vd` | HuggingFace model ID or local checkpoint path |

### YOLO backend (`backend: yolo`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `model_path` | `yolov8s.pt` | Path to a `.pt` weights file (standard or YOLO-World) |
| `iou_threshold` | `0.45` | NMS IoU threshold |
| `class_names_path` | `""` | Path to `id2label.json` or `dataset.yaml` for open-vocabulary class set |

Parameters are read once in `on_configure`. To change them, deactivate → cleanup → reconfigure.

---

## Topics

| Topic | Direction | Type | Description |
| ----- | --------- | ---- | ----------- |
| `/camera/image_raw` | Sub | `sensor_msgs/Image` | Input stream (remappable) |
| `/camera/image_raw/compressed` | Sub | `sensor_msgs/CompressedImage` | Compressed input (`compressed:=true`) |
| `detections` | Pub | `vision_msgs/Detection2DArray` | Detection results |
| `debug_image` | Pub | `sensor_msgs/Image` | Annotated image (`debug:=true`) |

Resolved names assume `namespace: transformers` as set in the provided launch files.

---

## Extending DetectorNode

`DetectorNode` is designed to be subclassed by downstream packages. The inference pipeline is a template method with three override points:

```python
from transformers_bridge.detector_node import DetectorNode
from rclpy.lifecycle import TransitionCallbackReturn, State

class MyDetectorNode(DetectorNode):
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        result = super().on_configure(state)   # loads backend + warm-up
        if result != TransitionCallbackReturn.SUCCESS:
            return result
        # load additional models, declare extra params, etc.
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        result = super().on_activate(state)    # starts thread + base pubs/subs
        # add your publishers/subscribers here
        return result

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        # destroy YOUR pubs/subs BEFORE calling super()
        return super().on_deactivate(state)

    def _post_process(self, detections, cv_image, frame):
        """Called after backend inference, before drawing and publishing.
        self._camera_info is available here if camera_info_topic is set.
        Return the (modified) detections list."""
        return detections

    def _extra_publish(self, detections, cv_image, frame):
        """Called after Detection2DArray is published.
        Override to publish Detection3DArray, context, depth maps, etc."""

    def _draw(self, img, detections):
        """Override for custom visualization (tracking IDs, depth overlay, etc.)."""
        return super()._draw(img, detections)
```

The normalized detection format passed through the pipeline:
```python
[{"score": float, "label": str, "box": [x1, y1, x2, y2]}, ...]
```

### Adding a new backend

Implement `BaseDetector` from `transformers_bridge.backends.base`:

```python
from transformers_bridge.backends.base import BaseDetector

class MyBackend(BaseDetector):
    def load(self, params: dict, logger) -> None: ...
    def warm_up(self, image_size: int) -> None: ...
    def infer(self, image, image_size, threshold) -> list[dict]: ...
    def unload(self) -> None: ...
```

Then register it in `_BACKEND_PACKAGES` and the `if/elif` block in `detector_node.py`.

---

## Model Registry (Transformers backend)

`model_registry.py` maps model name substrings to the correct HuggingFace processor and model classes. `resolve_model_config(model_name)` returns the first registry entry whose key is a substring of the lowercased model name, falling back to `AutoImageProcessor` / `AutoModelForObjectDetection`.

### Registered architectures

| Key | Processor | Model | Tested | Notes |
| --- | --------- | ----- | ------ | ----- |
| `rtdetr` | `RTDetrImageProcessor` | `RTDetrV2ForObjectDetection` | Yes | `image_size=640` |
| `detr` | `AutoImageProcessor` | `AutoModelForObjectDetection` | No | — |
| `yolos` | `AutoImageProcessor` | `AutoModelForObjectDetection` | No | — |
| `grounding-dino` | `AutoProcessor` | `AutoModelForObjectDetection` | No | Requires `text_prompt` param |

```bash
ros2 run transformers_bridge list_models
```

---

## Benchmark

`scripts/benchmark.py` measures inference latency outside ROS using the same model registry and venv auto-inject as the node.

```bash
python scripts/benchmark.py \
  --model PekingU/rtdetr_v2_r18vd \
  --image /path/to/image.jpg
```

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--model` | required | HuggingFace model ID or local path |
| `--image` | required | Image file or directory |
| `--runs` | `100` | Number of timed passes |
| `--image-size` | `640` | Square resize before inference |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
The venv was not found. Run `ros2 run transformers_bridge venv_setup` to check which path the node would inject.

**`ModuleNotFoundError: No module named 'rclpy'` inside venv**
Venv was created without `--system-site-packages`. Recreate: `python3 -m venv venv --system-site-packages --clear`

**Node stays `unconfigured` after `fast.launch.py`**
The 0.5 s startup timer may be too short. Increase `period=2.0` in `fast.launch.py`.

**AV1 codec errors with `test_image_pub`**
OpenCV on most Linux systems lacks software AV1 decoding. Convert first:
```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 18 -preset fast output.mp4
```

**Encoding warnings on image topic**
The node handles `rgb8`, `bgr8`, `8UC3`, `mono8`, `mono16`, `rgba8`, `bgra8`. Unknown encodings are treated as BGR with a throttled warning.
