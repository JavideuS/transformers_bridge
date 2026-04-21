# transformers_bridge

A ROS 2 package that bridges **HuggingFace Transformers** object-detection models into the ROS 2 perception stack. Any model compatible with the `AutoModelForObjectDetection` / `AutoImageProcessor` API works out of the box; known architectures (RT-DETRv2, DETR, YOLOS, Grounding DINO) are handled by a model registry that picks the correct classes automatically.

---

## Architecture

```
/camera/image_raw  ──►  [transformers_bridge]  ──►  /transformers/detections   (Detection2DArray)
                                                ──►  /transformers/debug_image  (Image, optional)
```

The detector node is a **LifecycleNode**. Inference runs on a dedicated background thread so the ROS executor is never blocked.

| Lifecycle phase | What happens                                                                                                                      |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `configure`     | Model registry resolves the correct processor/model classes, weights are downloaded or loaded from cache, CUDA warm-up passes run |
| `activate`      | Subscriber and publishers are created, inference thread starts                                                                    |
| `deactivate`    | Inference thread joins, pubs/sub destroyed — model stays in VRAM                                                                  |
| `cleanup`       | GPU memory freed                                                                                                                  |

The subscriber stores only the **latest frame** (no queue). The inference thread wakes on each new frame and processes at GPU throughput, naturally dropping stale frames when the model is slower than the camera.

---

## Package layout

```
transformers_bridge/
├── config/
│   └── rtdetrv2.yaml               # Default parameters
├── launch/
│   ├── default.launch.py           # Start node; drive lifecycle manually
│   └── fast.launch.py              # Start node + auto configure → activate
├── scripts/
│   ├── benchmark.py                # Standalone latency/throughput benchmark
│   └── venv_setup.py               # Venv auto-inject helper
├── transformers_bridge/
│   ├── detrv2.py                   # Detector LifecycleNode
│   ├── model_registry.py           # Model factory + list_models entry point
│   └── test_image_publisher.py     # Video → /camera/image_raw publisher (testing)
├── requirements.txt
└── package.xml
```

---

## Prerequisites

### ROS 2 dependencies

All ROS 2 dependencies are declared in `package.xml`, so `rosdep` resolves them automatically:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### Python / ML dependencies

`torch` and `transformers` are large packages with many sub-dependencies. A virtual environment isolates them from the system Python that ROS 2 tools rely on.

**Auto-inject:** The node detects any directory named `*venv*` inside the package root and injects it into `sys.path` at runtime. You do not need to source the venv manually before launching.

```bash
# 1. Create a venv inside the package folder
cd ~/cv_ws/src/transformers_bridge
python3 -m venv venv --system-site-packages

# 2. Install ML packages
source venv/bin/activate
pip install -r requirements.txt

# CUDA (optional, replace cu124 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Alternatively, inside an isolated container (Docker / Distrobox), a system-wide install also works:

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

### Quick start (automatic lifecycle)

```bash
ros2 launch transformers_bridge fast.launch.py
```

The launch file configures and activates the node automatically. No manual lifecycle transitions are needed.

### Manual lifecycle (development / debugging)

```bash
# Terminal 1 — start node (stays in 'unconfigured')
ros2 launch transformers_bridge default.launch.py

# Terminal 2 — drive lifecycle manually
ros2 lifecycle set /transformers/transformers_node configure   # loads model
ros2 lifecycle set /transformers/transformers_node activate    # starts inference
```

### Remap the camera topic

The node subscribes to `/camera/image_raw` by default. Remap at launch time:

```bash
ros2 launch transformers_bridge fast.launch.py \
  --ros-args --remap /camera/image_raw:=/your_camera/image_raw
```

Or in a launch file:

```python
remappings=[('/camera/image_raw', '/your_camera/image_raw')]
```

For compressed topics set `compressed:=true` and remap `/camera/image_raw/compressed` instead.

### Simulate camera input (testing)

```bash
# Publish a video file as /camera/image_raw
ros2 run transformers_bridge test_image_pub \
  --ros-args -p video_path:=/path/to/video.mp4 -p loop:=true
```

### Override parameters at launch

```bash
ros2 launch transformers_bridge fast.launch.py \
  --ros-args -p model_name:=facebook/detr-resnet-50 \
             -p threshold:=0.7 \
             -p debug:=true \
             -p device:=cpu
```

---

## Parameters

| Parameter    | Default                   | Description                                                   |
| ------------ | ------------------------- | ------------------------------------------------------------- |
| `model_name` | `PekingU/rtdetr_v2_r18vd` | HuggingFace model ID or local path                            |
| `threshold`  | `0.5`                     | Minimum confidence score for published detections             |
| `debug`      | `false`                   | Publish annotated image on `debug_image`                      |
| `device`     | `auto`                    | `auto` \| `cpu` \| `cuda` \| `cuda:N`                         |
| `image_size` | `640`                     | Resize input to this square size before inference             |
| `compressed` | `false`                   | Subscribe to `sensor_msgs/CompressedImage` instead of `Image` |

Parameters are read once in `on_configure`. To change them, deactivate → cleanup → reconfigure the node.

---

## Topics

| Topic                          | Direction | Type                           | Description                                |
| ------------------------------ | --------- | ------------------------------ | ------------------------------------------ |
| `/camera/image_raw`            | Sub       | `sensor_msgs/Image`            | Input camera stream (remappable)           |
| `/camera/image_raw/compressed` | Sub       | `sensor_msgs/CompressedImage`  | Compressed input (when `compressed:=true`) |
| `detections`                   | Pub       | `vision_msgs/Detection2DArray` | Detection results                          |
| `debug_image`                  | Pub       | `sensor_msgs/Image`            | Annotated image (only when `debug:=true`)  |

Default resolved names assume the node is launched under `namespace: transformers` (as configured in the provided launch files).

---

## Model Registry

`transformers_bridge/model_registry.py` implements a factory that maps model name substrings to the correct HuggingFace processor and model classes. This avoids hard-coding `AutoImageProcessor` everywhere and allows architecture-specific classes (e.g. `RTDetrImageProcessor`) to be used where they give better results.

### How it works

`resolve_model_config(model_name)` iterates the registry keys. The first key that is a **substring** of the lowercased model name wins. If nothing matches, it falls back to `AutoImageProcessor` / `AutoModelForObjectDetection` and logs a warning.

### Registered architectures

| Key              | Processor              | Model                         | Tested | Notes                            |
| ---------------- | ---------------------- | ----------------------------- | ------ | -------------------------------- |
| `rtdetr`         | `RTDetrImageProcessor` | `RTDetrV2ForObjectDetection`  | Yes    | Use `image_size=640`             |
| `detr`           | `AutoImageProcessor`   | `AutoModelForObjectDetection` | No     | —                                |
| `yolos`          | `AutoImageProcessor`   | `AutoModelForObjectDetection` | No     | —                                |
| `grounding-dino` | `AutoProcessor`        | `AutoModelForObjectDetection` | No     | Requires `text_prompt` parameter |

### Listing available models

```bash
ros2 run transformers_bridge list_models
```

Prints the full registry table, including processor/model class names, tested status, extra parameters, and notes.

### Adding a new model

Add an entry to `REGISTRY` in `model_registry.py`:

```python
"owlvit": {
    "processor_cls": AutoProcessor,
    "model_cls": AutoModelForZeroShotObjectDetection,
    "notes": "Zero-shot; requires text queries",
    "extra_params": ["text_queries"],
    "tested": False,
},
```

The node picks it up automatically on the next `configure` transition — no other code changes needed.

---

## Benchmark

`scripts/benchmark.py` measures inference latency and throughput entirely outside ROS. It uses the same model registry and venv auto-inject as the node.

```bash
# Basic run — 100 iterations, image_size=640, auto device
python scripts/benchmark.py \
  --model PekingU/rtdetr_v2_r18vd \
  --image madison.jpg

# Directory of images, custom settings
python scripts/benchmark.py \
  --model facebook/detr-resnet-50 \
  --image /path/to/images/ \
  --runs 200 \
  --image-size 512 \
  --device cuda
```

### Arguments

| Argument       | Default    | Description                            |
| -------------- | ---------- | -------------------------------------- |
| `--model`      | (required) | HuggingFace model ID or local path     |
| `--image`      | (required) | Single image file or directory         |
| `--runs`       | `100`      | Number of timed inference passes       |
| `--image-size` | `640`      | Square resize applied before inference |
| `--device`     | `auto`     | `auto` \| `cpu` \| `cuda` \| `cuda:N`  |

### Output

- Per-run latency list (ms)
- Mean, std, min, max, FPS
- Peak GPU VRAM usage (`torch.cuda.max_memory_allocated`)
- Sanity check — asserts that detections are non-empty on the test image at `threshold=0.1`
- Markdown table ready to paste into this README

### Example output

```
Transformers Bridge Benchmark
========================================
  model  : PekingU/rtdetr_v2_r18vd
  images : 1 file(s)  [madison.jpg]
  runs   : 100

[OK]   Sanity check passed — 8 detection(s) on 'madison.jpg' at threshold=0.1

  Mean       :   18.34 ms
  Std        :    0.91 ms
  Min        :   17.10 ms
  Max        :   23.47 ms
  FPS        :    54.5
  Peak VRAM  :  312.4 MiB
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

The venv was not found or does not contain `torch`. Verify the venv directory is inside the package root and contains a valid `site-packages`:

```bash
ros2 run transformers_bridge venv_setup
```

This prints the `PYTHONPATH` export the node would inject. If it errors, follow the path it reports to diagnose the missing venv.

### `ModuleNotFoundError: No module named 'rclpy'`

The venv was created without `--system-site-packages`, so ROS 2 Python packages are invisible from inside it. Recreate it:

```bash
python3 -m venv venv --system-site-packages --clear
```

### Node stays `unconfigured` after `fast.launch.py`

The 0.5 s startup timer may be too short on slow machines. Increase it in `fast.launch.py`:

```python
TimerAction(period=2.0, actions=[configure_event])
```

### AV1 codec errors with `test_image_pub`

OpenCV on most Linux systems lacks software AV1 decoding. Convert the file first:

```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 18 -preset fast output_h264.mp4
```

---

## Possible extensions

Depth integration (`Detection3DArray`) and a lightweight tracker (SORT / ByteTrack) are natural next steps if the package grows beyond 2D detection.
