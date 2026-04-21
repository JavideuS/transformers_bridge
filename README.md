# transformers_bridge

A ROS 2 package that bridges **HuggingFace Transformers** models into the ROS 2 ecosystem as managed [`LifecycleNode`](https://design.ros2.org/articles/node_lifecycle.html)s.

Currently ships with an **RT-DETRv2** object detector that publishes [`vision_msgs/Detection2DArray`](https://github.com/ros-perception/vision_msgs) and optional annotated debug images.

> **Roadmap**: ONNX Runtime export + inference backend is planned for leaner deployment without the full PyTorch stack.

---

## Architecture

```
/camera/image_raw  ──►  [transformers_bridge]  ──►  /transformers/detections       (Detection2DArray)
                                                ──►  /transformers/debug_image      (Image, optional)
```

The detector runs as a **LifecycleNode** with three distinct phases:

| Phase        | What happens                                                              |
| ------------ | ------------------------------------------------------------------------- |
| `configure`  | Model weights downloaded/loaded, GPU memory allocated, warm-up passes run |
| `activate`   | Subscriber + publishers created, inference thread starts                  |
| `deactivate` | Inference thread joins, pubs/sub destroyed — model stays in VRAM          |
| `cleanup`    | GPU memory freed                                                          |

The **subscription** (`image_in`) stores only the latest frame (no queue build-up). A dedicated **inference thread** wakes on each new frame and processes at the GPU's maximum throughput, naturally dropping stale frames when the model is slower than the camera.

---

## Package layout

```
transformers_bridge/
├── config/
│   └── rtdetrv2.yaml          # Default parameters
├── launch/
│   ├── default.launch.py      # Start node, drive lifecycle manually
│   └── fast.launch.py         # Start node + auto configure → activate
├── transformers_bridge/
│   ├── detrv2.py              # RTDETRv2 LifecycleNode
│   └── test_image_publisher.py# Video → /camera/image_raw publisher (testing)
├── requirements.txt
└── package.xml
```

---

## Prerequisites

### ROS 2 dependencies

```bash
sudo apt install ros-$ROS_DISTRO-vision-msgs \
                 ros-$ROS_DISTRO-cv-bridge    \
                 ros-$ROS_DISTRO-sensor-msgs
```

### Python dependencies

#### Option A — Virtual environment (recommended, keeps system Python clean)

`torch` and `transformers` are large and bring many sub-dependencies. A venv isolates them from the system Python that ROS 2 tools rely on.

**The Magic:** The node automatically detects if you have a virtual environment anywhere inside the package directory containing `*venv*` in the name, and injects it into `sys.path` **at runtime**. You do not need to manually source the venv to launch the node.

```bash
# 1 — Create a venv inside the package folder
cd ~/cv_ws/src/transformers_bridge
python3 -m venv venv

# 2 — Install ML packages (activate first)
source venv/bin/activate
pip install -r requirements.txt
# CUDA: install torch before requirements.txt
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

That's it! When you run `ros2 run` or `ros2 launch`, the node will find `ubuntu_venv` and load the packages automatically.

#### Option B — System install (distrobox / Docker containers)

Safe inside an isolated container where "breaking system packages" only affects that container:

```bash
pip install -r requirements.txt --break-system-packages
```

No venv required before launching.

---

### Previous Approaches Tried (For Reference)

- **`setup.cfg` script rewriting**: We evaluated modifying `build_scripts` to rewrite the shebang to the venv interpreter, but it led to issues when testing across different setups.
- **Launch file PYTHONPATH hacks**: Injecting `SetEnvironmentVariable('PYTHONPATH', ...)` into the ROS2 launch context works, but it breaks `ros2 run` workflows and pollutes launch files.

The current **dynamic auto-injection** inside the node via `sys.path.insert()` at runtime is the cleanest approach and lets you use standard `ros2 launch` out of the box.

---

## Build

```bash
cd ~/cv_ws
colcon build --symlink-install --packages-select transformers_bridge
source install/setup.bash
```

---

## Usage

### Quick start (auto lifecycle)

```bash
ros2 launch transformers_bridge fast.launch.py
```

The node loads the model, runs warm-up passes, and starts publishing — no manual lifecycle steps needed.

### Manual lifecycle (development / debugging)

```bash
# Terminal 1 — start node (stays in 'unconfigured')
ros2 launch transformers_bridge default.launch.py

# Terminal 2 — drive lifecycle manually
ros2 lifecycle set /transformers/transformers_node configure   # loads model
ros2 lifecycle set /transformers/transformers_node activate    # starts inference
```

### Simulate camera input (testing)

```bash
# Publish a video file as /camera/image_raw
ros2 run transformers_bridge test_image_pub \
  --ros-args -p video_path:=/path/to/video.mp4 -p loop:=true

# Or use the bundled test video (H.264, hardware-decoded)
ros2 run transformers_bridge test_image_pub
```

### Override parameters at launch

```bash
ros2 launch transformers_bridge fast.launch.py \
  --ros-args -p threshold:=0.7 -p debug:=true -p device:=cpu
```

---

## Parameters

| Parameter    | Default                   | Description                                            |
| ------------ | ------------------------- | ------------------------------------------------------ |
| `model_name` | `PekingU/rtdetr_v2_r18vd` | HuggingFace model ID                                   |
| `threshold`  | `0.5`                     | Detection confidence threshold                         |
| `debug`      | `false`                   | Publish annotated image to `debug_image`               |
| `device`     | `auto`                    | `auto` \| `cpu` \| `cuda`                              |
| `image_size` | `640`                     | Resize input image before inference (RT-DETR specific) |

---

## Topics

Default resolved names assume `namespace: transformers`. Use **remapping** to connect to your camera:

```bash
ros2 launch transformers_bridge fast.launch.py \
  --ros-args --remap image_in:=/your_camera/image_raw
```

Or in a launch file:

```python
remappings=[('image_in', '/your_camera/image_raw')]
```

| Topic         | Direction | Type                           | Description                               |
| ------------- | --------- | ------------------------------ | ----------------------------------------- |
| `image_in`    | Sub       | `sensor_msgs/Image`            | Input camera stream                       |
| `detections`  | Pub       | `vision_msgs/Detection2DArray` | Detection results                         |
| `debug_image` | Pub       | `sensor_msgs/Image`            | Annotated image (only when `debug:=true`) |

---

## Roadmap

- [ ] **ONNX backend** — export models to ONNX for deployment without the full PyTorch stack
- [ ] **Custom message support** — add tracking ID, 3D box, depth integration
- [ ] **Multi-model support** — pluggable model registry (YOLO, DINO, SAM, …)
- [ ] **TensorRT** — optional TRT compilation for Xavier / Orin targets

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'rclpy'`

The venv was created without `--system-site-packages`. Recreate it:

```bash
python3 -m venv ~/.venvs/ros_ml --system-site-packages --clear
```

### `ModuleNotFoundError: No module named 'torch'`

The venv is not active when ROS 2 runs the node. Ensure you activate the venv **before** sourcing ROS 2 in your shell startup file.

### AV1 codec errors when using `test_image_pub`

OpenCV on most Linux systems lacks software AV1 decoding. Convert the video first:

```bash
ffmpeg -i input.mp4 -c:v libx264 -crf 18 -preset fast output_h264.mp4
```

### Node stays `unconfigured` after `fast.launch.py`

The 0.5 s startup delay may be too short on very slow machines. Temporarily increase it in `fast.launch.py`:

```python
TimerAction(period=2.0, actions=[configure_event])
```
