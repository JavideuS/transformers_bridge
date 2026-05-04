import numpy as np
from transformers_bridge.backends.base import BaseDetector


class YoloDetector(BaseDetector):
    def __init__(self):
        self._model = None
        self._iou_threshold = 0.45
        self._threshold = 0.5
        self._class_names = None   # set when class_names_path is used

    def load(self, params: dict, logger) -> None:
        from pathlib import Path
        from ultralytics import YOLO

        weights_dir = Path.home() / ".cache" / "ultralytics"
        weights_dir.mkdir(parents=True, exist_ok=True)

        model_path = params["model_path"]
        # Resolve bare filenames to the cache dir so downloads land there
        # instead of the CWD. Explicit paths (absolute or relative with //) are left as-is.
        p = Path(model_path)
        if p.name == str(p):   # no directory component
            model_path = str(weights_dir / p)
            logger.info(f"Resolved '{p}' → {model_path}")
        device = params["device"]
        self._iou_threshold = params.get("iou_threshold", 0.45)
        self._threshold = params.get("threshold", 0.5)

        logger.info(f"Loading YOLO '{model_path}' on {device} …")
        self._model = YOLO(model_path)
        self._model.to(device)

        class_names_path = params.get("class_names_path", "")
        if class_names_path:
            self._class_names = _load_class_names(class_names_path)
            self._model.set_classes(self._class_names)
            logger.info(f"Open-vocabulary: {len(self._class_names)} classes from '{class_names_path}'")

    def warm_up(self, image_size: int) -> None:
        dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for _ in range(2):
            self._model(
                dummy, imgsz=image_size,
                conf=self._threshold, iou=self._iou_threshold, verbose=False,
            )

    def infer(self, image: np.ndarray, image_size: int, threshold: float) -> list[dict]:
        results = self._model(
            image, imgsz=image_size,
            conf=threshold, iou=self._iou_threshold, verbose=False,
        )
        result = results[0]
        if result.boxes is None:
            return []
        # result.names may not reflect set_classes() in all Ultralytics versions;
        # use the stored list when available.
        name_fn = (lambda i: self._class_names[i]) if self._class_names is not None \
                  else (lambda i: result.names[i])
        return [
            {
                "score": float(box.conf[0]),
                "label": name_fn(int(box.cls[0])),
                "box": box.xyxy[0].tolist(),
            }
            for box in result.boxes
        ]

    def unload(self) -> None:
        import torch
        del self._model
        torch.cuda.empty_cache()
        self._model = None
        self._class_names = None


def _load_class_names(path: str) -> list:
    """Load ordered class name list from id2label.json or dataset.yaml."""
    import json
    import yaml
    from pathlib import Path
    p = Path(path)
    if p.suffix == ".json":
        raw = json.loads(p.read_text())
        return [raw[str(i)] for i in range(len(raw))]
    if p.suffix in (".yaml", ".yml"):
        with open(p) as f:
            data = yaml.safe_load(f)
        names = data["names"]
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else names
    raise ValueError(f"Unsupported class names format: {p.suffix}")
