import torch
import numpy as np
from transformers_bridge.backends.base import BaseDetector
from transformers_bridge.model_registry import resolve_model_config


class TransformersDetector(BaseDetector):
    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    def load(self, params: dict, logger) -> None:
        model_name = params["model_name"]
        self._device = params["device"]

        cfg, matched_key = resolve_model_config(model_name)
        if matched_key is None:
            logger.warn(f"'{model_name}' not in registry — using Auto classes")
        elif not cfg.get("tested"):
            logger.warn(f"'{matched_key}' is untested end-to-end")
        if cfg.get("notes"):
            logger.info(f"Registry note for '{matched_key}': {cfg['notes']}")

        processor_cls = cfg["processor_cls"]
        model_cls = cfg["model_cls"]
        logger.info(
            f"Loading '{model_name}' on {self._device} "
            f"[{processor_cls.__name__} / {model_cls.__name__}] …"
        )
        self._processor = processor_cls.from_pretrained(model_name)
        self._model = model_cls.from_pretrained(model_name).to(self._device).eval()

    def warm_up(self, image_size: int) -> None:
        dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        inputs = self._processor(images=dummy, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            for _ in range(2):
                self._forward(inputs)

    def _forward(self, inputs):
        if self._device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self._model(**inputs)
        return self._model(**inputs)

    def infer(self, image: np.ndarray, image_size: int, threshold: float) -> list[dict]:
        h, w = image.shape[:2]
        inputs = self._processor(
            images=image, return_tensors="pt",
            size={"height": image_size, "width": image_size},
        ).to(self._device)
        with torch.inference_mode():
            outputs = self._forward(inputs)
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(h, w)], device=self._device),
            threshold=threshold,
        )[0]
        return [
            {
                "score": float(score),
                "label": self._model.config.id2label[label_id.item()],
                "box": box.tolist(),
            }
            for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]
            )
        ]

    def unload(self) -> None:
        del self._model, self._processor
        torch.cuda.empty_cache()
        self._model = self._processor = None
