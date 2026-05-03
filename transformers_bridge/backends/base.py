from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    """Interface all inference backends must implement.

    Backends are plain Python objects with no ROS2 dependency.
    The node owns lifecycle, threading, and I/O.
    Backends own model loading and inference.
    """

    @abstractmethod
    def load(self, params: dict, logger) -> None:
        """Load model from params. Raise on failure so the node can return FAILURE."""

    @abstractmethod
    def warm_up(self, image_size: int) -> None:
        """Run dummy forward passes to compile CUDA kernels before first real frame."""

    @abstractmethod
    def infer(self, image: np.ndarray, image_size: int, threshold: float) -> list[dict]:
        """Run inference.

        Returns list of {"score": float, "label": str, "box": [x1, y1, x2, y2]}.
        Boxes are absolute pixel coordinates in the original (pre-resize) image space.
        """

    def unload(self) -> None:
        """Free GPU/CPU memory. Called on lifecycle cleanup."""
