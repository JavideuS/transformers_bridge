from scripts.venv_setup import auto_inject_venv
auto_inject_venv(packages=['transformers'])

from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor

try:
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
except ImportError:
    RTDetrImageProcessor = AutoImageProcessor       # type: ignore[assignment,misc]
    RTDetrV2ForObjectDetection = AutoModelForObjectDetection  # type: ignore[assignment,misc]

# Each entry is keyed by a lowercase substring that must appear in the model name.
# Fields:
#   processor_cls  – class with a .from_pretrained() classmethod
#   model_cls      – class with a .from_pretrained() classmethod
#   tested         – True if the combo has been verified end-to-end in this package
#   notes          – human-readable hint shown at load time
#   extra_params   – ROS 2 parameters that must be declared for this model to work
REGISTRY: dict[str, dict] = {
    "rtdetr": {
        "processor_cls": RTDetrImageProcessor,
        "model_cls": RTDetrV2ForObjectDetection,
        "notes": "Use image_size=640",
        "tested": True,
    },
    "detr": {
        "processor_cls": AutoImageProcessor,
        "model_cls": AutoModelForObjectDetection,
        "tested": False,
    },
    "yolos": {
        "processor_cls": AutoImageProcessor,
        "model_cls": AutoModelForObjectDetection,
        "tested": False,
    },
    "grounding-dino": {
        "processor_cls": AutoProcessor,
        "model_cls": AutoModelForObjectDetection,
        "notes": "Requires text_prompt parameter",
        "extra_params": ["text_prompt"],
        "tested": False,
    },
}

_FALLBACK: dict = {
    "processor_cls": AutoImageProcessor,
    "model_cls": AutoModelForObjectDetection,
}


import os
import json
from pathlib import Path

def resolve_model_config(model_name: str) -> tuple[dict, str | None]:
    """Return (config, matched_key).

    For local directories: reads config.json first (authoritative), then falls
    back to substring match on the path.  For HuggingFace IDs (no local dir):
    substring match only.
    """
    model_path = Path(model_name)

    # 1. Local checkpoint — read config.json before guessing from the path name
    if model_path.is_dir():
        config_file = model_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                # architectures field is the most reliable signal
                for arch in config_data.get("architectures", []):
                    arch_lower = arch.lower()
                    for key, cfg in REGISTRY.items():
                        if key in arch_lower:
                            return cfg, key
                # model_type as secondary fallback
                model_type = config_data.get("model_type", "").lower()
                if model_type:
                    for key, cfg in REGISTRY.items():
                        if key in model_type:
                            return cfg, key
            except Exception:
                pass

    # 2. Substring match on model name / HuggingFace ID
    lower = model_name.lower()
    for key, cfg in REGISTRY.items():
        if key in lower:
            return cfg, key

    return _FALLBACK, None


def main() -> None:
    """Print the model registry as a table.

    Entry point: ros2 run transformers_bridge list_models
    """
    col = (28, 30, 32, 8, 16)
    header = (
        f"{'Key':<{col[0]}}"
        f"{'Processor':<{col[1]}}"
        f"{'Model':<{col[2]}}"
        f"{'Tested':<{col[3]}}"
        f"{'Extra params':<{col[4]}}"
        f"Notes"
    )
    sep = "-" * (sum(col) + 10)

    print("\nTransformers Bridge — Model Registry")
    print(sep)
    print(header)
    print(sep)

    for key, cfg in REGISTRY.items():
        proc = cfg["processor_cls"].__name__
        mdl = cfg["model_cls"].__name__
        tested = "yes" if cfg.get("tested") else "no"
        extra = ", ".join(cfg.get("extra_params", [])) or "-"
        notes = cfg.get("notes", "-")
        print(
            f"{key:<{col[0]}}"
            f"{proc:<{col[1]}}"
            f"{mdl:<{col[2]}}"
            f"{tested:<{col[3]}}"
            f"{extra:<{col[4]}}"
            f"{notes}"
        )

    print(sep)
    fb_proc = _FALLBACK["processor_cls"].__name__
    fb_mdl = _FALLBACK["model_cls"].__name__
    print(f"Fallback (no match): {fb_proc} / {fb_mdl}\n")


if __name__ == "__main__":
    main()
