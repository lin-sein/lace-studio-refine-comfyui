from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_MODULE_NAME = "lace_studio_refine_v01_runtime"
OPENCV_DISTRIBUTIONS = (
    "opencv-python-headless",
    "opencv-python",
    "opencv-contrib-python-headless",
    "opencv-contrib-python",
)


def _runtime_root() -> Path:
    bundled = PACKAGE_ROOT / "runtime"
    if (bundled / "scripts" / "generalized-full-image-refine.py").is_file():
        return bundled
    for candidate in PACKAGE_ROOT.parents:
        if (candidate / "scripts" / "generalized-full-image-refine.py").is_file():
            return candidate
    raise RuntimeError("Lace Studio refinement runtime is missing")


def _is_usable_refiner(module: Any, source: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file or not callable(getattr(module, "refine_array", None)):
        return False
    try:
        return Path(module_file).resolve() == source.resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _installed_opencv_versions() -> str:
    installed: list[str] = []
    for distribution in OPENCV_DISTRIBUTIONS:
        try:
            installed.append(
                f"{distribution}={importlib.metadata.version(distribution)}"
            )
        except importlib.metadata.PackageNotFoundError:
            continue
    return ", ".join(installed) if installed else "none detected"


def _load_refiner() -> Any:
    runtime_root = _runtime_root()
    source = runtime_root / "scripts" / "generalized-full-image-refine.py"
    cached = sys.modules.get(RUNTIME_MODULE_NAME)
    if cached is not None and _is_usable_refiner(cached, source):
        return cached
    if cached is not None:
        sys.modules.pop(RUNTIME_MODULE_NAME, None)

    spec = importlib.util.spec_from_file_location(RUNTIME_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Lace Studio refinement runtime: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[RUNTIME_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        if not _is_usable_refiner(module, source):
            raise AttributeError("runtime did not define callable refine_array")
    except Exception as error:
        if sys.modules.get(RUNTIME_MODULE_NAME) is module:
            sys.modules.pop(RUNTIME_MODULE_NAME, None)
        raise RuntimeError(
            "Lace Studio refinement runtime import failed; "
            f"numpy={np.__version__}; OpenCV distributions: "
            f"{_installed_opencv_versions()}; original error: "
            f"{type(error).__name__}: {error}. Reinstall this node's "
            "requirements in the ComfyUI Python environment and restart ComfyUI."
        ) from error
    return module


class LaceStudioRefineV01:
    """Deterministic lace cleanup and repeat reconstruction for ComfyUI."""

    CATEGORY = "Lace Studio/Refine v0.1"
    FUNCTION = "refine"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("refined_image", "foreground_mask", "report_json")
    DESCRIPTION = (
        "Corrects lace placement/repeat geometry, removes scanner backgrounds, "
        "and returns a topology-preserving foreground mask without model generation."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "background": (["auto", "white", "black"], {"default": "auto"}),
                "repeat_reconstruction": (["auto", "off"], {"default": "auto"}),
                "edge_cleanup": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                    },
                ),
            }
        }

    def refine(
        self,
        image: torch.Tensor,
        background: str,
        repeat_reconstruction: str,
        edge_cleanup: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 4
            or image.shape[-1] < 3
        ):
            raise ValueError(
                "image must be a ComfyUI IMAGE tensor with shape [B,H,W,C]"
            )
        refiner = _load_refiner()
        runtime_root = _runtime_root()
        outputs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        reports: list[dict[str, Any]] = []
        for batch_index, item in enumerate(image):
            rgb = np.clip(
                item[..., :3].detach().cpu().numpy() * 255.0 + 0.5,
                0,
                255,
            ).astype(np.uint8)
            refined, mask, report = refiner.refine_array(
                rgb,
                repo_root=runtime_root,
                target_background=background,
                repeat_reconstruction=repeat_reconstruction,
                edge_cleanup=int(edge_cleanup),
            )
            outputs.append(refined.astype(np.float32) / 255.0)
            masks.append(mask.astype(np.float32))
            reports.append({"batchIndex": batch_index, **report})

        output_tensor = torch.from_numpy(np.stack(outputs)).to(
            device=image.device, dtype=torch.float32
        )
        mask_tensor = torch.from_numpy(np.stack(masks)).to(
            device=image.device, dtype=torch.float32
        )
        report_json = json.dumps(
            {
                "schemaVersion": 1,
                "engine": "lace-studio-comfyui-v0.1",
                "settings": {
                    "background": background,
                    "repeatReconstruction": repeat_reconstruction,
                    "edgeCleanup": int(edge_cleanup),
                },
                "batches": reports,
            },
            ensure_ascii=False,
        )
        return output_tensor, mask_tensor, report_json


NODE_CLASS_MAPPINGS = {"LaceStudioRefineV01": LaceStudioRefineV01}
NODE_DISPLAY_NAME_MAPPINGS = {"LaceStudioRefineV01": "Lace Studio 精修 v0.1"}
