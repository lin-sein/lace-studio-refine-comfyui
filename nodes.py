from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parent


def _runtime_root() -> Path:
    bundled = PACKAGE_ROOT / "runtime"
    if (bundled / "scripts" / "generalized-full-image-refine.py").is_file():
        return bundled
    for candidate in PACKAGE_ROOT.parents:
        if (candidate / "scripts" / "generalized-full-image-refine.py").is_file():
            return candidate
    raise RuntimeError("Lace Studio refinement runtime is missing")


def _load_refiner() -> Any:
    runtime_root = _runtime_root()
    source = runtime_root / "scripts" / "generalized-full-image-refine.py"
    module_name = "lace_studio_refine_v01_runtime"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Lace Studio refinement runtime: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
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
                    {"default": 50, "min": 0, "max": 100, "step": 1, "display": "slider"},
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
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("image must be a ComfyUI IMAGE tensor with shape [B,H,W,C]")
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

        output_tensor = torch.from_numpy(np.stack(outputs)).to(device=image.device, dtype=torch.float32)
        mask_tensor = torch.from_numpy(np.stack(masks)).to(device=image.device, dtype=torch.float32)
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
