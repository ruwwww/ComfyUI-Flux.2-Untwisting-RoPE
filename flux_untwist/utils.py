from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def repeat_to_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    """Repeat a tensor along batch dim using ComfyUI's helper when available."""
    if int(x.shape[0]) == int(batch):
        return x
    try:
        import comfy.utils  # type: ignore

        return comfy.utils.repeat_to_batch_size(x, int(batch))
    except Exception:
        reps = math.ceil(int(batch) / max(1, int(x.shape[0])))
        return x.repeat((reps,) + (1,) * (x.ndim - 1))[: int(batch)]


def clone_model_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Copy only the mutable parts we mutate: transformer_options.patch lists."""
    out = dict(options or {})
    transformer_options = dict(out.get("transformer_options", {}) or {})
    patches = transformer_options.get("patches", {}) or {}
    patches_copy: Dict[str, List[Any]] = {}
    for key, value in patches.items():
        patches_copy[key] = list(value) if isinstance(value, list) else [value]
    transformer_options["patches"] = patches_copy
    out["transformer_options"] = transformer_options
    return out


def append_transformer_patch(model: Any, patch_name: str, patch_fn: Any) -> None:
    """Append a ComfyUI transformer patch without relying on optional ModelPatcher helpers."""
    model.model_options = clone_model_options(getattr(model, "model_options", {}) or {})
    transformer_options = model.model_options.setdefault("transformer_options", {})
    patches = transformer_options.setdefault("patches", {})
    patches.setdefault(patch_name, []).append(patch_fn)


def sigma_from_timestep(timestep: Any) -> float:
    """Normalize Comfy/Flux timestep-like values into a sigma-like [0, 1] value."""
    try:
        if torch.is_tensor(timestep):
            value = float(timestep.detach().float().mean().item())
        else:
            value = float(timestep)
        if not math.isfinite(value):
            return 1.0
        if 0.0 <= value <= 1.0:
            return max(0.0, min(1.0, value))
        if 1.0 < value <= 1000.0:
            return max(0.0, min(1.0, value / 1000.0))
    except Exception:
        pass
    return 1.0


def progress_from_timestep(timestep: Any) -> float:
    return max(0.0, min(1.0, 1.0 - sigma_from_timestep(timestep)))


def safe_get_diffusion_model(model_patcher: Any) -> Any:
    """Find the wrapped Flux diffusion model in common ComfyUI ModelPatcher layouts."""
    roots: List[Any] = []
    if hasattr(model_patcher, "model"):
        roots.append(model_patcher.model)
    roots.append(model_patcher)

    attr_paths = (
        "diffusion_model",
        "model.diffusion_model",
        "model.model.diffusion_model",
        "inner_model.diffusion_model",
        "model.inner_model.diffusion_model",
    )
    for root in roots:
        for path in attr_paths:
            obj = root
            ok = True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and _looks_like_flux(obj):
                return obj

    seen = set()
    stack = list(roots)
    while stack and len(seen) < 512:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if _looks_like_flux(obj):
            return obj
        for name in ("model", "inner_model", "diffusion_model", "unet", "wrapped"):
            if hasattr(obj, name):
                try:
                    stack.append(getattr(obj, name))
                except Exception:
                    pass
    raise RuntimeError("Could not locate a Flux-like diffusion model on the supplied MODEL.")


def _looks_like_flux(obj: Any) -> bool:
    return (
        hasattr(obj, "process_img")
        and hasattr(obj, "single_blocks")
        and hasattr(obj, "double_blocks")
        and hasattr(obj, "params")
    )


def process_latent_for_model(model_patcher: Any, latent_samples: torch.Tensor) -> torch.Tensor:
    """Convert a ComfyUI LATENT tensor to model input latent space when available."""
    processor = None
    try:
        processor = getattr(getattr(model_patcher, "model", None), "process_latent_in", None)
    except Exception:
        processor = None
    if callable(processor):
        return processor(latent_samples)
    return latent_samples


def normalize_ref_latents(existing: Any) -> List[torch.Tensor]:
    if existing is None:
        return []
    if torch.is_tensor(existing):
        return [existing]
    if isinstance(existing, (list, tuple)):
        return [x for x in existing if torch.is_tensor(x)]
    return []
