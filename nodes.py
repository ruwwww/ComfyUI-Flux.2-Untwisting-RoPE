from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from .flux_untwist.config import (
        _PREFIX,
        UntwistConfig,
        clamp_float,
        clamp_int,
        coerce_bool,
        normalize_percent_window,
        normalize_reference_method,
        safe_axes_dim,
        schedule_fraction,
    )
    from .flux_untwist.patches import flux_untwist_attn1_patch
    from .flux_untwist.utils import (
        append_transformer_patch,
        clone_model_options,
        normalize_ref_latents,
        process_latent_for_model,
        progress_from_timestep,
        repeat_to_batch,
        safe_get_diffusion_model,
    )
except ImportError:
    from flux_untwist.config import (
        _PREFIX,
        UntwistConfig,
        clamp_float,
        clamp_int,
        coerce_bool,
        normalize_percent_window,
        normalize_reference_method,
        safe_axes_dim,
        schedule_fraction,
    )
    from flux_untwist.patches import flux_untwist_attn1_patch
    from flux_untwist.utils import (
        append_transformer_patch,
        clone_model_options,
        normalize_ref_latents,
        process_latent_for_model,
        progress_from_timestep,
        repeat_to_batch,
        safe_get_diffusion_model,
    )


class Flux2UntwistRoPE:
    """Patch FLUX/FLUX.2 reference attention with frequency-aware RoPE key scaling."""

    CATEGORY = "model_patches/Flux2 Untwisting RoPE"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "Adds reference latents to a Flux/Flux.2 model and scales only reference image keys "
        "in single-stream attention according to RoPE frequency bands."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "high_scale_start": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Scale for highest-frequency reference-key RoPE channels at the start of the active window. <1 reduces positional copying.",
                    },
                ),
                "high_scale_end": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Scale for highest-frequency reference-key RoPE channels at the end of the active window.",
                    },
                ),
                "low_scale_start": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Scale for lowest-frequency reference-key RoPE channels at the start of the active window.",
                    },
                ),
                "low_scale_end": (
                    "FLOAT",
                    {
                        "default": 1.40,
                        "min": -4.0,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Scale for lowest-frequency reference-key RoPE channels at the end of the active window. >1 increases global style/reference pull.",
                    },
                ),
                "beta": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.01,
                        "max": 32.0,
                        "step": 0.01,
                        "tooltip": "Polynomial interpolation exponent from high-frequency to low-frequency scales. Paper default is 2.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "start_single_block": (
                    "INT",
                    {"default": 0, "min": 0, "max": 999, "step": 1},
                ),
                "end_single_block": (
                    "INT",
                    {"default": 999, "min": 0, "max": 999, "step": 1},
                ),
                "reference_latents_method": (
                    ["index", "offset", "uxo", "index_timestep_zero"],
                    {
                        "default": "index",
                        "tooltip": "How Flux positions appended reference latent tokens. index keeps the spatial grid aligned and differs only by image index; offset/uxo use spatial offsets.",
                    },
                ),
                "qk_adain_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Optional StyleAligned-style AdaIN on target image Q/K token statistics. 0 preserves native Flux.2 behavior.",
                    },
                ),
                "verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_latent": (
                    "LATENT",
                    {
                        "tooltip": "Encoded reference image latent. You may leave this unconnected if your conditioning already supplies ref_latents.",
                    },
                ),
            },
        }

    def patch(
        self,
        model: Any,
        high_scale_start: float,
        high_scale_end: float,
        low_scale_start: float,
        low_scale_end: float,
        beta: float,
        start_percent: float,
        end_percent: float,
        start_single_block: int,
        end_single_block: int,
        reference_latents_method: str,
        qk_adain_strength: float,
        verbose: bool = False,
        reference_latent: Optional[Dict[str, Any]] = None,
    ):
        node_verbose = coerce_bool(verbose)
        start_percent, end_percent = normalize_percent_window(start_percent, end_percent)
        start_single_block = clamp_int(start_single_block, 0, 999, 0)
        end_single_block = clamp_int(end_single_block, 0, 999, 999)
        if end_single_block < start_single_block:
            start_single_block, end_single_block = end_single_block, start_single_block

        ref_samples_cpu: Optional[torch.Tensor] = None
        if isinstance(reference_latent, dict) and torch.is_tensor(reference_latent.get("samples")):
            samples = reference_latent["samples"].detach()
            if samples.ndim != 4:
                raise RuntimeError(f"reference_latent['samples'] must be BCHW; got shape {tuple(samples.shape)}")
            ref_samples_cpu = process_latent_for_model(model, samples).detach().to(device="cpu").clone()
        elif reference_latent is not None:
            raise RuntimeError("reference_latent must be a ComfyUI LATENT dict with a tensor 'samples' entry.")

        model_clone = model.clone()
        dm = safe_get_diffusion_model(model_clone)
        axes_dim = safe_axes_dim(dm)
        if not axes_dim:
            # Leave the patch usable on forks that keep Flux shape but hide params.axes_dim.
            axes_dim = tuple()

        model_clone.model_options = clone_model_options(getattr(model_clone, "model_options", {}) or {})
        old_wrapper = model_clone.model_options.get("model_function_wrapper", None)
        append_transformer_patch(model_clone, "attn1_patch", flux_untwist_attn1_patch)
        method = normalize_reference_method(reference_latents_method)

        # Values are copied into transformer_options each call; the attention patch reads them there.
        base_high_start = clamp_float(high_scale_start, -4.0, 8.0, 0.25)
        base_high_end = clamp_float(high_scale_end, -4.0, 8.0, 0.75)
        base_low_start = clamp_float(low_scale_start, -4.0, 8.0, 1.0)
        base_low_end = clamp_float(low_scale_end, -4.0, 8.0, 1.4)
        base_beta = clamp_float(beta, 0.01, 32.0, 2.0)
        base_adain = clamp_float(qk_adain_strength, 0.0, 1.0, 0.0)

        def model_function_wrapper(apply_model, args: Dict[str, Any]):
            input_x = args["input"]
            timestep = args["timestep"]
            c = dict(args["c"])
            cond_or_uncond = args.get("cond_or_uncond", None)

            progress = progress_from_timestep(timestep)
            active, _t = schedule_fraction(progress, start_percent, end_percent)

            to = dict(c.get("transformer_options", {}) or {})
            existing_refs = normalize_ref_latents(c.get("ref_latents", None))
            refs: List[torch.Tensor] = []

            if ref_samples_cpu is not None and active:
                ref = ref_samples_cpu.to(device=input_x.device, dtype=input_x.dtype)
                ref = repeat_to_batch(ref, int(input_x.shape[0]))
                refs.append(ref)

            # Preserve explicit reference latents already present in conditioning/model kwargs.
            if existing_refs:
                for ref in existing_refs:
                    refs.append(repeat_to_batch(ref.to(device=input_x.device, dtype=input_x.dtype), int(input_x.shape[0])))

            cfg = UntwistConfig(
                enabled=bool(active and refs),
                axes_dim=axes_dim,
                high_scale_start=base_high_start,
                high_scale_end=base_high_end,
                low_scale_start=base_low_start,
                low_scale_end=base_low_end,
                beta=base_beta,
                start_percent=start_percent,
                end_percent=end_percent,
                start_single_block=start_single_block,
                end_single_block=end_single_block,
                qk_adain_strength=base_adain,
                reference_latents_method=method,
                progress=progress,
                verbose=node_verbose,
            )
            to["flux_untwist_rope"] = cfg.as_transformer_options()
            c["transformer_options"] = to

            if refs:
                c["ref_latents"] = refs
                # Do not override a method already set directly by upstream conditioning unless
                # this node supplies the reference. That keeps native workflows composable.
                if ref_samples_cpu is not None or "ref_latents_method" not in c:
                    c["ref_latents_method"] = method

            if node_verbose:
                ref_shapes = [tuple(r.shape) for r in refs]
                print(
                    f"{_PREFIX} call: progress={progress:.3f} active={active} refs={len(refs)} "
                    f"method={c.get('ref_latents_method', '<native>')} shapes={ref_shapes}"
                )

            if old_wrapper is not None:
                return old_wrapper(
                    apply_model,
                    {
                        "input": input_x,
                        "timestep": timestep,
                        "c": c,
                        "cond_or_uncond": cond_or_uncond,
                    },
                )
            return apply_model(input_x, timestep, **c)

        model_clone.set_model_unet_function_wrapper(model_function_wrapper)

        if node_verbose:
            print(f"{_PREFIX} patched Flux-like model: {type(dm).__name__}")
            print(f"{_PREFIX} axes_dim={axes_dim or '<unknown>'} blocks={start_single_block}..{end_single_block}")
            print(f"{_PREFIX} high={base_high_start:.3f}->{base_high_end:.3f} low={base_low_start:.3f}->{base_low_end:.3f} beta={base_beta:.3f}")

        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "Flux2UntwistRoPE": Flux2UntwistRoPE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2UntwistRoPE": "Flux.2 Untwist RoPE",
}
