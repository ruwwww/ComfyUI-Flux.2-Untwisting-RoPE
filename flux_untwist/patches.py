from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from comfy.ldm.cosmos.predict2 import apply_rotary_pos_emb

from .config import _PREFIX, clamp_float, schedule_fraction



def lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def build_frequency_scale_vector(
    head_dim: int,
    axes_dim: Sequence[int],
    high_scale: float,
    low_scale: float,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a per-channel scale vector for RoPE pair chunks.

    The vector is applied before ComfyUI applies RoPE. Since each 2D RoPE pair receives
    a scalar shared by both entries, that scaling commutes with the 2D rotation and is
    equivalent to scaling the already-rotated reference keys.
    """
    head_dim = int(head_dim)
    axes = [int(x) for x in axes_dim if int(x) > 0]
    if not axes or sum(axes) != head_dim:
        axes = [head_dim]

    beta = max(0.01, float(beta))
    pieces: List[torch.Tensor] = []
    has_three_axes = len(axes) == 3

    for axis_index, axis_dim in enumerate(axes):
        n_pairs = axis_dim // 2
        if n_pairs <= 0:
            pieces.append(torch.ones(axis_dim, device=device, dtype=dtype))
            continue

        # FLUX-style axes are normally [image/ref index, y, x]. The paper treats
        # the non-spatial partition as not responsible for spatial copying, so keep
        # it on the low-frequency/reference-preserving side instead of suppressing it.
        if has_three_axes and axis_index == 0:
            pair_scales = torch.full((n_pairs,), float(low_scale), device=device, dtype=torch.float32)
        else:
            if n_pairs == 1:
                d = torch.zeros((1,), device=device, dtype=torch.float32)
            else:
                d = torch.linspace(0.0, 1.0, n_pairs, device=device, dtype=torch.float32)
            pair_scales = float(high_scale) + (float(low_scale) - float(high_scale)) * d.pow(beta)

        pieces.append(pair_scales.to(dtype=dtype).repeat_interleave(2))
        if axis_dim % 2:
            pieces.append(torch.ones(1, device=device, dtype=dtype))

    out = torch.cat(pieces, dim=0) if pieces else torch.ones(head_dim, device=device, dtype=dtype)
    if out.numel() < head_dim:
        out = torch.nn.functional.pad(out, (0, head_dim - out.numel()), value=1.0)
    return out[:head_dim]


def _reference_ranges_from_options(seq_len: int, extra_options: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
    img_slice = extra_options.get("img_slice", None)
    ref_counts = extra_options.get("reference_image_num_tokens", None)
    if not isinstance(img_slice, (list, tuple)) or len(img_slice) != 2:
        return None, []

    try:
        img_start = max(0, min(int(img_slice[0]), int(seq_len)))
        img_end = max(img_start, min(int(img_slice[1]), int(seq_len)))
    except Exception:
        return None, []

    if ref_counts is None:
        return (img_start, img_end), []
    if isinstance(ref_counts, int):
        counts = [int(ref_counts)]
    elif isinstance(ref_counts, (list, tuple)):
        counts = []
        for value in ref_counts:
            try:
                count = int(value)
            except Exception:
                continue
            if count > 0:
                counts.append(count)
    else:
        return (img_start, img_end), []

    total_ref = sum(counts)
    total_img = img_end - img_start
    if total_ref <= 0 or total_ref > total_img:
        return (img_start, img_end), []

    ref_start = img_end - total_ref
    ranges: List[Tuple[int, int]] = []
    cur = ref_start
    for count in counts:
        end = min(img_end, cur + int(count))
        if end > cur:
            ranges.append((cur, end))
        cur = end
    return (img_start, ref_start), ranges


def _adain_tokens(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """AdaIN over the token dimension for tensors shaped [B, H, L, D]."""
    if target.numel() == 0 or style.numel() == 0:
        return target
    t_float = target.float()
    s_float = style.float()
    t_mean = t_float.mean(dim=2, keepdim=True)
    s_mean = s_float.mean(dim=2, keepdim=True)
    t_std = t_float.var(dim=2, keepdim=True, unbiased=False).add(eps).sqrt()
    s_std = s_float.var(dim=2, keepdim=True, unbiased=False).add(eps).sqrt()
    return ((t_float - t_mean) / t_std * s_std + s_mean).to(dtype=target.dtype)


def _concat_ranges(x: torch.Tensor, ranges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    parts = []
    length = int(x.shape[2])
    for start, end in ranges:
        s = max(0, min(int(start), length))
        e = max(s, min(int(end), length))
        if e > s:
            parts.append(x[:, :, s:e, :])
    if not parts:
        return x[:, :, 0:0, :]
    return torch.cat(parts, dim=2)


def flux_untwist_attn1_patch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pe: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """ComfyUI `attn1_patch` hook for FLUX single-stream blocks."""
    if extra_options is None:
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    cfg = extra_options.get("flux_untwist_rope", None)
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    # The paper applies reference sharing only in FLUX single-stream blocks.
    if str(extra_options.get("block_type", "")) != "single":
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    block_index = int(extra_options.get("block_index", -1))
    if block_index < int(cfg.get("start_single_block", 0)) or block_index > int(cfg.get("end_single_block", 999)):
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    seq_len = int(k.shape[2])
    target_range, ref_ranges = _reference_ranges_from_options(seq_len, extra_options)
    if not ref_ranges:
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    active, t = schedule_fraction(
        float(cfg.get("progress", 0.0)),
        float(cfg.get("start_percent", 0.0)),
        float(cfg.get("end_percent", 1.0)),
    )
    if not active:
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    high_scale = lerp(cfg.get("high_scale_start", 0.25), cfg.get("high_scale_end", 0.75), t)
    low_scale = lerp(cfg.get("low_scale_start", 1.0), cfg.get("low_scale_end", 1.4), t)
    beta = clamp_float(cfg.get("beta", 2.0), 0.01, 32.0, 2.0)

    scale_vec = build_frequency_scale_vector(
        head_dim=int(k.shape[-1]),
        axes_dim=cfg.get("axes_dim", []) or [],
        high_scale=high_scale,
        low_scale=low_scale,
        beta=beta,
        device=k.device,
        dtype=k.dtype,
    ).view(1, 1, 1, int(k.shape[-1]))

    q_out = q
    k_out = k.clone()

    for start, end in ref_ranges:
        k_out[:, :, start:end, :] = k_out[:, :, start:end, :] * scale_vec

    adain_strength = clamp_float(cfg.get("qk_adain_strength", 0.0), 0.0, 1.0, 0.0)
    if adain_strength > 0.0 and target_range is not None:
        target_start, target_end = target_range
        if target_end > target_start:
            ref_q = _concat_ranges(q, ref_ranges)
            ref_k = _concat_ranges(k, ref_ranges)
            if ref_q.shape[2] > 0 and ref_k.shape[2] > 0:
                q_out = q.clone()
                q_adain = _adain_tokens(q_out[:, :, target_start:target_end, :], ref_q)
                k_adain = _adain_tokens(k_out[:, :, target_start:target_end, :], ref_k)
                q_out[:, :, target_start:target_end, :] = (
                    q_out[:, :, target_start:target_end, :] * (1.0 - adain_strength)
                    + q_adain * adain_strength
                )
                k_out[:, :, target_start:target_end, :] = (
                    k_out[:, :, target_start:target_end, :] * (1.0 - adain_strength)
                    + k_adain * adain_strength
                )

    return {"q": q_out, "k": k_out, "v": v, "pe": pe, "attn_mask": attn_mask}


def anima_untwist_self_attn_patch(
    original_compute_qkv: Any,
    self: Any,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    rope_emb: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Monkey-patch for Anima Attention.compute_qkv (self-attention only)."""
    from einops import rearrange

    def _call_orig(x_in, ctx_in, rope_in):
        if hasattr(original_compute_qkv, "__self__"):
            return original_compute_qkv(x_in, context=ctx_in, rope_emb=rope_in)
        return original_compute_qkv(self, x_in, context=ctx_in, rope_emb=rope_in)

    # Only patch self-attention
    if not getattr(self, "is_selfattn", False):
        return _call_orig(x, context, rope_emb)

    # Walk up the call stack to retrieve transformer_options
    import sys
    transformer_options = None
    frame = sys._getframe(1)
    for _ in range(10):
        if frame is None:
            break
        to = frame.f_locals.get("transformer_options", None)
        if isinstance(to, dict):
            transformer_options = to
            break
        frame = frame.f_back

    if transformer_options is None:
        return _call_orig(x, context, rope_emb)

    cfg = transformer_options.get("anima_untwist_rope", None)
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return _call_orig(x, context, rope_emb)

    block_index = getattr(self, "_block_index", -1)
    if block_index < int(cfg.get("start_block", 0)) or block_index > int(cfg.get("end_block", 27)):
        return _call_orig(x, context, rope_emb)

    # Call original compute_qkv to get Q, K, V (with RoPE already applied)
    q, k, v = _call_orig(x, context, rope_emb)

    # Identify reference ranges
    ref_ranges = cfg.get("ref_ranges", [])
    if not ref_ranges:
        return q, k, v

    high_scale = float(cfg.get("high_scale", 1.0))
    low_scale = float(cfg.get("low_scale", 1.0))
    beta = float(cfg.get("beta", 2.0))
    axes_dim = cfg.get("axes_dim", []) or []

    # Build scale vector
    scale_vec = build_frequency_scale_vector(
        head_dim=int(self.head_dim),
        axes_dim=axes_dim,
        high_scale=high_scale,
        low_scale=low_scale,
        beta=beta,
        device=k.device,
        dtype=k.dtype,
    ).view(1, 1, 1, -1)

    k_out = k.clone()
    for start, end in ref_ranges:
        k_out[:, start:end, :, :] = k_out[:, start:end, :, :] * scale_vec

    # QK adain
    adain_strength = float(cfg.get("qk_adain_strength", 0.0))
    target_range = cfg.get("target_range", None)
    if adain_strength > 0.0 and target_range is not None:
        target_start, target_end = target_range
        q_out = q.clone()
        ref_q_parts = [q[:, r_start:r_end, :, :] for r_start, r_end in ref_ranges]
        ref_k_parts = [k[:, r_start:r_end, :, :] for r_start, r_end in ref_ranges]
        if ref_q_parts and ref_k_parts:
            ref_q = torch.cat(ref_q_parts, dim=1)
            ref_k = torch.cat(ref_k_parts, dim=1)

            # Permute to [B, H, S, D]
            target_q_perm = q_out[:, target_start:target_end, :, :].permute(0, 2, 1, 3)
            ref_q_perm = ref_q.permute(0, 2, 1, 3)
            q_adain_perm = _adain_tokens(target_q_perm, ref_q_perm)
            q_adain = q_adain_perm.permute(0, 2, 1, 3)

            target_k_perm = k_out[:, target_start:target_end, :, :].permute(0, 2, 1, 3)
            ref_k_perm = ref_k.permute(0, 2, 1, 3)
            k_adain_perm = _adain_tokens(target_k_perm, ref_k_perm)
            k_adain = k_adain_perm.permute(0, 2, 1, 3)

            q_out[:, target_start:target_end, :, :] = (
                q_out[:, target_start:target_end, :, :] * (1.0 - adain_strength)
                + q_adain * adain_strength
            )
            k_out[:, target_start:target_end, :, :] = (
                k_out[:, target_start:target_end, :, :] * (1.0 - adain_strength)
                + k_adain * adain_strength
            )
            q = q_out

    # Ensure all output tensors have matching dtypes to prevent any downstream errors
    k_out = k_out.to(dtype=q.dtype)
    v = v.to(dtype=q.dtype)

    return q, k_out, v


