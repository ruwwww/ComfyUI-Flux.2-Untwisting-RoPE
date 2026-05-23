# ComfyUI Flux.2 Untwisting RoPE

Frequency-aware RoPE reference-attention modulation for Flux/Flux.2 models in ComfyUI.

This custom node is a clean Flux/Flux.2-oriented implementation inspired by:

Aryan Mikaeili, Or Patashnik, Andrea Tagliasacchi, Daniel Cohen-Or, Ali Mahdavi-Amiri, **“Untwisting RoPE: Frequency Control for Shared Attention in DiTs,”** arXiv:2602.05013, 2026. <https://arxiv.org/abs/2602.05013>

It is designed for ComfyUI Flux-family models.

## What it does

The node patches Flux single-stream attention so appended reference-image tokens can contribute keys/values, while only the reference image **keys** receive a smooth RoPE-frequency scale:

- high-frequency RoPE key channels can be reduced to suppress spatial lock/copying;
- low-frequency RoPE key channels can be increased to preserve broader style/reference association;
- target keys, text keys, all queries, and values are left unchanged by default;
- the scaling is scheduled over denoising progress.

The implementation uses ComfyUI’s Flux transformer patch hook (`attn1_patch`) instead of monkey-patching Flux block classes. That keeps the implementation smaller and more tolerant of upstream Flux block changes.

## Node

### Flux.2 Untwist RoPE

Inputs:

| Input | Type | Purpose |
| --- | --- | --- |
| `model` | MODEL | Flux/Flux.2 model to patch. |
| `reference_latent` | LATENT, optional | Encoded reference image. Leave unconnected only if your conditioning/model kwargs already provide `ref_latents`. |
| `high_scale_start` / `high_scale_end` | FLOAT | Highest-frequency reference-key scale across the active denoising window. Values below 1 reduce spatial copying. |
| `low_scale_start` / `low_scale_end` | FLOAT | Lowest-frequency reference-key scale across the active window. Values above 1 increase global reference/style pull. |
| `beta` | FLOAT | Polynomial interpolation exponent across RoPE frequency chunks. Paper default: `2`. |
| `start_percent` / `end_percent` | FLOAT | Active denoising progress window. `0` is the start of sampling, `1` is the end. |
| `start_single_block` / `end_single_block` | INT | Inclusive Flux single-stream block range. |
| `reference_latents_method` | COMBO | Method passed to Flux for reference latent token positions: `index`, `offset`, `uxo`, `index_timestep_zero`. Default: `index`. |
| `qk_adain_strength` | FLOAT | Optional StyleAligned-style AdaIN on target image Q/K token statistics. Default `0` preserves native Flux.2 behavior. |
| `verbose` | BOOLEAN | Print per-call patch diagnostics. |

Output:

| Output | Type |
| --- | --- |
| `model` | MODEL |

## Suggested starting settings for Flux.2 / Flux.2 Klein

Start conservative:

```text
high_scale_start = 0.25
high_scale_end   = 0.75
low_scale_start  = 1.00
low_scale_end    = 1.40
beta             = 2.0
start_percent    = 0.0
end_percent      = 1.0
start_single_block = 0
end_single_block   = 999
reference_latents_method = index
qk_adain_strength = 0.0
```

Tuning guidance:

- Too much reference copying: lower `high_scale_start` and/or `high_scale_end`.
- Too weak style transfer: raise `low_scale_end` first.
- Late-stage texture too weak: raise `high_scale_end` slightly, not `high_scale_start`.
- Structural leakage or pose copying: keep `reference_latents_method=index`; avoid spatially shifted methods first.
- Washed or unstable attention: keep `qk_adain_strength=0.0`; only try AdaIN after key scaling is validated.

## Installation

Clone or copy this repo into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Flux.2-Untwisting-RoPE
```

There are no extra Python package dependencies beyond ComfyUI and PyTorch.

Restart ComfyUI.

## Basic workflow placement

Typical Flux.2 style-reference graph:

1. Load Flux.2 / Flux.2 Klein model normally.
2. Encode your reference image with the Flux.2 VAE.
3. Connect the encoded reference LATENT to `Flux.2 Untwist RoPE.reference_latent`.
4. Put `Flux.2 Untwist RoPE` between the model loader and the guider/sampler.
5. Use the patched model for sampling.

The node passes the reference latent through Flux’s native `ref_latents` path, then modifies single-stream attention via ComfyUI’s transformer patch mechanism.

## Design notes

This repo intentionally does not copy the unofficial Z-Image implementation structure. The source repo is useful as a proof that frequency-aware reference-key scaling can be exposed in ComfyUI, but its architecture is not the right target for Flux.2:

- it patches Z-Image/NextDiT internals directly;
- it assumes `patchify_and_embed`, `layers.0..29.attention`, and Z-Image-specific masks;
- it includes a large RF inversion subsystem that is not necessary for Flux.2’s native reference-latent path;
- it stores most behavior in one large `__init__.py`, making regressions difficult to isolate.

This implementation instead uses Flux’s existing reference latent handling and ComfyUI’s attention patch API.

## Validation plan

Minimum validation before publishing:

1. Import smoke test: ComfyUI starts and loads `Flux.2 Untwist RoPE` without custom-node import errors.
2. Baseline parity: with no `reference_latent`, output should match the unpatched model except for negligible wrapper overhead.
3. No-op patch check: with a reference connected but `start_percent/end_percent` excluding all steps, sampling should behave like no reference from this node.
4. Token range check: enable `verbose` and confirm reference token counts are detected during single-stream blocks.
5. A/B visual tests at fixed seed:
   - no reference;
   - native Flux.2 reference workflow;
   - this node at defaults;
   - lower `high_scale_end` for copying-heavy references;
   - higher `low_scale_end` for weak style transfer.
6. Resolution sweep: test at the target Flux.2 resolutions you actually use, especially 2–4 MP, because reference token count and attention cost scale with resolution.

## Known risks

- Flux.2 already has native reference conditioning. This node changes how reference tokens influence attention; it may improve style transfer in some cases and harm edit/identity fidelity in others.
- The implementation assumes current ComfyUI Flux blocks expose `attn1_patch`, `img_slice`, `reference_image_num_tokens`, and `block_type="single"`. If ComfyUI changes these hook contracts, the node will safely degrade to a no-op or need a small compatibility patch.
- Appending reference latents increases attention context length and VRAM use. High-resolution references can be expensive.
- `qk_adain_strength` is optional and disabled by default because it is more invasive than reference-key frequency scaling.

## License

MIT. The paper and any model weights remain under their own licenses.
