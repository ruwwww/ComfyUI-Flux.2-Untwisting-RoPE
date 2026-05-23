# Source analysis and design decisions

## Paper behavior to preserve

The paper’s core mechanism is not “style transfer by any reference image.” It is a targeted attention intervention:

1. RoPE decomposes into frequency components with different positional sensitivity.
2. High-frequency RoPE components make shared attention strongly local and can cause reference copying.
3. Low-frequency components preserve broader/global association.
4. The intervention should modulate **reference image keys** in shared attention, not target/text keys and not values.
5. The intervention is intended for Flux-style MMDiT single-stream blocks.
6. The frequency scale should be smooth across the RoPE spectrum and schedulable over denoising.

Those points are preserved.

## Unofficial source repo behavior

The attached unofficial implementation is a single-file ComfyUI custom node targeting Z-Image/NextDiT rather than Flux/Flux.2. Its real behavior is:

- `RFInversion` stores a reference latent and captures sampler sigmas through a sampler wrapper.
- `UntwistingRoPE` clones a model, finds a Z-Image/NextDiT-like diffusion model, monkey-patches `patchify_and_embed`, context-refiner mask modules, and `layers.N.attention` modules.
- The attention patch creates a doubled target/reference batch, runs target queries against target + reference keys/values, scales reference keys with a frequency vector, and discards the reference branch prediction.
- It has optional cross-batch AdaIN for Q/K and a large RF trajectory cache/inversion subsystem.
- It assumes Z-Image names and shapes: `patchify_and_embed`, `layers.0..29.attention`, `context_refiner`, `n_local_heads`, `n_local_kv_heads`, `head_dim`, and `optimized_attention_masked` usage.

## Preserve

- Frequency-aware per-pair key scaling.
- Smooth polynomial high-to-low frequency interpolation.
- Denoising-progress schedule for high/low scales.
- Single-stream-only default behavior.
- Optional Q/K AdaIN as an exposed, disabled-by-default experiment.
- Conservative fallback behavior: if no reference token ranges are present, the attention patch returns unchanged tensors.

## Rewrite

- The entire ComfyUI integration layer.
- Reference injection: use Flux’s native `ref_latents` path instead of constructing a doubled batch and discarding half the prediction.
- Attention integration: use ComfyUI’s `attn1_patch` hook instead of replacing module `forward` methods.
- Token range detection: use Flux-provided `img_slice` and `reference_image_num_tokens` from transformer options instead of Z-Image patchify metadata.
- Model detection: detect Flux-like models by `process_img`, `single_blocks`, `double_blocks`, and `params`, not Z-Image internals.

## Fix

- Remove hard-coded `layers.0..29.attention` patching.
- Remove Z-Image-specific mask surgery.
- Avoid mutation of global/module state for sampler trajectory cache.
- Avoid in-place modification of original `model_options` patch lists.
- Avoid one giant `__init__.py`; split config, utilities, patches, and nodes.
- Use paper-consistent `beta=2.0` default instead of an extreme default.
- Default to non-spatial reference indexing (`index`) rather than shifted spatial placement.

## Drop

- RF inversion subsystem. For Flux.2, ComfyUI already exposes native reference latent handling. RF inversion adds model calls, persistent state, cache invalidation hazards, and sampler-wrapper coupling that are not justified for the Flux.2 target path.
- Z-Image/NextDiT support. It is a different architecture target and would dilute the implementation.
- Context-refiner cap-mask patching. Flux.2 does not need the Z-Image mask workaround.
- Direct attention module monkey-patching. It is brittle against upstream ComfyUI changes.
- Example images and large workflow JSON copied from the source repo.

## New architecture

```text
ComfyUI-Flux.2-Untwisting-RoPE/
  __init__.py                  # ComfyUI node registration
  nodes.py                     # public node class
  flux_untwist/
    config.py                  # config normalization and scheduling
    patches.py                 # Flux attn1_patch implementation
    utils.py                   # model/latent/model_options helpers
  tests/
    test_frequency_scale.py    # pure unit tests for core math/ranges
  README.md
  SOURCE_ANALYSIS.md
  pyproject.toml
  LICENSE
```

## Main invariant

When active and reference tokens are present:

- only reference image key channels are frequency-scaled;
- target keys are unchanged except for optional AdaIN when explicitly enabled;
- text keys are unchanged;
- all values are unchanged;
- all queries are unchanged except for optional AdaIN when explicitly enabled;
- if reference ranges cannot be identified, the patch is a no-op.

## Assumptions and deviations

- Assumes ComfyUI Flux/Flux.2 still provides `attn1_patch`, `img_slice`, `reference_image_num_tokens`, and `block_type` in `extra_options` / `transformer_options`.
- Assumes Flux.2 reference latents are accepted through the same `ref_latents` kwarg path used by current ComfyUI Flux code.
- Uses pre-RoPE key scaling because ComfyUI’s `attn1_patch` runs before `attention()` applies RoPE. This is equivalent for this operation because each 2D RoPE pair is multiplied by one scalar shared across the pair, and scalar multiplication commutes with the 2D rotation.
- Defaults are conservative for Flux.2 because Flux.2 already has trained/native reference behavior. Stronger settings may be useful but should be validated with fixed-seed A/B tests.

## Regression risks

- Upstream ComfyUI may rename or remove the Flux transformer patch hooks.
- Some Flux.2 forks may store reference latents in conditioning under a different key than `ref_latents`; direct `reference_latent` input remains the primary supported path.
- Very large reference images increase sequence length and attention cost.
- `index_timestep_zero` is supported because Flux code paths exist for it, but it is not the default because it changes modulation semantics more than plain `index`.
