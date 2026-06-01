# AI agent instructions for Anima model

**Session ID:** ses_17f67f5beffeKq0EJMHZjfupbo
**Created:** 6/1/2026, 12:31:48 AM
**Updated:** 6/1/2026, 12:44:29 AM

---

## User

can you create an isntruction for ai agent to implement this for anima model instead 

---

## Assistant (Build · DeepSeek V4 Flash · 35.0s)

_Thinking:_

The user wants me to create an instruction for an AI agent to implement something for the "Anima" model instead. I need to first understand what this repository does and what "this" refers to. Let me explore the codebase to understand the current implementation.

**Tool: task**

**Input:**
```json
{
  "description": "Explore codebase structure",
  "prompt": "Explore the codebase at /home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE thoroughly. I need to understand:\n\n1. What is the purpose of this custom node?\n2. What model does it currently target (Flux?).\n3. What files exist and what are the key classes/nodes defined?\n4. How does the implementation work - what are the core operations?\n5. Are there any README or documentation files explaining the approach?\n\nReturn a detailed summary of the codebase structure, key files, classes/functions, and the core logic/flows. Include the full content of the most important files (under 200 lines) and snippets of larger files.",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_17f67df4bffePrSg7cAJyT8Dqv" state="completed">
<task_result>

</task_result>
</task>
```

---

## Assistant (Build · DeepSeek V4 Flash · 3.4s)

---

## User

continue

---

## Assistant (Build · DeepSeek V4 Flash · 3.7s)

_Thinking:_

The explore agent didn't return a result. Let me read the files directly to understand the codebase structure.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE"
}
```

**Output:**
```
<path>/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE</path>
<type>directory</type>
<entries>
__init__.py
__pycache__/
.git/
.gitignore
flux_untwist/
LICENSE
nodes.py
pyproject.toml
README.md
SOURCE_ANALYSIS.md
tests/

(11 entries)
</entries>
```

---

## Assistant (Build · DeepSeek V4 Flash · 2.3s)

---

## User

continue

---

## Assistant (Build · MiMo V2.5 Free · 24.0s)

_Thinking:_

Let me read the key files to understand what this codebase does.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/README.md"
}
```

**Output:**
```
<path>/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/README.md</path>
<type>file</type>
<content>
1: # ComfyUI Flux.2 Untwisting RoPE
2: 
3: Frequency-aware RoPE reference-attention modulation for Flux/Flux.2 models in ComfyUI.
4: 
5: This custom node is a clean Flux/Flux.2-oriented implementation inspired by:
6: 
7: Aryan Mikaeili, Or Patashnik, Andrea Tagliasacchi, Daniel Cohen-Or, Ali Mahdavi-Amiri, **“Untwisting RoPE: Frequency Control for Shared Attention in DiTs,”** arXiv:2602.05013, 2026. <https://arxiv.org/abs/2602.05013>
8: 
9: It is designed for ComfyUI Flux-family models.
10: 
11: ## What it does
12: 
13: The node patches Flux single-stream attention so appended reference-image tokens can contribute keys/values, while only the reference image **keys** receive a smooth RoPE-frequency scale:
14: 
15: - high-frequency RoPE key channels can be reduced to suppress spatial lock/copying;
16: - low-frequency RoPE key channels can be increased to preserve broader style/reference association;
17: - target keys, text keys, all queries, and values are left unchanged by default;
18: - the scaling is scheduled over denoising progress.
19: 
20: The implementation uses ComfyUI’s Flux transformer patch hook (`attn1_patch`) instead of monkey-patching Flux block classes. That keeps the implementation smaller and more tolerant of upstream Flux block changes.
21: 
22: ## Node
23: 
24: ### Flux.2 Untwist RoPE
25: 
26: Inputs:
27: 
28: | Input | Type | Purpose |
29: | --- | --- | --- |
30: | `model` | MODEL | Flux/Flux.2 model to patch. |
31: | `reference_latent` | LATENT, optional | Encoded reference image. Leave unconnected only if your conditioning/model kwargs already provide `ref_latents`. |
32: | `high_scale_start` / `high_scale_end` | FLOAT | Highest-frequency reference-key scale across the active denoising window. Values below 1 reduce spatial copying. |
33: | `low_scale_start` / `low_scale_end` | FLOAT | Lowest-frequency reference-key scale across the active window. Values above 1 increase global reference/style pull. |
34: | `beta` | FLOAT | Polynomial interpolation exponent across RoPE frequency chunks. Paper default: `2`. |
35: | `start_percent` / `end_percent` | FLOAT | Active denoising progress window. `0` is the start of sampling, `1` is the end. |
36: | `start_single_block` / `end_single_block` | INT | Inclusive Flux single-stream block range. |
37: | `reference_latents_method` | COMBO | Method passed to Flux for reference latent token positions: `index`, `offset`, `uxo`, `index_timestep_zero`. Default: `index`. |
38: | `qk_adain_strength` | FLOAT | Optional StyleAligned-style AdaIN on target image Q/K token statistics. Default `0` preserves native Flux.2 behavior. |
39: | `verbose` | BOOLEAN | Print per-call patch diagnostics. |
40: 
41: Output:
42: 
43: | Output | Type |
44: | --- | --- |
45: | `model` | MODEL |
46: 
47: ## Suggested starting settings for Flux.2 / Flux.2 Klein
48: 
49: Start conservative:
50: 
51: ```text
52: high_scale_start = 0.25
53: high_scale_end   = 0.75
54: low_scale_start  = 1.00
55: low_scale_end    = 1.40
56: beta             = 2.0
57: start_percent    = 0.0
58: end_percent      = 1.0
59: start_single_block = 0
60: end_single_block   = 999
61: reference_latents_method = index
62: qk_adain_strength = 0.0
63: ```
64: 
65: Tuning guidance:
66: 
67: - Too much reference copying: lower `high_scale_start` and/or `high_scale_end`.
68: - Too weak style transfer: raise `low_scale_end` first.
69: - Late-stage texture too weak: raise `high_scale_end` slightly, not `high_scale_start`.
70: - Structural leakage or pose copying: keep `reference_latents_method=index`; avoid spatially shifted methods first.
71: - Washed or unstable attention: keep `qk_adain_strength=0.0`; only try AdaIN after key scaling is validated.
72: 
73: ## Installation
74: 
75: Clone or copy this repo into `ComfyUI/custom_nodes`:
76: 
77: ```bash
78: cd ComfyUI/custom_nodes
79: git clone https://github.com/xmarre/ComfyUI-Flux.2-Untwisting-RoPE
80: ```
81: 
82: There are no extra Python package dependencies beyond ComfyUI and PyTorch.
83: 
84: Restart ComfyUI.
85: 
86: ## Basic workflow placement
87: 
88: Typical Flux.2 style-reference graph:
89: 
90: 1. Load Flux.2 / Flux.2 Klein model normally.
91: 2. Encode your reference image with the Flux.2 VAE.
92: 3. Connect the encoded reference LATENT to `Flux.2 Untwist RoPE.reference_latent`.
93: 4. Put `Flux.2 Untwist RoPE` between the model loader and the guider/sampler.
94: 5. Use the patched model for sampling.
95: 
96: The node passes the reference latent through Flux’s native `ref_latents` path, then modifies single-stream attention via ComfyUI’s transformer patch mechanism.
97: 
98: ## Design notes
99: 
100: This repo intentionally does not copy the unofficial Z-Image implementation structure. The source repo is useful as a proof that frequency-aware reference-key scaling can be exposed in ComfyUI, but its architecture is not the right target for Flux.2:
101: 
102: - it patches Z-Image/NextDiT internals directly;
103: - it assumes `patchify_and_embed`, `layers.0..29.attention`, and Z-Image-specific masks;
104: - it includes a large RF inversion subsystem that is not necessary for Flux.2’s native reference-latent path;
105: - it stores most behavior in one large `__init__.py`, making regressions difficult to isolate.
106: 
107: This implementation instead uses Flux’s existing reference latent handling and ComfyUI’s attention patch API.
108: 
109: ## Validation plan
110: 
111: Minimum validation before publishing:
112: 
113: 1. Import smoke test: ComfyUI starts and loads `Flux.2 Untwist RoPE` without custom-node import errors.
114: 2. Baseline parity: with no `reference_latent`, output should match the unpatched model except for negligible wrapper overhead.
115: 3. No-op patch check: with a reference connected but `start_percent/end_percent` excluding all steps, sampling should behave like no reference from this node.
116: 4. Token range check: enable `verbose` and confirm reference token counts are detected during single-stream blocks.
117: 5. A/B visual tests at fixed seed:
118:    - no reference;
119:    - native Flux.2 reference workflow;
120:    - this node at defaults;
121:    - lower `high_scale_end` for copying-heavy references;
122:    - higher `low_scale_end` for weak style transfer.
123: 6. Resolution sweep: test at the target Flux.2 resolutions you actually use, especially 2–4 MP, because reference token count and attention cost scale with resolution.
124: 
125: ## Known risks
126: 
127: - Flux.2 already has native reference conditioning. This node changes how reference tokens influence attention; it may improve style transfer in some cases and harm edit/identity fidelity in others.
128: - The implementation assumes current ComfyUI Flux blocks expose `attn1_patch`, `img_slice`, `reference_image_num_tokens`, and `block_type="single"`. If ComfyUI changes these hook contracts, the node will safely degrade to a no-op or need a small compatibility patch.
129: - Appending reference latents increases attention context length and VRAM use. High-resolution references can be expensive.
130: - `qk_adain_strength` is optional and disabled by default because it is more invasive than reference-key frequency scaling.
131: 
132: ## License
133: 
134: MIT. The paper and any model weights remain under their own licenses.

(End of file - total 134 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/nodes.py"
}
```

**Output:**
```
<path>/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/nodes.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: from typing import Any, Dict, List, Optional, Tuple
4: 
5: import torch
6: 
7: try:
8:     from .flux_untwist.config import (
9:         _PREFIX,
10:         UntwistConfig,
11:         clamp_float,
12:         clamp_int,
13:         coerce_bool,
14:         normalize_percent_window,
15:         normalize_reference_method,
16:         safe_axes_dim,
17:         schedule_fraction,
18:     )
19:     from .flux_untwist.patches import flux_untwist_attn1_patch
20:     from .flux_untwist.utils import (
21:         append_transformer_patch,
22:         clone_model_options,
23:         normalize_ref_latents,
24:         process_latent_for_model,
25:         progress_from_timestep,
26:         repeat_to_batch,
27:         safe_get_diffusion_model,
28:     )
29: except ImportError:
30:     from flux_untwist.config import (
31:         _PREFIX,
32:         UntwistConfig,
33:         clamp_float,
34:         clamp_int,
35:         coerce_bool,
36:         normalize_percent_window,
37:         normalize_reference_method,
38:         safe_axes_dim,
39:         schedule_fraction,
40:     )
41:     from flux_untwist.patches import flux_untwist_attn1_patch
42:     from flux_untwist.utils import (
43:         append_transformer_patch,
44:         clone_model_options,
45:         normalize_ref_latents,
46:         process_latent_for_model,
47:         progress_from_timestep,
48:         repeat_to_batch,
49:         safe_get_diffusion_model,
50:     )
51: 
52: 
53: class Flux2UntwistRoPE:
54:     """Patch FLUX/FLUX.2 reference attention with frequency-aware RoPE key scaling."""
55: 
56:     CATEGORY = "model_patches/Flux2 Untwisting RoPE"
57:     RETURN_TYPES = ("MODEL",)
58:     RETURN_NAMES = ("model",)
59:     FUNCTION = "patch"
60:     DESCRIPTION = (
61:         "Adds reference latents to a Flux/Flux.2 model and scales only reference image keys "
62:         "in single-stream attention according to RoPE frequency bands."
63:     )
64: 
65:     @classmethod
66:     def INPUT_TYPES(cls):
67:         return {
68:             "required": {
69:                 "model": ("MODEL",),
70:                 "high_scale_start": (
71:                     "FLOAT",
72:                     {
73:                         "default": 0.25,
74:                         "min": -4.0,
75:                         "max": 8.0,
76:                         "step": 0.01,
77:                         "tooltip": "Scale for highest-frequency reference-key RoPE channels at the start of the active window. <1 reduces positional copying.",
78:                     },
79:                 ),
80:                 "high_scale_end": (
81:                     "FLOAT",
82:                     {
83:                         "default": 0.75,
84:                         "min": -4.0,
85:                         "max": 8.0,
86:                         "step": 0.01,
87:                         "tooltip": "Scale for highest-frequency reference-key RoPE channels at the end of the active window.",
88:                     },
89:                 ),
90:                 "low_scale_start": (
91:                     "FLOAT",
92:                     {
93:                         "default": 1.00,
94:                         "min": -4.0,
95:                         "max": 8.0,
96:                         "step": 0.01,
97:                         "tooltip": "Scale for lowest-frequency reference-key RoPE channels at the start of the active window.",
98:                     },
99:                 ),
100:                 "low_scale_end": (
101:                     "FLOAT",
102:                     {
103:                         "default": 1.40,
104:                         "min": -4.0,
105:                         "max": 8.0,
106:                         "step": 0.01,
107:                         "tooltip": "Scale for lowest-frequency reference-key RoPE channels at the end of the active window. >1 increases global style/reference pull.",
108:                     },
109:                 ),
110:                 "beta": (
111:                     "FLOAT",
112:                     {
113:                         "default": 2.0,
114:                         "min": 0.01,
115:                         "max": 32.0,
116:                         "step": 0.01,
117:                         "tooltip": "Polynomial interpolation exponent from high-frequency to low-frequency scales. Paper default is 2.",
118:                     },
119:                 ),
120:                 "start_percent": (
121:                     "FLOAT",
122:                     {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
123:                 ),
124:                 "end_percent": (
125:                     "FLOAT",
126:                     {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
127:                 ),
128:                 "start_single_block": (
129:                     "INT",
130:                     {"default": 0, "min": 0, "max": 999, "step": 1},
131:                 ),
132:                 "end_single_block": (
133:                     "INT",
134:                     {"default": 999, "min": 0, "max": 999, "step": 1},
135:                 ),
136:                 "reference_latents_method": (
137:                     ["index", "offset", "uxo", "index_timestep_zero"],
138:                     {
139:                         "default": "index",
140:                         "tooltip": "How Flux positions appended reference latent tokens. index keeps the spatial grid aligned and differs only by image index; offset/uxo use spatial offsets.",
141:                     },
142:                 ),
143:                 "qk_adain_strength": (
144:                     "FLOAT",
145:                     {
146:                         "default": 0.0,
147:                         "min": 0.0,
148:                         "max": 1.0,
149:                         "step": 0.01,
150:                         "tooltip": "Optional StyleAligned-style AdaIN on target image Q/K token statistics. 0 preserves native Flux.2 behavior.",
151:                     },
152:                 ),
153:                 "verbose": ("BOOLEAN", {"default": False}),
154:             },
155:             "optional": {
156:                 "reference_latent": (
157:                     "LATENT",
158:                     {
159:                         "tooltip": "Encoded reference image latent. You may leave this unconnected if your conditioning already supplies ref_latents.",
160:                     },
161:                 ),
162:             },
163:         }
164: 
165:     def patch(
166:         self,
167:         model: Any,
168:         high_scale_start: float,
169:         high_scale_end: float,
170:         low_scale_start: float,
171:         low_scale_end: float,
172:         beta: float,
173:         start_percent: float,
174:         end_percent: float,
175:         start_single_block: int,
176:         end_single_block: int,
177:         reference_latents_method: str,
178:         qk_adain_strength: float,
179:         verbose: bool = False,
180:         reference_latent: Optional[Dict[str, Any]] = None,
181:     ):
182:         node_verbose = coerce_bool(verbose)
183:         start_percent, end_percent = normalize_percent_window(start_percent, end_percent)
184:         start_single_block = clamp_int(start_single_block, 0, 999, 0)
185:         end_single_block = clamp_int(end_single_block, 0, 999, 999)
186:         if end_single_block < start_single_block:
187:             start_single_block, end_single_block = end_single_block, start_single_block
188: 
189:         ref_samples_cpu: Optional[torch.Tensor] = None
190:         if isinstance(reference_latent, dict) and torch.is_tensor(reference_latent.get("samples")):
191:             samples = reference_latent["samples"].detach()
192:             if samples.ndim != 4:
193:                 raise RuntimeError(f"reference_latent['samples'] must be BCHW; got shape {tuple(samples.shape)}")
194:             ref_samples_cpu = process_latent_for_model(model, samples).detach().to(device="cpu").clone()
195:         elif reference_latent is not None:
196:             raise RuntimeError("reference_latent must be a ComfyUI LATENT dict with a tensor 'samples' entry.")
197: 
198:         model_clone = model.clone()
199:         dm = safe_get_diffusion_model(model_clone)
200:         axes_dim = safe_axes_dim(dm)
201:         if not axes_dim:
202:             # Leave the patch usable on forks that keep Flux shape but hide params.axes_dim.
203:             axes_dim = tuple()
204: 
205:         model_clone.model_options = clone_model_options(getattr(model_clone, "model_options", {}) or {})
206:         old_wrapper = model_clone.model_options.get("model_function_wrapper", None)
207:         append_transformer_patch(model_clone, "attn1_patch", flux_untwist_attn1_patch)
208:         method = normalize_reference_method(reference_latents_method)
209: 
210:         # Values are copied into transformer_options each call; the attention patch reads them there.
211:         base_high_start = clamp_float(high_scale_start, -4.0, 8.0, 0.25)
212:         base_high_end = clamp_float(high_scale_end, -4.0, 8.0, 0.75)
213:         base_low_start = clamp_float(low_scale_start, -4.0, 8.0, 1.0)
214:         base_low_end = clamp_float(low_scale_end, -4.0, 8.0, 1.4)
215:         base_beta = clamp_float(beta, 0.01, 32.0, 2.0)
216:         base_adain = clamp_float(qk_adain_strength, 0.0, 1.0, 0.0)
217: 
218:         def model_function_wrapper(apply_model, args: Dict[str, Any]):
219:             input_x = args["input"]
220:             timestep = args["timestep"]
221:             c = dict(args["c"])
222:             cond_or_uncond = args.get("cond_or_uncond", None)
223: 
224:             progress = progress_from_timestep(timestep)
225:             active, _t = schedule_fraction(progress, start_percent, end_percent)
226: 
227:             to = dict(c.get("transformer_options", {}) or {})
228:             existing_refs = normalize_ref_latents(c.get("ref_latents", None))
229:             refs: List[torch.Tensor] = []
230: 
231:             if ref_samples_cpu is not None and active:
232:                 ref = ref_samples_cpu.to(device=input_x.device, dtype=input_x.dtype)
233:                 ref = repeat_to_batch(ref, int(input_x.shape[0]))
234:                 refs.append(ref)
235: 
236:             # Preserve explicit reference latents already present in conditioning/model kwargs.
237:             if existing_refs:
238:                 for ref in existing_refs:
239:                     refs.append(repeat_to_batch(ref.to(device=input_x.device, dtype=input_x.dtype), int(input_x.shape[0])))
240: 
241:             cfg = UntwistConfig(
242:                 enabled=bool(active and refs),
243:                 axes_dim=axes_dim,
244:                 high_scale_start=base_high_start,
245:                 high_scale_end=base_high_end,
246:                 low_scale_start=base_low_start,
247:                 low_scale_end=base_low_end,
248:                 beta=base_beta,
249:                 start_percent=start_percent,
250:                 end_percent=end_percent,
251:                 start_single_block=start_single_block,
252:                 end_single_block=end_single_block,
253:                 qk_adain_strength=base_adain,
254:                 reference_latents_method=method,
255:                 progress=progress,
256:                 verbose=node_verbose,
257:             )
258:             to["flux_untwist_rope"] = cfg.as_transformer_options()
259:             c["transformer_options"] = to
260: 
261:             if refs:
262:                 c["ref_latents"] = refs
263:                 # Do not override a method already set directly by upstream conditioning unless
264:                 # this node supplies the reference. That keeps native workflows composable.
265:                 if ref_samples_cpu is not None or "ref_latents_method" not in c:
266:                     c["ref_latents_method"] = method
267: 
268:             if node_verbose:
269:                 ref_shapes = [tuple(r.shape) for r in refs]
270:                 print(
271:                     f"{_PREFIX} call: progress={progress:.3f} active={active} refs={len(refs)} "
272:                     f"method={c.get('ref_latents_method', '<native>')} shapes={ref_shapes}"
273:                 )
274: 
275:             if old_wrapper is not None:
276:                 return old_wrapper(
277:                     apply_model,
278:                     {
279:                         "input": input_x,
280:                         "timestep": timestep,
281:                         "c": c,
282:                         "cond_or_uncond": cond_or_uncond,
283:                     },
284:                 )
285:             return apply_model(input_x, timestep, **c)
286: 
287:         model_clone.set_model_unet_function_wrapper(model_function_wrapper)
288: 
289:         if node_verbose:
290:             print(f"{_PREFIX} patched Flux-like model: {type(dm).__name__}")
291:             print(f"{_PREFIX} axes_dim={axes_dim or '<unknown>'} blocks={start_single_block}..{end_single_block}")
292:             print(f"{_PREFIX} high={base_high_start:.3f}->{base_high_end:.3f} low={base_low_start:.3f}->{base_low_end:.3f} beta={base_beta:.3f}")
293: 
294:         return (model_clone,)
295: 
296: 
297: NODE_CLASS_MAPPINGS = {
298:     "Flux2UntwistRoPE": Flux2UntwistRoPE,
299: }
300: 
301: NODE_DISPLAY_NAME_MAPPINGS = {
302:     "Flux2UntwistRoPE": "Flux.2 Untwist RoPE",
303: }

(End of file - total 303 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/__init__.py"
}
```

**Output:**
```
<path>/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/__init__.py</path>
<type>file</type>
<content>
1: try:
2:     from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
3: except ImportError:
4:     from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
5: 
6: __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

(End of file - total 6 lines)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 33.9s)

_Thinking:_

Now let me read the core implementation files to understand the full logic.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/flux_untwist"
}
```

**Output:**
```
<path>/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/flux_untwist</path>
<type>directory</type>
<entries>
__init__.py
__pycache__/
config.py
patches.py
utils.py

(5 entries)
</entries>
```

---

## Assistant (Build · MiMo V2.5 Free · 36.9s)

_Thinking:_

Let me read the core files in parallel to understand the implementation fully.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/flux_untwist/config.py"
}
```

**Output:**
```
<path>/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/flux_untwist/config.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: from dataclasses import dataclass
4: from typing import Any, Dict, Iterable, List, Tuple
5: 
6: 
7: _PREFIX = "[Flux2UntwistRoPE]"
8: 
9: 
10: @dataclass(frozen=True)
11: class UntwistConfig:
12:     """Runtime configuration copied into ComfyUI transformer_options per model call."""
13: 
14:     enabled: bool
15:     axes_dim: Tuple[int, ...]
16:     high_scale_start: float
17:     high_scale_end: float
18:     low_scale_start: float
19:     low_scale_end: float
20:     beta: float
21:     start_percent: float
22:     end_percent: float
23:     start_single_block: int
24:     end_single_block: int
25:     qk_adain_strength: float
26:     reference_latents_method: str
27:     progress: float
28:     verbose: bool
29: 
30:     def as_transformer_options(self) -> Dict[str, Any]:
31:         return {
32:             "enabled": bool(self.enabled),
33:             "axes_dim": list(self.axes_dim),
34:             "high_scale_start": float(self.high_scale_start),
35:             "high_scale_end": float(self.high_scale_end),
36:             "low_scale_start": float(self.low_scale_start),
37:             "low_scale_end": float(self.low_scale_end),
38:             "beta": float(self.beta),
39:             "start_percent": float(self.start_percent),
40:             "end_percent": float(self.end_percent),
41:             "start_single_block": int(self.start_single_block),
42:             "end_single_block": int(self.end_single_block),
43:             "qk_adain_strength": float(self.qk_adain_strength),
44:             "reference_latents_method": str(self.reference_latents_method),
45:             "progress": float(self.progress),
46:             "verbose": bool(self.verbose),
47:         }
48: 
49: 
50: def clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
51:     try:
52:         out = float(value)
53:     except Exception:
54:         out = float(default)
55:     if out != out or out in (float("inf"), float("-inf")):
56:         out = float(default)
57:     return max(float(lo), min(float(hi), out))
58: 
59: 
60: def clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
61:     try:
62:         out = int(value)
63:     except Exception:
64:         out = int(default)
65:     return max(int(lo), min(int(hi), out))
66: 
67: 
68: def coerce_bool(value: Any) -> bool:
69:     if isinstance(value, str):
70:         return value.strip().lower() in {"1", "true", "yes", "on", "y", "t"}
71:     return bool(value)
72: 
73: 
74: def normalize_percent_window(start: Any, end: Any) -> Tuple[float, float]:
75:     s = clamp_float(start, 0.0, 1.0, 0.0)
76:     e = clamp_float(end, 0.0, 1.0, 1.0)
77:     if e < s:
78:         s, e = e, s
79:     if abs(e - s) < 1e-6:
80:         e = min(1.0, s + 1e-6)
81:     return s, e
82: 
83: 
84: def schedule_fraction(progress: float, start_percent: float, end_percent: float) -> Tuple[bool, float]:
85:     """Return (active, normalized_t) for a denoising progress in [0, 1]."""
86:     p = clamp_float(progress, 0.0, 1.0, 0.0)
87:     s, e = normalize_percent_window(start_percent, end_percent)
88:     if p < s or p > e:
89:         return False, 0.0 if p < s else 1.0
90:     return True, max(0.0, min(1.0, (p - s) / max(e - s, 1e-6)))
91: 
92: 
93: def normalize_reference_method(method: Any) -> str:
94:     text = str(method or "index").strip().lower()
95:     if text in {"uxo/uno", "uno", "uso"}:
96:         return "uxo"
97:     if text in {"index", "offset", "uxo", "index_timestep_zero"}:
98:         return text
99:     return "index"
100: 
101: 
102: def safe_axes_dim(params_or_model: Any) -> Tuple[int, ...]:
103:     """Extract FLUX axes_dim from a diffusion model or its params object."""
104:     obj = getattr(params_or_model, "params", params_or_model)
105:     axes = getattr(obj, "axes_dim", None)
106:     if axes is None:
107:         return tuple()
108:     try:
109:         return tuple(int(x) for x in axes)
110:     except Exception:
111:         return tuple()

(End of file - total 111 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/flux_untwist/patches.py"
}
```

**Output:**
```
<path>/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/flux_untwist/patches.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
4: 
5: import torch
6: 
7: from .config import _PREFIX, clamp_float, schedule_fraction
8: 
9: 
10: def lerp(a: float, b: float, t: float) -> float:
11:     return float(a) + (float(b) - float(a)) * float(t)
12: 
13: 
14: def build_frequency_scale_vector(
15:     head_dim: int,
16:     axes_dim: Sequence[int],
17:     high_scale: float,
18:     low_scale: float,
19:     beta: float,
20:     device: torch.device,
21:     dtype: torch.dtype,
22: ) -> torch.Tensor:
23:     """
24:     Build a per-channel scale vector for RoPE pair chunks.
25: 
26:     The vector is applied before ComfyUI applies RoPE. Since each 2D RoPE pair receives
27:     a scalar shared by both entries, that scaling commutes with the 2D rotation and is
28:     equivalent to scaling the already-rotated reference keys.
29:     """
30:     head_dim = int(head_dim)
31:     axes = [int(x) for x in axes_dim if int(x) > 0]
32:     if not axes or sum(axes) != head_dim:
33:         axes = [head_dim]
34: 
35:     beta = max(0.01, float(beta))
36:     pieces: List[torch.Tensor] = []
37:     has_three_axes = len(axes) == 3
38: 
39:     for axis_index, axis_dim in enumerate(axes):
40:         n_pairs = axis_dim // 2
41:         if n_pairs <= 0:
42:             pieces.append(torch.ones(axis_dim, device=device, dtype=dtype))
43:             continue
44: 
45:         # FLUX-style axes are normally [image/ref index, y, x]. The paper treats
46:         # the non-spatial partition as not responsible for spatial copying, so keep
47:         # it on the low-frequency/reference-preserving side instead of suppressing it.
48:         if has_three_axes and axis_index == 0:
49:             pair_scales = torch.full((n_pairs,), float(low_scale), device=device, dtype=torch.float32)
50:         else:
51:             if n_pairs == 1:
52:                 d = torch.zeros((1,), device=device, dtype=torch.float32)
53:             else:
54:                 d = torch.linspace(0.0, 1.0, n_pairs, device=device, dtype=torch.float32)
55:             pair_scales = float(high_scale) + (float(low_scale) - float(high_scale)) * d.pow(beta)
56: 
57:         pieces.append(pair_scales.to(dtype=dtype).repeat_interleave(2))
58:         if axis_dim % 2:
59:             pieces.append(torch.ones(1, device=device, dtype=dtype))
60: 
61:     out = torch.cat(pieces, dim=0) if pieces else torch.ones(head_dim, device=device, dtype=dtype)
62:     if out.numel() < head_dim:
63:         out = torch.nn.functional.pad(out, (0, head_dim - out.numel()), value=1.0)
64:     return out[:head_dim]
65: 
66: 
67: def _reference_ranges_from_options(seq_len: int, extra_options: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
68:     img_slice = extra_options.get("img_slice", None)
69:     ref_counts = extra_options.get("reference_image_num_tokens", None)
70:     if not isinstance(img_slice, (list, tuple)) or len(img_slice) != 2:
71:         return None, []
72: 
73:     try:
74:         img_start = max(0, min(int(img_slice[0]), int(seq_len)))
75:         img_end = max(img_start, min(int(img_slice[1]), int(seq_len)))
76:     except Exception:
77:         return None, []
78: 
79:     if ref_counts is None:
80:         return (img_start, img_end), []
81:     if isinstance(ref_counts, int):
82:         counts = [int(ref_counts)]
83:     elif isinstance(ref_counts, (list, tuple)):
84:         counts = []
85:         for value in ref_counts:
86:             try:
87:                 count = int(value)
88:             except Exception:
89:                 continue
90:             if count > 0:
91:                 counts.append(count)
92:     else:
93:         return (img_start, img_end), []
94: 
95:     total_ref = sum(counts)
96:     total_img = img_end - img_start
97:     if total_ref <= 0 or total_ref > total_img:
98:         return (img_start, img_end), []
99: 
100:     ref_start = img_end - total_ref
101:     ranges: List[Tuple[int, int]] = []
102:     cur = ref_start
103:     for count in counts:
104:         end = min(img_end, cur + int(count))
105:         if end > cur:
106:             ranges.append((cur, end))
107:         cur = end
108:     return (img_start, ref_start), ranges
109: 
110: 
111: def _adain_tokens(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
112:     """AdaIN over the token dimension for tensors shaped [B, H, L, D]."""
113:     if target.numel() == 0 or style.numel() == 0:
114:         return target
115:     t_float = target.float()
116:     s_float = style.float()
117:     t_mean = t_float.mean(dim=2, keepdim=True)
118:     s_mean = s_float.mean(dim=2, keepdim=True)
119:     t_std = t_float.var(dim=2, keepdim=True, unbiased=False).add(eps).sqrt()
120:     s_std = s_float.var(dim=2, keepdim=True, unbiased=False).add(eps).sqrt()
121:     return ((t_float - t_mean) / t_std * s_std + s_mean).to(dtype=target.dtype)
122: 
123: 
124: def _concat_ranges(x: torch.Tensor, ranges: Sequence[Tuple[int, int]]) -> torch.Tensor:
125:     parts = []
126:     length = int(x.shape[2])
127:     for start, end in ranges:
128:         s = max(0, min(int(start), length))
129:         e = max(s, min(int(end), length))
130:         if e > s:
131:             parts.append(x[:, :, s:e, :])
132:     if not parts:
133:         return x[:, :, 0:0, :]
134:     return torch.cat(parts, dim=2)
135: 
136: 
137: def flux_untwist_attn1_patch(
138:     q: torch.Tensor,
139:     k: torch.Tensor,
140:     v: torch.Tensor,
141:     pe: Optional[torch.Tensor] = None,
142:     attn_mask: Optional[torch.Tensor] = None,
143:     extra_options: Optional[Dict[str, Any]] = None,
144: ) -> Dict[str, Any]:
145:     """ComfyUI `attn1_patch` hook for FLUX single-stream blocks."""
146:     if extra_options is None:
147:         return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
148: 
149:     cfg = extra_options.get("flux_untwist_rope", None)
150:     if not isinstance(cfg, dict) or not cfg.get("enabled", False):
151:         return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
152: 
153:     # The paper applies reference sharing only in FLUX single-stream blocks.
154:     if str(extra_options.get("block_type", "")) != "single":
155:         return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
156: 
157:     block_index = int(extra_options.get("block_index", -1))
158:     if block_index < int(cfg.get("start_single_block", 0)) or block_index > int(cfg.get("end_single_block", 999)):
159:         return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
160: 
161:     seq_len = int(k.shape[2])
162:     target_range, ref_ranges = _reference_ranges_from_options(seq_len, extra_options)
163:     if not ref_ranges:
164:         return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
165: 
166:     active, t = schedule_fraction(
167:         float(cfg.get("progress", 0.0)),
168:         float(cfg.get("start_percent", 0.0)),
169:         float(cfg.get("end_percent", 1.0)),
170:     )
171:     if not active:
172:         return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
173: 
174:     high_scale = lerp(cfg.get("high_scale_start", 0.25), cfg.get("high_scale_end", 0.75), t)
175:     low_scale = lerp(cfg.get("low_scale_start", 1.0), cfg.get("low_scale_end", 1.4), t)
176:     beta = clamp_float(cfg.get("beta", 2.0), 0.01, 32.0, 2.0)
177: 
178:     scale_vec = build_frequency_scale_vector(
179:         head_dim=int(k.shape[-1]),
180:         axes_dim=cfg.get("axes_dim", []) or [],
181:         high_scale=high_scale,
182:         low_scale=low_scale,
183:         beta=beta,
184:         device=k.device,
185:         dtype=k.dtype,
186:     ).view(1, 1, 1, int(k.shape[-1]))
187: 
188:     q_out = q
189:     k_out = k.clone()
190: 
191:     for start, end in ref_ranges:
192:         k_out[:, :, start:end, :] = k_out[:, :, start:end, :] * scale_vec
193: 
194:     adain_strength = clamp_float(cfg.get("qk_adain_strength", 0.0), 0.0, 1.0, 0.0)
195:     if adain_strength > 0.0 and target_range is not None:
196:         target_start, target_end = target_range
197:         if target_end > target_start:
198:             ref_q = _concat_ranges(q, ref_ranges)
199:             ref_k = _concat_ranges(k, ref_ranges)
200:             if ref_q.shape[2] > 0 and ref_k.shape[2] > 0:
201:                 q_out = q.clone()
202:                 q_adain = _adain_tokens(q_out[:, :, target_start:target_end, :], ref_q)
203:                 k_adain = _adain_tokens(k_out[:, :, target_start:target_end, :], ref_k)
204:                 q_out[:, :, target_start:target_end, :] = (
205:                     q_out[:, :, target_start:target_end, :] * (1.0 - adain_strength)
206:                     + q_adain * adain_strength
207:                 )
208:                 k_out[:, :, target_start:target_end, :] = (
209:                     k_out[:, :, target_start:target_end, :] * (1.0 - adain_strength)
210:                     + k_adain * adain_strength
211:                 )
212: 
213:     return {"q": q_out, "k": k_out, "v": v, "pe": pe, "attn_mask": attn_mask}

(End of file - total 213 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/flux_untwist/utils.py"
}
```

**Output:**
```
<path>/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/flux_untwist/utils.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: import copy
4: import math
5: from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
6: 
7: import torch
8: 
9: 
10: def repeat_to_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
11:     """Repeat a tensor along batch dim using ComfyUI's helper when available."""
12:     if int(x.shape[0]) == int(batch):
13:         return x
14:     try:
15:         import comfy.utils  # type: ignore
16: 
17:         return comfy.utils.repeat_to_batch_size(x, int(batch))
18:     except Exception:
19:         reps = math.ceil(int(batch) / max(1, int(x.shape[0])))
20:         return x.repeat((reps,) + (1,) * (x.ndim - 1))[: int(batch)]
21: 
22: 
23: def clone_model_options(options: Dict[str, Any]) -> Dict[str, Any]:
24:     """Copy only the mutable parts we mutate: transformer_options.patch lists."""
25:     out = dict(options or {})
26:     transformer_options = dict(out.get("transformer_options", {}) or {})
27:     patches = transformer_options.get("patches", {}) or {}
28:     patches_copy: Dict[str, List[Any]] = {}
29:     for key, value in patches.items():
30:         patches_copy[key] = list(value) if isinstance(value, list) else [value]
31:     transformer_options["patches"] = patches_copy
32:     out["transformer_options"] = transformer_options
33:     return out
34: 
35: 
36: def append_transformer_patch(model: Any, patch_name: str, patch_fn: Any) -> None:
37:     """Append a ComfyUI transformer patch without relying on optional ModelPatcher helpers."""
38:     model.model_options = clone_model_options(getattr(model, "model_options", {}) or {})
39:     transformer_options = model.model_options.setdefault("transformer_options", {})
40:     patches = transformer_options.setdefault("patches", {})
41:     patches.setdefault(patch_name, []).append(patch_fn)
42: 
43: 
44: def sigma_from_timestep(timestep: Any) -> float:
45:     """Normalize Comfy/Flux timestep-like values into a sigma-like [0, 1] value."""
46:     try:
47:         if torch.is_tensor(timestep):
48:             value = float(timestep.detach().float().mean().item())
49:         else:
50:             value = float(timestep)
51:         if not math.isfinite(value):
52:             return 1.0
53:         if 0.0 <= value <= 1.0:
54:             return max(0.0, min(1.0, value))
55:         if 1.0 < value <= 1000.0:
56:             return max(0.0, min(1.0, value / 1000.0))
57:     except Exception:
58:         pass
59:     return 1.0
60: 
61: 
62: def progress_from_timestep(timestep: Any) -> float:
63:     return max(0.0, min(1.0, 1.0 - sigma_from_timestep(timestep)))
64: 
65: 
66: def safe_get_diffusion_model(model_patcher: Any) -> Any:
67:     """Find the wrapped Flux diffusion model in common ComfyUI ModelPatcher layouts."""
68:     roots: List[Any] = []
69:     if hasattr(model_patcher, "model"):
70:         roots.append(model_patcher.model)
71:     roots.append(model_patcher)
72: 
73:     attr_paths = (
74:         "diffusion_model",
75:         "model.diffusion_model",
76:         "model.model.diffusion_model",
77:         "inner_model.diffusion_model",
78:         "model.inner_model.diffusion_model",
79:     )
80:     for root in roots:
81:         for path in attr_paths:
82:             obj = root
83:             ok = True
84:             for part in path.split("."):
85:                 if not hasattr(obj, part):
86:                     ok = False
87:                     break
88:                 obj = getattr(obj, part)
89:             if ok and _looks_like_flux(obj):
90:                 return obj
91: 
92:     seen = set()
93:     stack = list(roots)
94:     while stack and len(seen) < 512:
95:         obj = stack.pop()
96:         if id(obj) in seen:
97:             continue
98:         seen.add(id(obj))
99:         if _looks_like_flux(obj):
100:             return obj
101:         for name in ("model", "inner_model", "diffusion_model", "unet", "wrapped"):
102:             if hasattr(obj, name):
103:                 try:
104:                     stack.append(getattr(obj, name))
105:                 except Exception:
106:                     pass
107:     raise RuntimeError("Could not locate a Flux-like diffusion model on the supplied MODEL.")
108: 
109: 
110: def _looks_like_flux(obj: Any) -> bool:
111:     return (
112:         hasattr(obj, "process_img")
113:         and hasattr(obj, "single_blocks")
114:         and hasattr(obj, "double_blocks")
115:         and hasattr(obj, "params")
116:     )
117: 
118: 
119: def process_latent_for_model(model_patcher: Any, latent_samples: torch.Tensor) -> torch.Tensor:
120:     """Convert a ComfyUI LATENT tensor to model input latent space when available."""
121:     processor = None
122:     try:
123:         processor = getattr(getattr(model_patcher, "model", None), "process_latent_in", None)
124:     except Exception:
125:         processor = None
126:     if callable(processor):
127:         return processor(latent_samples)
128:     return latent_samples
129: 
130: 
131: def normalize_ref_latents(existing: Any) -> List[torch.Tensor]:
132:     if existing is None:
133:         return []
134:     if torch.is_tensor(existing):
135:         return [existing]
136:     if isinstance(existing, (list, tuple)):
137:         return [x for x in existing if torch.is_tensor(x)]
138:     return []

(End of file - total 138 lines)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 14.7s)

_Thinking:_

Now I have a comprehensive understanding of the codebase. Let me also check the SOURCE_ANALYSIS.md for any additional context.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/SOURCE_ANALYSIS.md"
}
```

**Output:**
```
<path>/home/kuroko/ComfyUI/custom_nodes/ComfyUI-Flux.2-Untwisting-RoPE/SOURCE_ANALYSIS.md</path>
<type>file</type>
<content>
1: # Source analysis and design decisions
2: 
3: ## Paper behavior to preserve
4: 
5: The paper’s core mechanism is not “style transfer by any reference image.” It is a targeted attention intervention:
6: 
7: 1. RoPE decomposes into frequency components with different positional sensitivity.
8: 2. High-frequency RoPE components make shared attention strongly local and can cause reference copying.
9: 3. Low-frequency components preserve broader/global association.
10: 4. The intervention should modulate **reference image keys** in shared attention, not target/text keys and not values.
11: 5. The intervention is intended for Flux-style MMDiT single-stream blocks.
12: 6. The frequency scale should be smooth across the RoPE spectrum and schedulable over denoising.
13: 
14: Those points are preserved.
15: 
16: ## Unofficial source repo behavior
17: 
18: The attached unofficial implementation is a single-file ComfyUI custom node targeting Z-Image/NextDiT rather than Flux/Flux.2. Its real behavior is:
19: 
20: - `RFInversion` stores a reference latent and captures sampler sigmas through a sampler wrapper.
21: - `UntwistingRoPE` clones a model, finds a Z-Image/NextDiT-like diffusion model, monkey-patches `patchify_and_embed`, context-refiner mask modules, and `layers.N.attention` modules.
22: - The attention patch creates a doubled target/reference batch, runs target queries against target + reference keys/values, scales reference keys with a frequency vector, and discards the reference branch prediction.
23: - It has optional cross-batch AdaIN for Q/K and a large RF trajectory cache/inversion subsystem.
24: - It assumes Z-Image names and shapes: `patchify_and_embed`, `layers.0..29.attention`, `context_refiner`, `n_local_heads`, `n_local_kv_heads`, `head_dim`, and `optimized_attention_masked` usage.
25: 
26: ## Preserve
27: 
28: - Frequency-aware per-pair key scaling.
29: - Smooth polynomial high-to-low frequency interpolation.
30: - Denoising-progress schedule for high/low scales.
31: - Single-stream-only default behavior.
32: - Optional Q/K AdaIN as an exposed, disabled-by-default experiment.
33: - Conservative fallback behavior: if no reference token ranges are present, the attention patch returns unchanged tensors.
34: 
35: ## Rewrite
36: 
37: - The entire ComfyUI integration layer.
38: - Reference injection: use Flux’s native `ref_latents` path instead of constructing a doubled batch and discarding half the prediction.
39: - Attention integration: use ComfyUI’s `attn1_patch` hook instead of replacing module `forward` methods.
40: - Token range detection: use Flux-provided `img_slice` and `reference_image_num_tokens` from transformer options instead of Z-Image patchify metadata.
41: - Model detection: detect Flux-like models by `process_img`, `single_blocks`, `double_blocks`, and `params`, not Z-Image internals.
42: 
43: ## Fix
44: 
45: - Remove hard-coded `layers.0..29.attention` patching.
46: - Remove Z-Image-specific mask surgery.
47: - Avoid mutation of global/module state for sampler trajectory cache.
48: - Avoid in-place modification of original `model_options` patch lists.
49: - Avoid one giant `__init__.py`; split config, utilities, patches, and nodes.
50: - Use paper-consistent `beta=2.0` default instead of an extreme default.
51: - Default to non-spatial reference indexing (`index`) rather than shifted spatial placement.
52: 
53: ## Drop
54: 
55: - RF inversion subsystem. For Flux.2, ComfyUI already exposes native reference latent handling. RF inversion adds model calls, persistent state, cache invalidation hazards, and sampler-wrapper coupling that are not justified for the Flux.2 target path.
56: - Z-Image/NextDiT support. It is a different architecture target and would dilute the implementation.
57: - Context-refiner cap-mask patching. Flux.2 does not need the Z-Image mask workaround.
58: - Direct attention module monkey-patching. It is brittle against upstream ComfyUI changes.
59: - Example images and large workflow JSON copied from the source repo.
60: 
61: ## New architecture
62: 
63: ```text
64: ComfyUI-Flux.2-Untwisting-RoPE/
65:   __init__.py                  # ComfyUI node registration
66:   nodes.py                     # public node class
67:   flux_untwist/
68:     config.py                  # config normalization and scheduling
69:     patches.py                 # Flux attn1_patch implementation
70:     utils.py                   # model/latent/model_options helpers
71:   tests/
72:     test_frequency_scale.py    # pure unit tests for core math/ranges
73:   README.md
74:   SOURCE_ANALYSIS.md
75:   pyproject.toml
76:   LICENSE
77: ```
78: 
79: ## Main invariant
80: 
81: When active and reference tokens are present:
82: 
83: - only reference image key channels are frequency-scaled;
84: - target keys are unchanged except for optional AdaIN when explicitly enabled;
85: - text keys are unchanged;
86: - all values are unchanged;
87: - all queries are unchanged except for optional AdaIN when explicitly enabled;
88: - if reference ranges cannot be identified, the patch is a no-op.
89: 
90: ## Assumptions and deviations
91: 
92: - Assumes ComfyUI Flux/Flux.2 still provides `attn1_patch`, `img_slice`, `reference_image_num_tokens`, and `block_type` in `extra_options` / `transformer_options`.
93: - Assumes Flux.2 reference latents are accepted through the same `ref_latents` kwarg path used by current ComfyUI Flux code.
94: - Uses pre-RoPE key scaling because ComfyUI’s `attn1_patch` runs before `attention()` applies RoPE. This is equivalent for this operation because each 2D RoPE pair is multiplied by one scalar shared across the pair, and scalar multiplication commutes with the 2D rotation.
95: - Defaults are conservative for Flux.2 because Flux.2 already has trained/native reference behavior. Stronger settings may be useful but should be validated with fixed-seed A/B tests.
96: 
97: ## Regression risks
98: 
99: - Upstream ComfyUI may rename or remove the Flux transformer patch hooks.
100: - Some Flux.2 forks may store reference latents in conditioning under a different key than `ref_latents`; direct `reference_latent` input remains the primary supported path.
101: - Very large reference images increase sequence length and attention cost.
102: - `index_timestep_zero` is supported because Flux code paths exist for it, but it is not the default because it changes modulation semantics more than plain `index`.

(End of file - total 102 lines)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 38.2s)

_Thinking:_

Now I have a thorough understanding of the codebase. This is a ComfyUI custom node for Flux/Flux.2 models that implements the "Untwisting RoPE" paper's technique for reference-image attention modulation. The user wants me to create an instruction document for an AI agent to implement this same technique for the "Anima" model instead.

I need to understand what the "Anima" model is. Let me search for information about it.

Now I need to understand what the "Anima" model is to write accurate adaptation instructions. Let me search for it.

**Tool: websearch**

**Input:**
```json
{
  "query": "Anima model AI diffusion DiT architecture ComfyUI 2026",
  "numResults": 8
}
```

**Output:**
```
{
  "search_id": "search_79981208d43d4f50a0ed78b1aa8ebf21",
  "results": [
    {
      "url": "https://deepwiki.com/hustvl/DiG/2.1-dit-architecture",
      "title": "DiT Architecture | hustvl/DiG | DeepWiki",
      "publish_date": "2025-05-09",
      "excerpts": [
        "For information about the DiG model implementation, see DiG Model .\n\n## Architecture Overview\n\nDiT is a diffusion model that employs a Transformer backbone for image generation. It combines diffusion models with Vision Transformers to create a highly effective architecture for generating high-quality images. The architecture processes noisy images and corresponding timesteps to predict either the noise or the clean image.\n\nSources: [models\\_dit.py 152-265](https://github.com/hustvl/DiG/blob/e3ac0bae/models_dit.py)\n\n## Core Components\n\n### Patch Embedding\n\nThe DiT architecture begins by transforming the input image (or latent representation) into patch embeddings using a PatchEmbed layer. This process divides the image into fixed-size patches and projects each patch into the model's hidden dimension.\n\nSources: [models\\_dit.py 176-177](https://github.com/hustvl/DiG/blob/e3ac0bae/models_dit.py) [models\\_dit.py 207-209](https://github.com/hustvl/DiG/blob/e3ac0bae/models_dit.\npy) [models\\_dit.py 291-338](https://github.com/hustvl/DiG/blob/e3ac0bae/models_dit.py)\n\n### Conditioning Embeddings\n\nDiT uses two types of conditioning inputs that are embedded and combined:\n\n1. **Timestep Embedding** : Encodes the diffusion timestep using sinusoidal embeddings\n2. **Label Embedding** : Encodes class labels for conditional generation with support for classifier-free guidance\n\nSources: [models\\_dit.py 27-65](https://github.com/hustvl/DiG/blob/e3ac0bae/models_dit.py) [models\\_dit.py 68-95](https://github.com/hustvl/DiG/blob/e3ac0bae/models_dit.py)\n\n### DiT Block with adaLN-Zero\n\nThe core building block of DiT is the DiTBlock, which implements a transformer block with adaptive Layer Normalization Zero (adaLN-Zero) conditioning. This allows the diffusion timestep and class conditioning to modulate the behavior of each block.\n\nSources: [models\\_dit.py 102-123](https://github.com/hustvl/DiG/blob/e3ac0bae/models_dit.py)\n\n### Final Layer\nThe FinalLayer applies a final normalization and projection to transform the features into the output space. Like the DiT blocks, it uses adaptive layer normalization based on the conditioning.\n\nSources: [models\\_dit.py 126-149](https://github.com/hustvl/DiG/blob/e3ac0bae/models_dit.py)\n\n### Unpatchify Process\n\nAfter the final layer, the model applies an unpatchify operation to reconstruct the spatial dimensions of the output.\n\nSources: [models\\_dit.py 234-247](https://github.com/hustvl/DiG/blob/e3ac0bae/models_dit.py)\n\n## Model Variants and Configurations\n\nDiT comes in multiple variants that differ in model size and patch size. This flexibility allows for balancing computational requirements and generation quality.\n\n|Variant |Depth |Hidden Size |Num Heads |Parameters\\* |\n| --- | --- | --- | --- | --- |\n|DiT-XL |28 |1152 |16 |~1B |\n|DiT-L |24 |1024 |16 |~600M |\n|DiT-B |12 |768 |12 |~300M |\n|DiT-S |12 |384 |6 |~80M |\n\n\\*Approximate parameter counts"
      ]
    },
    {
      "url": "https://civarchive.com/models/2458426?modelVersionId=2836417",
      "title": "Anima<!-- --> - CivArchive (CivitAI Archive)",
      "publish_date": "2026-03-11",
      "excerpts": [
        "Anima is a 2 billion parameter text-to-image model created via a collaboration between CircleStone Labs and Comfy Org. It is focused mainly on anime concepts, characters, and styles, but is also capable of generating a wide variety of other non-photorealistic content. The model is designed for making illustrations and artistic images, and will not work well at realism.\n\nIt is trained on several million anime images and about 800k non-anime artistic images. No synthetic data was used for training. The knowledge cut-off date for the anime training data is September 2025.\n\n**NEW** : Try the [Turbo LoRA](https://civarchive.com/models/2560840/anima-turbo-lora) for better stability and much faster generations.\n\n# Versions\n\n* Anima-Base\n  \n    + The pretrained, unrefined base model. Maximum flexibility, diversity, and style adherence.\n* Anima-Turbo\n  \n    + Coming soon.\n\n# Installing and running\n\nGet the text encoder and VAE from the [HuggingFage page](https://huggingface.\nco/circlestone-labs/Anima) .\n\nThe model is natively supported in ComfyUI. The model files go in their respective folders inside your model directory:\n\n* anima-base-v1.0.safetensors goes in ComfyUI/models/diffusion\\_models\n* qwen\\_3\\_06b\\_base.safetensors goes in ComfyUI/models/text\\_encoders\n* qwen\\_image\\_vae.safetensors goes in ComfyUI/models/vae (this is the Qwen-Image VAE, you might already have it)\n\n## Generation settings\n\n* Works at resolutions between 512^2 and 1536^2 pixels.\n* 30-50 steps, CFG 4-6.\n* A variety of samplers work. Some of my favorites:\n  \n    + er\\_sde: neutral style, flat colors, sharp lines. I use this as a reasonable default.\n    + euler\\_a: Softer, thinner lines. Can sometimes tend towards a 2.5D look. CFG can be pushed a bit higher than other samplers without burning the image.\n    + dpmpp\\_2m\\_sde\\_gpu: similar in style to er\\_sde but can produce more variety and be more \"creative\". Depending on the prompt it can get too wild sometimes.\n* If going for a more realistic / painterly look, the beta57 scheduler (ComfyUI RES4LYF custom node pack) can help make better textures, since it puts more emphasis on low-noise timesteps.\n\n# Prompting\n\nThe model is trained on Danbooru-style tags, natural language captions, and combinations of tags and captions.\n\n* Use lowercase for tags, and spaces instead of underscores. Score tags are the only tags that use underscores.\n* Recommended positive prefix: \"masterpiece, best quality, score\\_7, safe, \"\n* Recommended negative: \"worst quality, low quality, score\\_1, score\\_2, score\\_3, artist name\"\n* When using a tag that is different between Danbooru and Gelbooru, prefer the Gelbooru version.\n* Prompt weighting works, but needs a weight higher than typically used for SDXL. Example: \"(chibi:2)\"\n\n## Tag order\n\n[quality/meta/year/safety tags] [1girl/1boy/1other etc] [character] [series] [artist] [general tags]\n\nWithin each tag section, the tags can be in arbitrary order.\n\n## Quality tags"
      ]
    },
    {
      "url": "https://civitai.com/models/2458426/anima",
      "title": "Anima - base-v1.0 | Anima Checkpoint | Civitai",
      "publish_date": "2026-03-11",
      "excerpts": [
        "It is focused mainly on anime concepts, characters, and styles, but is also capable of generating a wide variety of other non-photorealistic content. The model is designed for making illustrations and artistic images, and will not work well at realism.\n\nIt is trained on several million anime images and about 800k non-anime artistic images. No synthetic data was used for training. The knowledge cut-off date for the anime training data is September 2025.\n\n**NEW** : Try the [Turbo LoRA](https://civitai.com/models/2560840/anima-turbo-lora) for better stability and much faster generations.\n\n# Versions\n\n* Anima-Base\n  \n    + The pretrained, unrefined base model. Maximum flexibility, diversity, and style adherence.\n* Anima-Turbo\n  \n    + Coming soon.\n\n# Installing and running\n\nGet the text encoder and VAE from the [HuggingFage page](https://huggingface.co/circlestone-labs/Anima) .\n\nThe model is natively supported in ComfyUI.\nThe model files go in their respective folders inside your model directory:\n\n* anima-base-v1.0.safetensors goes in ComfyUI/models/diffusion\\_models\n* qwen\\_3\\_06b\\_base.safetensors goes in ComfyUI/models/text\\_encoders\n* qwen\\_image\\_vae.safetensors goes in ComfyUI/models/vae (this is the Qwen-Image VAE, you might already have it)\n\n## Generation settings\n\n* Works at resolutions between 512^2 and 1536^2 pixels.\n* 30-50 steps, CFG 4-6.\n* A variety of samplers work. Some of my favorites:\n  \n    + er\\_sde: neutral style, flat colors, sharp lines. I use this as a reasonable default.\n    + euler\\_a: Softer, thinner lines. Can sometimes tend towards a 2.5D look. CFG can be pushed a bit higher than other samplers without burning the image.\n    + dpmpp\\_2m\\_sde\\_gpu: similar in style to er\\_sde but can produce more variety and be more \"creative\". Depending on the prompt it can get too wild sometimes.\n* If going for a more realistic / painterly look, the beta57 scheduler (ComfyUI RES4LYF custom node pack) can help make better textures, since it puts more emphasis on low-noise timesteps.\n\n# Prompting\n\nThe model is trained on Danbooru-style tags, natural language captions, and combinations of tags and captions.\n\n* Use lowercase for tags, and spaces instead of underscores. Score tags are the only tags that use underscores.\n* Recommended positive prefix: \"masterpiece, best quality, score\\_7, safe, \"\n* Recommended negative: \"worst quality, low quality, score\\_1, score\\_2, score\\_3, artist name\"\n* When using a tag that is different between Danbooru and Gelbooru, prefer the Gelbooru version.\n* Prompt weighting works, but needs a weight higher than typically used for SDXL. Example: \"(chibi:2)\"\n\n## Tag order\n\n[quality/meta/year/safety tags] [1girl/1boy/1other etc] [character] [series] [artist] [general tags]\n\nWithin each tag section, the tags can be in arbitrary order.\n\n## Quality tags\n\n...\n\nIf you would like a commercial license, please email [[email protected]](/cdn-cgi/l/email-protection)\n\nBuilt on NVIDIA Cosmos.\n\n*[14 days ago]: 2026-"
      ]
    },
    {
      "url": "https://github.com/leejet/stable-diffusion.cpp/issues/1245",
      "title": "[Feature] Anima 2B model · Issue #1245 · leejet/stable-diffusion.cpp · GitHub",
      "publish_date": "2026-02-02",
      "excerpts": [
        "## Navigation Menu\n\nToggle navigation\n\nAppearance settings\n\nSearch or jump to...\n\n# Search code, repositories, users, issues, pull requests...\n\nAppearance settings\n\nResetting focus\n\n# [Feature] Anima 2B model \\\n\nNew issue\n\nCopy link\n\nNew issue\n\nCopy link\n\nOpen\n\nOpen\n\n[[Feature] Anima 2B model]() \\\n\nCopy link\n\nLabels\n\n[enhancement New feature or request](https://github.com/leejet/stable-diffusion.cpp/issues?q=state%3Aopen%20label%3A%22enhancement%22) New feature or request\n\n[@GreenShadows](https://github.com/GreenShadows)\n\n## Description\n\n[@GreenShadows](https://github.com/GreenShadows)\n\n[GreenShadows](https://github.com/GreenShadows)\n\nopened [on Feb 2, 2026](https://github.com/leejet/stable-diffusion.cpp/issues/1245)\n\nIssue body actions\n\n### Feature Summary\n\nAnima is a 2 billion parameter text-to-image model created via a collaboration between CircleStone Labs and Comfy Org.\n\n### Detailed Description\n\n<https://huggingface.co/circlestone-labs/Anima>\nAnima is a 2 billion parameter text-to-image model created via a collaboration between CircleStone Labs and Comfy Org. It is focused mainly on anime concepts, characters, and styles, but is also capable of generating a wide variety of other non-photorealistic content. The model is designed for making illustrations and artistic images, and will not work well at realism.\n\nIt is trained on several million anime images and about 800k non-anime artistic images. No synthetic data was used for training. The knowledge cut-off date for the anime training data is September 2025.\n\nThis preview version is an intermediate model checkpoint. The model is still training and the final version will improve, especially for fine details and overall aesthetics.\n\n### Alternatives you considered\n\n[Image](https://private-user-images.githubusercontent.com/118176337/543703800-7edc0b2e-f03c-48cf-a143-c8aa099c9f0a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.\neyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODAwNjMyMzksIm5iZiI6MTc4MDA2MjkzOSwicGF0aCI6Ii8xMTgxNzYzMzcvNTQzNzAzODAwLTdlZGMwYjJlLWYwM2MtNDhjZi1hMTQzLWM4YWEwOTljOWYwYS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUyOVQxMzU1MzlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02MjJmZjY2ODJhMDY4NTA3NWY3YjYzNjNmZjA4MDA0MjEyZDA4MWIxNTY4ZjAyMWE5ZmJjODExMTI2MDU4NTI4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.ycGxNX8nFlbGvLyqPLf2hssUTEuwhQPHD5T0l72TDoQ)\n\n### Additional context\n\n_No response_\n\nReactions are currently unavailable\n\n## Metadata\n\n## Metadata\n\n### Assignees\n\nNo one assigned\n\n### Labels\n\n[enhancement New feature or request](https://github.com/leejet/stable-diffusion.cpp/issues?q=state%3Aopen%20label%3A%22enhancement%22) New feature or request\n\n### Projects"
      ]
    },
    {
      "url": "https://www.emergentmind.com/topics/diffusion-transformer-dit-architecture",
      "title": "Diffusion Transformer (DiT) Architecture - emergentmind.com",
      "publish_date": null,
      "excerpts": [
        "Diffusion Transformer (DiT) Architecture\n\nPapers\n\nTopics\n\nAuthors\n\nRecent\n\nView all\n\nSearch\n\n2000 character limit reached\n\nChrome Extension\n\n[Install our Chrome Extension](https://chromewebstore.google.com/detail/emergent-mind-%E2%80%94-arxiv-int/hgmnadjffdiipehljmhagdgpaoiiklml) to automatically enhance arXiv.\n\nSponsor\n\nPromote your business to millions of monthly visitors.\n\n# Diffusion Transformer (DiT) Architecture\n\nUpdated 5 September 2025\n\n* DiT is a transformer-based diffusion model operating in the latent space of a pretrained VAE, replacing convolutional U-Nets with patch-based transformers.\n* It processes image patches through scalable transformer blocks using in-context conditioning, cross-attention, and adaptive layer normalization for flexible conditional generation.\n* DiT achieves state-of-the-art high-resolution image synthesis with lower compute by effectively scaling model depth, width, and token count.\n\nThe [Diffusion Transformer](https://www.emergentmind.\ncom/topics/diffusion-transformer-base-lpm) (DiT) architecture defines a family of [generative diffusion models](https://www.emergentmind.com/topics/generative-diffusion-models) that replace the pervasive convolutional U-Net backbone with a patch-based transformer stack, operating in the latent space of a pretrained [variational autoencoder](https://www.emergentmind.com/topics/variational-autoencoder-vae) ( [VAE](https://www.emergentmind.com/topics/multi-scale-variational-autoencoder-vae) ). DiT models introduce a scalable, flexible approach to conditional and unconditional generative modeling, leveraging the global modeling capability and scaling properties of the transformer, and establishing state-of-the-art performance on high-resolution image synthesis benchmarks.\n\n## 1\\. Architectural Foundations\n\nDiT adopts a fully transformer-based backbone for diffusion modeling within latent representations.\nGiven an input image, a pretrained VAE first encodes the data into a low-dimensional latent z (e.g., for a 256 × 256 pixel image, an example latent size is 32 × 32 × C channels). This latent is then partitioned into non-overlapping patches (of size p × p ), linearly embedded to form a sequence of tokens. Unlike U-Net backbones, this patchification admits an isotropic transformer structure reminiscent of the [Vision Transformer](https://www.emergentmind.com/topics/vision-transformer-swin-video) , discarding explicit spatial hierarchy.\n\nEach [patch embedding](https://www.emergentmind.com/topics/patch-embedding) is summed with sinusoidal positional encodings. The sequence is then processed through a stack of transformer blocks, each optionally incorporating multiple forms of conditional information (e.g., timestep, class label) via:\n\n* **In-Context Conditioning:** Appends condition embeddings to the token sequence."
      ]
    },
    {
      "url": "https://www.youngju.dev/blog/ai-papers/2026-03-03-diffusion-transformer-dit-analysis.en",
      "title": "Diffusion Transformer (DiT) Architecture Analysis: The Shift from U-Net to Transformer | Chaos and Order",
      "publish_date": "2026-03-03",
      "excerpts": [
        "ChaosAndOrder\n\nBlog Tags Projects Tools Explore About\n\n1. Home\n2. /\n3. Blog\n4. /\n5. blog\n6. /\n7. Diffusion Transformer (DiT) Architecture Analysis: The Shift from U-Net to Transformer\n\nPublished on\n    2026년 3월 3일 화요일\n\n# Diffusion Transformer (DiT) Architecture Analysis: The Shift from U-Net to Transformer\n\n🇰🇷 KO 한국어 🇺🇸 EN English 🇯🇵 JA 日本語\n\nSplit View 필사 모드\n\nAuthors\n    * avatar\n  \n  Name\n      Youngju Kim\n  Twitter\n      [@fjvbn20031](https://twitter.com/fjvbn20031)\n\n* Introduction\n* Background: Why Move Away from U-Net?\n  \n    + Limitations of U-Net\n    + Advantages of Transformers\n* Model Variants and Scaling\n  \n    + Impact of Patch Size\n* Classifier-Free Guidance\n* Downstream Impact of DiT\n  \n    + SORA (OpenAI, 2024)\n    + Stable Diffusion 3 (Stability AI, 2024)\n    + Dynamic DiT (ICLR 2025)\n* PyTorch Implementation Walkthrough\n  \n    + Custom Training\n* Summary\n* Quiz\n\n##  Introduction\nPublished in late 2022, **\"Scalable Diffusion Models with Transformers\"** (William Peebles & Saining Xie, ICCV 2023) marked a pivotal turning point in shifting the image generation backbone from U-Net to Transformer. The DiT (Diffusion Transformer) architecture proposed in this paper has since become the foundation for OpenAI's SORA, Stability AI's Stable Diffusion 3, and Dynamic DiT presented at ICLR 2025.\n\n##  Background: Why Move Away from U-Net?\n\n###  Limitations of U-Net\n\nTraditional DDPM and LDM (Latent Diffusion Models) use U-Net as the backbone. While U-Net performs well in image generation, it has fundamental limitations:\n\n* **No scaling laws** : Performance gains are unpredictable when scaling up the model\n* **Architectural complexity** : Managing skip connections and feature maps at various resolutions\n* **Computational inefficiency** : Memory and compute costs surge at high resolutions\n* **Difficult multi-modal integration** : Unnatural integration with text, video, and other modalities\n\n###  Advantages of Transformers\n\nIn contrast, Transformers (ViT) offer:\n\n* **Proven scaling laws** : Performance improves proportionally with parameter count (demonstrated in LLMs)\n* **Simple architecture** : Repetition of Self-attention + FFN blocks\n* **Modality agnostic** : Text, images, and video can all be processed with the same structure\n* **Hardware optimization** : Efficient parallel processing on GPUs/TPUs\n\n##  DiT Architecture in Detail\n\n###  Overall Pipeline\n\nDiT operates on top of the Latent Diffusion framework:\n\n1. Encode images into latent space using a VAE encoder\n2. Split the latent representation into patches (same as ViT)\n3. Predict noise using **DiT blocks** (Transformer)\n4. Reconstruct images with the VAE decoder\n\n###  Patch Embedding\n\n```\nimport  torch\n import  torch . nn  as  nn\n class PatchEmbed ( nn ."
      ]
    },
    {
      "url": "https://huggingface.co/circlestone-labs/Anima/discussions/35",
      "title": "circlestone-labs/Anima · The training script for the Anima model has already been implemented for sd-scripts",
      "publish_date": null,
      "excerpts": [
        "You can check my pull request here: <https://github.com/kohya-ss/sd-scripts/pull/2260> . And here is my repo for implement it: <https://github.com/duongve13112002/sd-scripts/tree/sd3>\n\nGuide:  \nI. Training\n\nThis model can be trained using the `sd-scripts` training pipeline with either LoRA or full finetuning.\n\n1. LoRA Training\n\nUse LoRA adapters to reduce memory usage and training time while keeping the base model frozen:\n\n```\naccelerate launch anima_train_network.py \\\n  --dit_path <path_to_anima_dit> \\\n  --vae_path <path_to_wanvae> \\\n  --qwen3_path <path_to_qwen3> \\\n  --dataset_config my_dataset_config.toml \\\n  --output_dir output \\\n  --output_name anima_lora \\\n  --network_module networks.lora_anima \\\n  --network_dim 16 \\\n  --learning_rate 1e-4 \\\n  --max_train_epochs 10 \\\n  --cache_text_encoder_outputs \\\n  --cache_latents\n```\n\n* * *\n\n1. Full Finetuning\n\nFor full model finetuning, including DiT and optional LLM Adapter components:\n\n```\naccelerate launch anima_train.py \\\n--dit_path <path_to_anima_dit> \\\n  --vae_path <path_to_wanvae> \\\n  --qwen3_path <path_to_qwen3> \\\n  --dataset_config my_dataset_config.toml \\\n  --output_dir output \\\n  --output_name anima_ft \\\n  --learning_rate 1e-5 \\\n  --max_train_epochs 10 \\\n  --cache_text_encoder_outputs \\\n  --cache_latents \\\n  --llm_adapter_lr 0\n```\n\nSetting `--llm_adapter_lr 0` freezes the LLM Adapter during training.\n\n* * *\n\nII. Notes\n\n* Anima is a ~2B parameter DiT-based model.\n* VRAM usage can be significantly reduced by enabling:\n\n\\--cache\\_text\\_encoder\\_outputs  \n\\--cache\\_latents  \n\\--gradient\\_checkpointing\n\n* LoRA training supports per-layer rank configuration via `--network_args` .\n\nSee translation\n\n❤️\n\n14\n\n14\n\n\\+\n\nkazuyi1222 Feb 5\n\nThis comment has been hidden (marked as Off-Topic)\n\nkazuyi1222 Feb 6\n\n[image](https://cdn-uploads.huggingface.co/production/uploads/688892c9c21ff9bea3ded510/RLC9aDz5T60xwhoAOYMtI.png)\n\n我使用你的分支训练了一个lora，好像出图不太对劲（"
      ]
    },
    {
      "url": "https://startupfortune.com/anima-base-v10-gives-creators-a-new-anime-model-to-test",
      "title": "Anima-Base v1.0 gives creators a new anime model to test – Startup Fortune",
      "publish_date": "2026-05-14",
      "excerpts": [
        "- [Opinion](https://startupfortune.com/category/opinion/)\n- [Latest](https://startupfortune.com/latest/)\n\nMost Read\n\n[Startup Newsletter](https://startupfortune.com/newsletter/) [Japan’s $2,000 cardboard drones are not…](https://startupfortune.com/japans-2000-cardboard-drones-are-not-a-novelty-and-the-defense-startup-implications-are-more-serious-than-the-material-suggests/) [A Georgia data center shows why…](https://startupfortune.com/a-georgia-data-center-shows-why-ai-has-a-water-problem/) [The Numbers Are In, Small Publishers…](https://startupfortune.com/the-numbers-are-in-small-publishers-are-losing-search-traffic-at-an-alarming-rate/) [Startup Newsletter](https://startupfortune.com/newsletter/) [Japan’s $2,000 cardboard drones are not…](https://startupfortune.com/japans-2000-cardboard-drones-are-not-a-novelty-and-the-defense-startup-implications-are-more-serious-than-the-material-suggests/) [A Georgia data center shows why…](https://startupfortune.\ncom/a-georgia-data-center-shows-why-ai-has-a-water-problem/) [The Numbers Are In, Small Publishers…](https://startupfortune.com/the-numbers-are-in-small-publishers-are-losing-search-traffic-at-an-alarming-rate/)\n\n[Home](https://startupfortune.com/) › [Ai](https://startupfortune.com/category/ai/) › Anima-Base v1.0 gives creators a new anime model to test\n\n# Anima-Base v1.0 gives creators a new anime model to test\n\n[Ai](https://startupfortune.com/category/ai/) / [Entrepreneurship](https://startupfortune.com/category/entrepreneurship/) | CircleStone Labs has released Anima-Base v1.0, a 2B-parameter text-to-image model for anime and illustration workflows. It could become useful infrastructure for creators, but its non-commercial license limits how quickly startups can build paid products around it.\n\n[Ron Patel](https://startupfortune.com/author/ron-patel/)\n\nRon Patel covers cryptocurrency markets, blockchain…\n\nMay 15, 2026 5 min read 351 views\n\nMay 15, 2026\n\n5 min read · 351 views"
      ]
    },
    {
      "url": "https://www.aimodels.fyi/models/huggingFace/anima-lllite-kohya-ss",
      "title": "Anima-LLLite: Image-to-Image model — overview, use cases, alternatives",
      "publish_date": null,
      "excerpts": [
        "**🤖 AIModels.fyi**\n\nCreators Models Collections Tools Papers Researchers [Notes](https://notes.aimodels.fyi) API NEW Sign in / sign up\n\n**🤖 AIModels.fyi**\n\nCreators Models Collections Tools Papers Researchers [Notes](https://notes.aimodels.fyi) API NEW Sign in / sign up\n\n# [Anima-LLLite](https://huggingface.co/kohya-ss/Anima-LLLite)\n\nMaintainer: kohya-ss \\- Last updated 26/5/2026\n\n### Actions for this model [Run model](https://huggingface.co/kohya-ss/Anima-LLLite)\n\n📈\n\nAnima-LLLite\n\nGet notified when new models like this one come out!\n\nEmail Address\n\nStart Free Trial →\n\n* * *\n\n* * *\n\nHave an account? We'll apply the trial to it\n\n## Model Overview\n\n#### Overview\n\n`Anima-LLLite` implements ControlNet-LLLite, a lightweight conditional control module designed for the Anima DiT (MiniTrainDIT) diffusion architecture. Built by [kohya-ss](https://aimodels.\nfyi/creators/huggingFace/kohya-ss) , these sample weights provide structured image-to-image control through five distinct conditioning modes: lineart, depth maps, pose, fake scribble, and inpainting. The architecture uses a LoRA-like injection mechanism with configurable embedding dimensions ( `--cond_emb_dim 32 --lllite_cond_dim 32 --lllite_mlp_dim 32` ) integrated into Anima's self-attention layers. The base model runs on Anima DiT ( `anima-preview3-base` ) with Qwen-Image VAE and Qwen3-0.6B text encoder. The weights are distributed as `.safetensors` files and require either direct integration with sd-scripts or the experimental ComfyUI node for inference. Most important caveat: these are community reference implementations with intentionally moderate effect strength, particularly the pose model, which functions as a soft prior rather than strict pose-locking control.\n\n#### Best use cases\n\n**Lineart-guided illustration refinement."
      ]
    },
    {
      "url": "https://deepwiki.com/kohya-ss/sd-scripts/7.3-anima-training",
      "title": "Anima Training | kohya-ss/sd-scripts | DeepWiki",
      "publish_date": "2026-04-13",
      "excerpts": [
        "Loading...\n\nIndex your code with Devin\n\nDeepWiki\n\nDeepWiki\n\n[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts \"Open repository\")\n\nIndex your code with\n\nDevin Edit Wiki Share\n\nLoading...\n\nMenu\n\n# Anima Training\n\nRelevant source files\n\n* [anima\\_train.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train.py)\n* [anima\\_train\\_network.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train_network.py)\n* [library/anima\\_models.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py)\n* [library/anima\\_train\\_utils.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_train_utils.py)\n* [library/anima\\_utils.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py)\n* [library/strategy\\_anima.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/strategy_anima.py)\n\nThis page documents the training implementation for the Anima architecture within the `sd-scripts` framework.\nAnima is a Diffusion Transformer (DiT) model inspired by the Cosmos-Predict2 style, featuring a unique LLMAdapter bridge for text conditioning, 3D Rotary Positional Embeddings (RoPE), and specialized training utilities for full fine-tuning and LoRA.\n\n## Architecture Overview\n\nThe Anima model is implemented in `library/anima_models.py` and consists of several key components that differentiate it from standard MMDiT or FLUX architectures.\n\n### Anima DiT Structure\n\nThe core model class `Anima` [library/anima\\_models.py 1143-1188](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py) manages the forward pass of the latent diffusion process. It uses a series of `AnimaBlock` layers [library/anima\\_models.py 978-1036](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py) which combine self-attention, cross-attention, and MLP processing.\n\n### LLMAdapter Bridge\n\nA defining feature of Anima is the `LLMAdapter` [library/anima\\_models."
      ]
    }
  ],
  "warnings": null,
  "usage": [
    {
      "name": "sku_search",
      "count": 1
    }
  ],
  "session_id": "ses_17f67f5beffeKq0EJMHZjfupbo"
}
```

---

## Assistant (Build · MiMo V2.5 Free · 31.7s)

_Thinking:_

Now I have good information about the Anima model. Let me gather more details about its architecture, particularly around RoPE, attention mechanisms, and how it differs from Flux.

**Tool: webfetch**

**Input:**
```json
{
  "url": "https://deepwiki.com/kohya-ss/sd-scripts/7.3-anima-training",
  "format": "markdown"
}
```

**Output:**
```
Anima Training | kohya-ss/sd-scripts | DeepWiki

Loading...

Index your code with Devin

[DeepWiki](/)

[DeepWiki](/)

[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts "Open repository")

Index your code with

Devin

Edit WikiShare

Loading...

Last indexed: 13 April 2026 ([308a0c](https://github.com/kohya-ss/sd-scripts/commits/308a0cc9))

-   [Overview](/kohya-ss/sd-scripts/1-overview)
-   [Installation and Setup](/kohya-ss/sd-scripts/1.1-installation-and-setup)
-   [Repository Structure](/kohya-ss/sd-scripts/1.2-repository-structure)
-   [Supported Models and Training Methods](/kohya-ss/sd-scripts/1.3-supported-models-and-training-methods)
-   [Core Training System](/kohya-ss/sd-scripts/2-core-training-system)
-   [Dataset Management System](/kohya-ss/sd-scripts/2.1-dataset-management-system)
-   [Model Loading and Optimization](/kohya-ss/sd-scripts/2.2-model-loading-and-optimization)
-   [Training Loop Infrastructure](/kohya-ss/sd-scripts/2.3-training-loop-infrastructure)
-   [Strategy Pattern Architecture](/kohya-ss/sd-scripts/2.4-strategy-pattern-architecture)
-   [Caching System](/kohya-ss/sd-scripts/2.5-caching-system)
-   [Standard SD Training Methods](/kohya-ss/sd-scripts/3-standard-sd-training-methods)
-   [DreamBooth Training](/kohya-ss/sd-scripts/3.1-dreambooth-training)
-   [Fine-tuning](/kohya-ss/sd-scripts/3.2-fine-tuning)
-   [LoRA Training](/kohya-ss/sd-scripts/3.3-lora-training)
-   [Textual Inversion](/kohya-ss/sd-scripts/3.4-textual-inversion)
-   [SDXL Training](/kohya-ss/sd-scripts/4-sdxl-training)
-   [SDXL Architecture and Utilities](/kohya-ss/sd-scripts/4.1-sdxl-architecture-and-utilities)
-   [SDXL Fine-tuning](/kohya-ss/sd-scripts/4.2-sdxl-fine-tuning)
-   [SDXL LoRA Training](/kohya-ss/sd-scripts/4.3-sdxl-lora-training)
-   [SDXL ControlNet Training](/kohya-ss/sd-scripts/4.4-sdxl-controlnet-training)
-   [SDXL Textual Inversion](/kohya-ss/sd-scripts/4.5-sdxl-textual-inversion)
-   [SD3 Training](/kohya-ss/sd-scripts/5-sd3-training)
-   [SD3 Architecture and Models](/kohya-ss/sd-scripts/5.1-sd3-architecture-and-models)
-   [SD3 Fine-tuning](/kohya-ss/sd-scripts/5.2-sd3-fine-tuning)
-   [SD3 LoRA Training](/kohya-ss/sd-scripts/5.3-sd3-lora-training)
-   [SD3 Inference](/kohya-ss/sd-scripts/5.4-sd3-inference)
-   [FLUX Training](/kohya-ss/sd-scripts/6-flux-training)
-   [FLUX Architecture and Models](/kohya-ss/sd-scripts/6.1-flux-architecture-and-models)
-   [FLUX Fine-tuning](/kohya-ss/sd-scripts/6.2-flux-fine-tuning)
-   [FLUX LoRA Training](/kohya-ss/sd-scripts/6.3-flux-lora-training)
-   [FLUX ControlNet Training](/kohya-ss/sd-scripts/6.4-flux-controlnet-training)
-   [FLUX Inference](/kohya-ss/sd-scripts/6.5-flux-inference)
-   [Newer Model Architectures](/kohya-ss/sd-scripts/7-newer-model-architectures)
-   [Lumina Image 2.0 Training](/kohya-ss/sd-scripts/7.1-lumina-image-2.0-training)
-   [HunyuanImage-2.1 Training](/kohya-ss/sd-scripts/7.2-hunyuanimage-2.1-training)
-   [Anima Training](/kohya-ss/sd-scripts/7.3-anima-training)
-   [LoRA Networks](/kohya-ss/sd-scripts/8-lora-networks)
-   [LoRA Core Implementation](/kohya-ss/sd-scripts/8.1-lora-core-implementation)
-   [LoRA Variants](/kohya-ss/sd-scripts/8.2-lora-variants)
-   [Model-Specific LoRA Features](/kohya-ss/sd-scripts/8.3-model-specific-lora-features)
-   [Regional LoRA and Advanced Features](/kohya-ss/sd-scripts/8.4-regional-lora-and-advanced-features)
-   [LoRA Utilities and Management](/kohya-ss/sd-scripts/8.5-lora-utilities-and-management)
-   [Image Generation](/kohya-ss/sd-scripts/9-image-generation)
-   [Standard SD Generation Pipeline](/kohya-ss/sd-scripts/9.1-standard-sd-generation-pipeline)
-   [SDXL Generation](/kohya-ss/sd-scripts/9.2-sdxl-generation)
-   [Attention Mechanisms and Optimizations](/kohya-ss/sd-scripts/9.3-attention-mechanisms-and-optimizations)
-   [Dataset Preparation](/kohya-ss/sd-scripts/10-dataset-preparation)
-   [Dataset Configuration System](/kohya-ss/sd-scripts/10.1-dataset-configuration-system)
-   [Bucket Management](/kohya-ss/sd-scripts/10.2-bucket-management)
-   [Image Preprocessing and Augmentation](/kohya-ss/sd-scripts/10.3-image-preprocessing-and-augmentation)
-   [Automated Captioning and Tagging](/kohya-ss/sd-scripts/10.4-automated-captioning-and-tagging)
-   [Advanced Topics](/kohya-ss/sd-scripts/11-advanced-topics)
-   [Loss Functions and Training Techniques](/kohya-ss/sd-scripts/11.1-loss-functions-and-training-techniques)
-   [Memory Optimization Techniques](/kohya-ss/sd-scripts/11.2-memory-optimization-techniques)
-   [Model Format Conversion](/kohya-ss/sd-scripts/11.3-model-format-conversion)
-   [Model Metadata and SAI ModelSpec](/kohya-ss/sd-scripts/11.4-model-metadata-and-sai-modelspec)
-   [Command-Line Arguments Reference](/kohya-ss/sd-scripts/11.5-command-line-arguments-reference)
-   [Validation and Testing](/kohya-ss/sd-scripts/11.6-validation-and-testing)
-   [Glossary](/kohya-ss/sd-scripts/12-glossary)

Menu

# Anima Training

Relevant source files

-   [anima\_train.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train.py)
-   [anima\_train\_network.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train_network.py)
-   [library/anima\_models.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py)
-   [library/anima\_train\_utils.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_train_utils.py)
-   [library/anima\_utils.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py)
-   [library/strategy\_anima.py](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/strategy_anima.py)

This page documents the training implementation for the Anima architecture within the `sd-scripts` framework. Anima is a Diffusion Transformer (DiT) model inspired by the Cosmos-Predict2 style, featuring a unique LLMAdapter bridge for text conditioning, 3D Rotary Positional Embeddings (RoPE), and specialized training utilities for full fine-tuning and LoRA.

## Architecture Overview

The Anima model is implemented in `library/anima_models.py` and consists of several key components that differentiate it from standard MMDiT or FLUX architectures.

### Anima DiT Structure

The core model class `Anima` [library/anima\_models.py1143-1188](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L1143-L1188) manages the forward pass of the latent diffusion process. It uses a series of `AnimaBlock` layers [library/anima\_models.py978-1036](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L978-L1036) which combine self-attention, cross-attention, and MLP processing.

### LLMAdapter Bridge

A defining feature of Anima is the `LLMAdapter` [library/anima\_models.py534-583](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L534-L583) This component acts as a bridge between the text encoder (Qwen3) and the DiT blocks. It processes text embeddings through a transformer-based architecture to provide refined conditioning signals to the cross-attention layers.

### 3D Rotary Positional Embeddings (RoPE)

Anima utilizes 3D RoPE to handle spatial and temporal dimensions. The implementation includes:

-   `apply_rotary_pos_emb`: The primary entry point for applying rotary embeddings [library/anima\_models.py175-212](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L175-L212)
-   `PositionEmbedding2D`: Handles spatial positional encoding [library/anima\_models.py328-361](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L328-L361)
-   `PositionEmbedding3D`: Extends encoding to include the temporal dimension for video or multi-frame support [library/anima\_models.py364-419](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L364-L419)

### Natural Language to Code Mapping

System Concept

Code Entity

File Path

**Main Model**

`Anima` class

[library/anima\_models.py1143](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L1143-L1143)

**Transformer Block**

`AnimaBlock` class

[library/anima\_models.py978](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L978-L978)

**Text Conditioning Bridge**

`LLMAdapter` class

[library/anima\_models.py534](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L534-L534)

**Positional Encoding**

`PositionEmbedding3D` class

[library/anima\_models.py364](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L364-L364)

**Text Encoder**

`Qwen2ForCausalLM` (Qwen3)

[library/anima\_utils.py168](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py#L168-L168)

**VAE**

`Qwen2ImageAutoencoderKL`

[library/qwen\_image\_autoencoder\_kl.py13](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/qwen_image_autoencoder_kl.py#L13-L13)

**Sources:** [library/anima\_models.py175-1188](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L175-L1188) [library/anima\_utils.py168-175](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py#L168-L175) [library/qwen\_image\_autoencoder\_kl.py13-50](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/qwen_image_autoencoder_kl.py#L13-L50)

---

## Data Flow and Training Logic

The training process involves tokenization across two different schemes, latent encoding, and a flow-matching based diffusion objective.

### Tokenization Strategy

Anima uses a dual tokenization approach via `AnimaTokenizeStrategy` [library/strategy\_anima.py22-70](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/strategy_anima.py#L22-L70):

1.  **Qwen3 Tokens**: Used for the actual text encoder to generate hidden states [library/strategy\_anima.py57-61](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/strategy_anima.py#L57-L61)
2.  **T5 Tokens**: Used as target input IDs for the `LLMAdapter`, though they are not encoded by a T5 model during this stage [library/strategy\_anima.py64-68](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/strategy_anima.py#L64-L68)

### Training Loop Integration

The training loop is managed by `AnimaNetworkTrainer` for LoRA [anima\_train\_network.py33-134](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train_network.py#L33-L134) or the `train` function in `anima_train.py` for full fine-tuning [anima\_train.py43-150](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train.py#L43-L150)

**Sources:** [library/strategy\_anima.py22-107](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/strategy_anima.py#L22-L107) [anima\_train\_network.py33-134](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train_network.py#L33-L134) [anima\_train.py43-150](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train.py#L43-L150)

---

## LoRA Training

LoRA training for Anima is implemented in `anima_train_network.py`. It supports targeting specific components of the DiT architecture.

### Target Modules

By default, LoRA can be applied to:

-   Self-attention layers (`.self_attn`)
-   Cross-attention layers (`.cross_attn`)
-   MLP layers (`.mlp`)

### Parameter Groups and Learning Rates

Anima supports separate learning rates for different architectural blocks via `get_anima_param_groups` [library/anima\_train\_utils.py161-230](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_train_utils.py#L161-L230) This allows freezing or fine-tuning specific parts of the model like the `LLMAdapter` or `AdaLN` modulation layers independently [library/anima\_train\_utils.py41-75](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_train_utils.py#L41-L75)

### Memory Optimizations

To handle the large parameter count of Anima models, several optimizations are available:

-   **Unsloth Offloaded Gradient Checkpointing**: Offloads activations to CPU RAM using non-blocking transfers [library/anima\_models.py75-124](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L75-L124)
-   **Block Swapping**: Dynamically swaps DiT blocks between CPU and GPU [anima\_train\_network.py130-132](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train_network.py#L130-L132)
-   **FP8 Scaling**: Supports loading the DiT model with FP8 precision [library/anima\_utils.py120-128](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py#L120-L128)

Feature

Argument

Implementation

**Unsloth Checkpointing**

`--unsloth_offload_checkpointing`

`UnslothOffloadedGradientCheckpointer` [library/anima\_models.py75](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L75-L75)

**Block Swapping**

`--blocks_to_swap`

`model.enable_block_swap()` [anima\_train\_network.py132](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train_network.py#L132-L132)

**Attention Mode**

`--attn_mode`

`load_anima_model` parameter [library/anima\_utils.py35](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py#L35-L35)

**Sources:** [library/anima\_train\_utils.py41-230](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_train_utils.py#L41-L230) [library/anima\_models.py75-124](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_models.py#L75-L124) [anima\_train\_network.py125-134](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/anima_train_network.py#L125-L134) [library/anima\_utils.py32-148](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py#L32-L148)

---

## Configuration and Utilities

### Model Loading

The utility `load_anima_model` [library/anima\_utils.py32-148](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py#L32-L148) handles the complex initialization of the Anima DiT, including:

-   Applying `WeightTransformHooks` to strip prefixes like `net.` from state dicts [library/anima\_utils.py106-118](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py#L106-L118)
-   Initializing the model with empty weights via `init_empty_weights()` to save memory during setup [library/anima\_utils.py99-103](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py#L99-L103)
-   Handling FP8 monkey patches if `fp8_scaled` is enabled [library/anima\_utils.py121](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py#L121-L121)

### Loss Weighting

Anima training supports multiple loss weighting schemes via `compute_loss_weighting_for_anima` [library/anima\_train\_utils.py143-157](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_train_utils.py#L143-L157) including `sigma_sqrt` and `cosmap` for rectified flow training.

**Sources:** [library/anima\_utils.py32-148](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_utils.py#L32-L148) [library/anima\_train\_utils.py143-157](https://github.com/kohya-ss/sd-scripts/blob/308a0cc9/library/anima_train_utils.py#L143-L157)

Dismiss

Refresh this wiki

Enter email to refresh

### On this page

-   [Anima Training](#anima-training)
-   [Architecture Overview](#architecture-overview)
-   [Anima DiT Structure](#anima-dit-structure)
-   [LLMAdapter Bridge](#llmadapter-bridge)
-   [3D Rotary Positional Embeddings (RoPE)](#3d-rotary-positional-embeddings-rope)
-   [Natural Language to Code Mapping](#natural-language-to-code-mapping)
-   [Data Flow and Training Logic](#data-flow-and-training-logic)
-   [Tokenization Strategy](#tokenization-strategy)
-   [Training Loop Integration](#training-loop-integration)
-   [LoRA Training](#lora-training)
-   [Target Modules](#target-modules)
-   [Parameter Groups and Learning Rates](#parameter-groups-and-learning-rates)
-   [Memory Optimizations](#memory-optimizations)
-   [Configuration and Utilities](#configuration-and-utilities)
-   [Model Loading](#model-loading)
-   [Loss Weighting](#loss-weighting)
```

**Tool: webfetch**

**Input:**
```json
{
  "url": "https://huggingface.co/circlestone-labs/Anima",
  "format": "markdown"
}
```

**Output:**
```
                

  circlestone-labs/Anima · Hugging Face  

 [![Hugging Face's logo](/front/assets/huggingface_logo-noborder.svg) Hugging Face](/)

-   [Models](/models)
-   [Datasets](/datasets)
-   [Spaces](/spaces)
-   [Buckets new](/storage)
-   [Docs](/docs)
-   [Enterprise](/enterprise)
-   [Pricing](/pricing)
-   -   Website
        
        -   [Tasks](/tasks)
        -   [HuggingChat](/chat)
        -   [Collections](/collections)
        -   [Languages](/languages)
        -   [Organizations](/organizations)
    -   Community
        
        -   [Blog](/blog)
        -   [Posts](/posts)
        -   [Daily Papers](/papers)
        -   [Learn](/learn)
        -   [Discord](/join/discord)
        -   [Forum](https://discuss.huggingface.co/)
        -   [GitHub](https://github.com/huggingface)
    -   Solutions
        
        -   [Team & Enterprise](/enterprise)
        -   [Hugging Face PRO](/pro)
        -   [Enterprise Support](/support)
        -   [Inference Providers](/inference/models)
        -   [Inference Endpoints](/inference-endpoints)
        -   [Storage Buckets](/storage)
    
-   ---
    
-   [Log In](/login)
-   [Sign Up](/join)

# 

[

![](https://www.gravatar.com/avatar/cf1eefba382909ceb1c5b8273d8a5c94?d=retro&size=100)

](/circlestone-labs)

[circlestone-labs](/circlestone-labs)

/

[Anima](/circlestone-labs/Anima)

like 1.61k

Follow

![](https://www.gravatar.com/avatar/cf1eefba382909ceb1c5b8273d8a5c94?d=retro&size=100) CircleStone Labs 757

[

Diffusion Single File

](/models?library=diffusion-single-file)[

comfyui

](/models?other=comfyui)

License: circlestone-labs-non-commercial-license

[Model card](/circlestone-labs/Anima) [Files Files and versions

xet

](/circlestone-labs/Anima/tree/main)[Community

183

](/circlestone-labs/Anima/discussions)

Copy to bucket new

Use this model

### Instructions to use circlestone-labs/Anima with libraries, inference providers, notebooks, and local apps. Follow these links to get started.

-   Libraries
-   [Diffusion Single File](/circlestone-labs/Anima?library=diffusion-single-file)
    
    How to use circlestone-labs/Anima with Diffusion Single File:
    
    \# No code snippets available yet for this library.
    
    # To use this model, check the repository files and the library's documentation.
    
    # Want to help? PRs adding snippets are welcome at:
    # https://github.com/huggingface/huggingface.js
    
-   Notebooks
-   [Google Colab](/circlestone-labs/Anima/colab)
-   [Kaggle](/circlestone-labs/Anima/kaggle)

-   [Versions](#versions "Versions")
    
-   [Installing and running](#installing-and-running "Installing and running")
    -   [Generation settings](#generation-settings "Generation settings")
        
-   [Prompting](#prompting "Prompting")
    -   [Tag order](#tag-order "Tag order")
        
    -   [Quality tags](#quality-tags "Quality tags")
        
    -   [Time period tags](#time-period-tags "Time period tags")
        
    -   [Meta tags](#meta-tags "Meta tags")
        
    -   [Safety tags](#safety-tags "Safety tags")
        
    -   [Artist tags](#artist-tags "Artist tags")
        
    -   [Full tag example](#full-tag-example "Full tag example")
        
    -   [Tag dropout](#tag-dropout "Tag dropout")
        
    -   [Dataset tags](#dataset-tags "Dataset tags")
        
    -   [Natural language prompting tips](#natural-language-prompting-tips "Natural language prompting tips")
        
-   [Model comparison](#model-comparison "Model comparison")
    
-   [Limitations](#limitations "Limitations")
    
-   [Finetuning Tips](#finetuning-tips "Finetuning Tips")
    
-   [License](#license "License")
    

![](/circlestone-labs/Anima/resolve/main/montage.jpg)

Anima is a 2 billion parameter text-to-image model created via a collaboration between CircleStone Labs and Comfy Org. It is focused mainly on anime concepts, characters, and styles, but is also capable of generating a wide variety of other non-photorealistic content. The model is designed for making illustrations and artistic images, and will not work well at realism.

It is trained on several million anime images and about 800k non-anime artistic images. No synthetic data was used for training. The knowledge cut-off date for the anime training data is September 2025.

**NEW:** Try the [Turbo LoRA](https://civitai.com/models/2560840/anima-turbo-lora) for better stability and much faster generations.

# [](#versions)Versions

-   Anima-Base
    -   The pretrained, unrefined base model. Maximum flexibility, diversity, and style adherence.
-   Anima-Turbo
    -   Coming soon.

# [](#installing-and-running)Installing and running

Workflow: ![](/circlestone-labs/Anima/resolve/main/example.png) The model is natively supported in ComfyUI. The above image contains a workflow; you can open it in ComfyUI or drag-and-drop to get the workflow. The model files go in their respective folders inside your model directory:

-   anima-base-v1.0.safetensors goes in ComfyUI/models/diffusion\_models
-   qwen\_3\_06b\_base.safetensors goes in ComfyUI/models/text\_encoders
-   qwen\_image\_vae.safetensors goes in ComfyUI/models/vae (this is the Qwen-Image VAE, you might already have it)

## [](#generation-settings)Generation settings

-   Works at resolutions between 512^2 and 1536^2 pixels.
-   30-50 steps, CFG 4-5.
-   A variety of samplers work. Some of my favorites:
    -   er\_sde: neutral style, flat colors, sharp lines. I use this as a reasonable default.
    -   euler\_a: Softer, thinner lines. Can sometimes tend towards a 2.5D look. CFG can be pushed a bit higher than other samplers without burning the image.
    -   dpmpp\_2m\_sde\_gpu: similar in style to er\_sde but can produce more variety and be more "creative". Depending on the prompt it can get too wild sometimes.
-   If going for a more realistic / painterly look, the beta57 scheduler (ComfyUI RES4LYF custom node pack) can help make better textures, since it puts more emphasis on low-noise timesteps.

# [](#prompting)Prompting

The model is trained on Danbooru-style tags, natural language captions, and combinations of tags and captions.

-   Use lowercase for tags, and spaces instead of underscores. Score tags are the only tags that use underscores.
-   Recommended positive prefix: "masterpiece, best quality, score\_7, safe, "
-   Recommended negative: "worst quality, low quality, score\_1, score\_2, score\_3, artist name"
-   When using a tag that is different between Danbooru and Gelbooru, prefer the Gelbooru version.
-   Prompt weighting works, but needs a weight higher than typically used for SDXL. Example: "(chibi:2)"

## [](#tag-order)Tag order

\[quality/meta/year/safety tags\] \[1girl/1boy/1other etc\] \[character\] \[series\] \[artist\] \[general tags\]

Within each tag section, the tags can be in arbitrary order.

## [](#quality-tags)Quality tags

Human score based: masterpiece, best quality, good quality, normal quality, low quality, worst quality

PonyV7 aesthetic model based: score\_9, score\_8, ..., score\_1

You can use either the human score quality tags, the aesthetic model tags, both together, or neither. All combinations work.

## [](#time-period-tags)Time period tags

Specific year: year 2025, year 2024, ...

Period: newest, recent, mid, early, old

## [](#meta-tags)Meta tags

highres, absurdres, anime screenshot, jpeg artifacts, official art, etc

## [](#safety-tags)Safety tags

safe, sensitive, nsfw, explicit

## [](#artist-tags)Artist tags

Prefix artist with @. E.g. "@big chungus". **You must put @ in front of the artist.** The effect will be very weak if you don't.

## [](#full-tag-example)Full tag example

year 2025, newest, normal quality, score\_5, highres, safe, 1girl, oomuro sakurako, yuru yuri, @nnn yryr, smile, brown hair, hat, solo, fur-trimmed gloves, open mouth, long hair, gift box, fang, skirt, red gloves, blunt bangs, gloves, one eye closed, shirt, brown eyes, santa costume, red hat, skin fang, twitter username, white background, holding bag, fur trim, simple background, brown skirt, bag, gift bag, looking at viewer, santa hat, ;d, red shirt, box, gift, fur-trimmed headwear, holding, red capelet, holding box, capelet

## [](#tag-dropout)Tag dropout

The model was trained with random tag dropout. You don't need to include every single relevant tag for the image.

## [](#dataset-tags)Dataset tags

To improve style and content diversity, the model was additionally trained on two non-anime datasets: LAION-POP (specifically the ye-pop version) and DeviantArt. Both were filtered to exclude photos. Because these datasets are qualitatively different from anime datasets, captions from them have been labeled with a "dataset tag". This occurs at the very beginning of a prompt followed by a newline. Optionally, the second line can contain either the image alt-text (ye-pop) or the title of the work (DeviantArt). Examples:

ye-pop  
For Sale: Others by Arun Prem  
Abstract, oil painting of three faceless, blue-skinned figures. Left: white, draped figure; center: yellow-shirted, dark-haired figure; right: red-veiled, dark-haired figure carrying another. Bold, textured colors, minimalist style.

deviantart  
Flame  
Digital painting of a fiery dragon with glowing yellow eyes, black horns, and a long, sinuous tail, perched on a glowing, molten rock formation. The background is a gradient of dark purple to orange.

## [](#natural-language-prompting-tips)Natural language prompting tips

-   Follow standard English capitalization rules for character and series names.
-   If using pure natural langauge, more descriptive is better. Aim for at least 2 sentences. Extremely short prompts can give unexpected results.
-   You can mix tags and natural language in arbitrary order.
-   You can put quality / artist tags at the beginning of a natural language prompt.
    -   "masterpiece, best quality, @big chungus. An anime girl with medium-length blonde hair is..."
-   Name a character, then describe their basic appearance.
    -   "Digital artwork of Fern from Sousou no Frieren, with long purple hair and purple eyes, wearing a black coat over a white dress with puffy sleeves..."
    -   This is extra important when prompting for multiple characters. If you just list off character names with no description of appearance, the model can get confused.

# [](#model-comparison)Model comparison

You may be interested in comparing Anima's outputs with other models. A ComfyUI workflow, anima\_comparison.json, is provided. This workflow generates a grid of images where each model is a column and the rows are different seeds. It can be configured to compare any number of models you select by changing a few output nodes. Supported model architectures: Anima, SDXL, Lumina, Chroma, Newbie-Image. The default configuration compares Anima, NetaYume, and Newbie-Image.

# [](#limitations)Limitations

-   The model doesn't do realism well. This is intended. It is an anime / illustration / art focused model.
-   The model may generate undesired content, especially if the prompt is short or lacking details.
    -   Avoid this by using the appropriate safety tags in the positive and negative prompts, and by writing sufficiently detailed prompts.
-   The model isn't great at text rendering. It can generally do single words and sometimes short phrases, but lengthy text rendering won't work well.
-   The base version is a true base model. It hasn't been aesthetic tuned on a curated dataset. The default style is very plain and neutral, which is especially apparent if you don't use artist or quality tags.

# [](#finetuning-tips)Finetuning Tips

-   **Don't train the LLM adapter.** My own training script, diffusion-pipe, lets you set llm\_adapter\_lr=0 to completely disable training it, and the example config has this as a default.
    -   Other trainers like sd-scripts have similar options that should be used.
    -   The LLM adapter processes the text embeddings before they get to the diffusion model, and therefore has an outsized influence on the generated images. The adapter itself contains a surprising amount of knowledge and is easy to degrade by training it.
-   Use a low learning rate. For a rank 32 LoRA, start with 2e-5 and adjust up or down from there.
    -   As a base model, there is no aggressive aesthetic tuning or RLHF you need to overcome when finetuning.
    -   The model has an extremely large and diverse amount of visual concepts baked in already. A light touch is all you need.
-   [Example](https://civitai.com/models/2536147) of a style LoRA, with dataset and configs shared.

# [](#license)License

This model is licensed under the CircleStone Labs Non-Commercial License. The model and derivatives are only usable for non-commercial purposes. Additionally, this model constitutes a "Derivative Model" of Cosmos-Predict2-2B-Text2Image, and therefore is subject to the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) insofar as it applies to Derivative Models.

If you would like a commercial license, please email [tdrussell@circlestone.ai](mailto:tdrussell@circlestone.ai)

Built on NVIDIA Cosmos.

Downloads last month

756,861

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

This model isn't deployed by any Inference Provider. [🙋 21 Ask for provider support](/spaces/huggingface/InferenceSupport/discussions/7853)

## Model tree for circlestone-labs/Anima[](/docs/hub/model-cards#specifying-a-base-model)

Adapters

[25 models](/models?other=base_model:adapter:circlestone-labs/Anima)

Finetunes

[41 models](/models?other=base_model:finetune:circlestone-labs/Anima)

Merges

[2 models](/models?other=base_model:merge:circlestone-labs/Anima)

Quantizations

[19 models](/models?other=base_model:quantized:circlestone-labs/Anima)

## Spaces using circlestone-labs/Anima 14

[

🎨

mrfakename/anima-v1

](/spaces/mrfakename/anima-v1)[

🖼

RioShiina/ImageGen

](/spaces/RioShiina/ImageGen)[

🎨

Luminia/Anima-2B-CPU

](/spaces/Luminia/Anima-2B-CPU)[

🔥

rikunarita/Anima-Preview-2

](/spaces/rikunarita/Anima-Preview-2)[

🔥

Bruhn/Comfy-Preview-2

](/spaces/Bruhn/Comfy-Preview-2)[

🔥

rikunarita-2/Anima-base-v1.0

](/spaces/rikunarita-2/Anima-base-v1.0)[

🎨

ASomeoneWhoInterestedWithAI/anima-v1-duplicate

](/spaces/ASomeoneWhoInterestedWithAI/anima-v1-duplicate)[

🎨

mrfakename/anima-v1-fast

](/spaces/mrfakename/anima-v1-fast)\+ 9 Spaces \+ 6 Spaces

System theme

Company

[TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/)[](/)

Website

[Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs)
```

---

## Assistant (Build · MiMo V2.5 Free · 33.7s)

_Thinking:_

Now I have a good understanding of both the Flux.2 Untwisting RoPE codebase and the Anima model architecture. Let me also look at the Anima model source code to understand the specific attention and RoPE implementation details.

**Tool: webfetch**

**Input:**
```json
{
  "format": "text",
  "url": "https://raw.githubusercontent.com/kohya-ss/sd-scripts/308a0cc9/library/anima_models.py"
}
```

**Output:**
```
# Anima Model Architecture
# Original code: NVIDIA CORPORATION & AFFILIATES, licensed under Apache-2.0

import math
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library import custom_offloading_utils, attention


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


# Unsloth Offloaded Gradient Checkpointing
# Based on Unsloth Zoo by Daniel Han-Chen & the Unsloth team
try:
    from deepspeed.runtime.activation_checkpointing.checkpointing import detach_variable
except ImportError:

    def detach_variable(inputs, device=None):
        """Detach tensors from computation graph, optionally moving to a device.

        Reimplementation of deepspeed.runtime.activation_checkpointing.checkpointing.detach_variable
        for environments without DeepSpeed installed.
        """
        if isinstance(inputs, tuple):
            out = []
            for inp in inputs:
                if not isinstance(inp, torch.Tensor):
                    out.append(inp)
                    continue
                requires_grad = inp.requires_grad
                if device is not None:
                    x = inp.to(device=device)
                else:
                    x = inp
                x = x.detach()
                x.requires_grad = requires_grad
                out.append(x)
            return tuple(out)
        else:
            raise RuntimeError(
                "Only tuple of tensors is supported. Got Unsupported input type: ",
                type(inputs).__name__,
            )


class UnslothOffloadedGradientCheckpointer(torch.autograd.Function):
    """Saves VRAM by offloading activations to CPU RAM using non-blocking transfers.

    Compared to standard cpu_offload_checkpointing which uses blocking transfers,
    this uses non_blocking=True to hide CPU<->GPU transfer latency behind compute.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, forward_function, hidden_states, *args):
        # Remember the original device for backward pass (multi-GPU support)
        ctx.input_device = hidden_states.device
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        # NOTE: args stored directly on ctx (not via save_for_backward) because
        # the training loop holds references to these tensors, preventing GC.
        # Using save_for_backward for all args would add complexity for no benefit.
        ctx.args = args
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grads):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to(ctx.input_device, non_blocking=True).detach()
        hidden_states.requires_grad_(True)
        args = detach_variable(ctx.args)
        inputs = (hidden_states,) + args
        with torch.enable_grad():
            outputs = ctx.forward_function(*inputs)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(
            outputs if isinstance(outputs, tuple) else (outputs,), grads if isinstance(grads, tuple) else (grads,)
        ):
            if isinstance(out, torch.Tensor) and out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None,) + tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs)


@torch._disable_dynamo
def unsloth_checkpoint(function, *args):
    """Wrapper for UnslothOffloadedGradientCheckpointer."""
    return UnslothOffloadedGradientCheckpointer.apply(function, *args)


from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Utility functions: RoPE for DiT
def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_base(
    t: torch.Tensor,
    freqs: torch.Tensor,
    start_positions: torch.Tensor = None,
    tensor_format: str = "sbhd",
    interleaved: bool = False,
) -> torch.Tensor:
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    if start_positions is not None:
        max_offset = torch.max(start_positions)
        assert max_offset + cur_seq_len <= max_seq_len, f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
        freqs = torch.concatenate([freqs[i : i + cur_seq_len] for i in start_positions], dim=1)

    assert cur_seq_len <= max_seq_len, f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    freqs = freqs[:cur_seq_len]

    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * cos_) + (_rotate_half(t, interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
    interleaved: bool = False,
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
    cp_size: int = 1,
) -> torch.Tensor:
    assert not (cp_size > 1 and start_positions is not None), "start_positions != None with CP SIZE > 1 is not supported!"

    assert tensor_format != "thd" or cu_seqlens is not None, "cu_seqlens must not be None when tensor_format is 'thd'."

    assert fused == False

    if tensor_format == "thd":
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return torch.cat(
            [
                _apply_rotary_pos_emb_base(
                    x.unsqueeze(1),
                    freqs,
                    start_positions=(start_positions[idx : idx + 1] if start_positions is not None else None),
                    interleaved=interleaved,
                )
                for idx, x in enumerate(torch.split(t, seqlens))
            ]
        ).squeeze(1)

    if tensor_format == "sbhd":
        seqlen = t.size(0)
    elif tensor_format == "bshd":
        seqlen = t.size(1)
    else:
        raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
    return _apply_rotary_pos_emb_base(
        t,
        freqs,
        start_positions,
        tensor_format,
        interleaved=interleaved,
    )


# Basic building blocks
class RMSNorm(torch.nn.Module):
    """RMS Normalization for DiT blocks."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight


class GPT2FeedForward(nn.Module):
    """GELU feedforward network."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


# Attention module for DiT
class Attention(nn.Module):
    """Multi-head attention supporting both self-attention and cross-attention.

    Uses QK-norm (RMSNorm on q/k) and optional RoPE (only for self-attention).
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> tuple:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=False)
            k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=False)

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        attn_params: attention.AttentionParams,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        if q.dtype != v.dtype:
            if (not attn_params.supports_fp32 or attn_params.requires_same_dtype) and torch.is_autocast_enabled():
                # FlashAttention requires fp16/bf16, xformers require same dtype; only cast when autocast is active.
                target_dtype = v.dtype  # v has fp16/bf16 dtype
                q = q.to(target_dtype)
                k = k.to(target_dtype)
        # return self.compute_attention(q, k, v)
        qkv = [q, k, v]
        del q, k, v
        result = attention.attention(qkv, attn_params=attn_params)
        return self.output_dropout(self.output_proj(result))


# Positional Embeddings
class VideoPositionEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def seq_dim(self) -> int:
        return 1

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor]) -> torch.Tensor:
        B_T_H_W_C = x_B_T_H_W_C.shape
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)
        return embeddings

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor]) -> Any:
        raise NotImplementedError


class VideoRopePosition3DEmb(VideoPositionEmb):
    """3D Rotary Position Embedding for video (T, H, W) dimensions."""

    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        self.dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> torch.Tensor:
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert (
                uniform_fps or B == 1 or T == 1
            ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"

            if fps is None:
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self) -> int:
        return 0


class LearnablePosEmbAxis(VideoPositionEmb):
    """Learnable axis-decomposed positional embeddings."""

    def __init__(
        self,
        *,
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"
        self.model_channels = model_channels

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.model_channels)
        torch.nn.init.trunc_normal_(self.pos_emb_h, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_w, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_t, std=std, a=-3 * std, b=3 * std)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, _ = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = (
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
                + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
                + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
        else:
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(1e-6, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return emb / norm.to(emb.dtype)


# Timestep Embedding
class Timesteps(nn.Module):
    """Sinusoidal timestep features."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dtype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    """Projects timestep features to model dimension, with optional AdaLN-LoRA."""

    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb

        return emb_B_T_D, adaln_lora_B_T_3D


# Commented out Fourier Features (not used in Anima). Kept for reference.
# class FourierFeatures(nn.Module):
#     """Fourier feature transform: [B] -> [B, D]."""

#     def __init__(self, num_channels: int, bandwidth: int = 1, normalize: bool = False):
#         super().__init__()
#         self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
#         self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
#         self.gain = np.sqrt(2) if normalize else 1
#         self.bandwidth = bandwidth
#         self.num_channels = num_channels
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         generator = torch.Generator()
#         generator.manual_seed(0)
#         self.freqs = 2 * np.pi * self.bandwidth * torch.randn(self.num_channels, generator=generator).to(self.freqs.device)
#         self.phases = 2 * np.pi * torch.rand(self.num_channels, generator=generator).to(self.freqs.device)

#     def forward(self, x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
#         in_dtype = x.dtype
#         x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
#         x = x.cos().mul(self.gain * gain).to(in_dtype)
#         return x


# Patch Embedding
class PatchEmbed(nn.Module):
    """Patch embedding: (B, C, T, H, W) -> (B, T', H', W', D)"""

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert (
            H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        ), f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


# Final Layer
class FinalLayer(nn.Module):
    """Final layer with AdaLN modulation + unpatchify."""

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False))

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        use_fp32: bool = False,
    ):
        # Compute AdaLN modulation parameters (in float32 when fp16 to avoid overflow in Linear layers)
        with torch.autocast(device_type=x_B_T_H_W_D.device.type, dtype=torch.float32, enabled=use_fp32):
            if self.use_adaln_lora:
                assert adaln_lora_B_T_3D is not None
                shift_B_T_D, scale_B_T_D = (
                    self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
                ).chunk(2, dim=-1)
            else:
                shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
        scale_B_T_1_1_D = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")

        x_B_T_H_W_D = self.layer_norm(x_B_T_H_W_D) * (1 + scale_B_T_1_1_D) + shift_B_T_1_1_D
        x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


# Transformer Block (DiT Block)
class Block(nn.Module):
    """Transformer block with self-attention + cross-attention + MLP, each modulated by AdaLN.

    Each sublayer: x = x + gate * sublayer(norm(x) * (1 + scale) + shift)
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
        )

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False, unsloth_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload if not unsloth_offload else False
        self.unsloth_offload_checkpointing = unsloth_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def _forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention.AttentionParams,
        use_fp32: bool = False,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_fp32:
            # Cast to float32 for better numerical stability in residual connections. Each module will cast back to float16 by enclosing autocast context.
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        # Compute AdaLN modulation parameters (in float32 when fp16 to avoid overflow in Linear layers)
        with torch.autocast(device_type=x_B_T_H_W_D.device.type, dtype=torch.float32, enabled=use_fp32):
            if self.use_adaln_lora:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                    3, dim=-1
                )
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape for broadcasting: (B, T, D) -> (B, T, 1, 1, D)
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _adaln_fn(_x, _norm_layer, _scale, _shift):
            return _norm_layer(_x) * (1 + _scale) + _shift

        # 1. Self-attention
        normalized_x = _adaln_fn(x_B_T_H_W_D, self.layer_norm_self_attn, scale_self_attn_B_T_1_1_D, shift_self_attn_B_T_1_1_D)
        result = rearrange(
            self.self_attn(
                rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                attn_params,
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result

        # 2. Cross-attention
        normalized_x = _adaln_fn(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_cross_attn_B_T_1_1_D, shift_cross_attn_B_T_1_1_D)
        result = rearrange(
            self.cross_attn(
                rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                attn_params,
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = result * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        # 3. MLP
        normalized_x = _adaln_fn(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp_B_T_1_1_D, shift_mlp_B_T_1_1_D)
        result = self.mlp(normalized_x)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result

        return x_B_T_H_W_D

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention.AttentionParams,
        use_fp32: bool = False,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            if self.unsloth_offload_checkpointing:
                # Unsloth: async non-blocking CPU RAM offload (fastest offload method)
                return unsloth_checkpoint(
                    self._forward,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    use_fp32,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D,
                    extra_per_block_pos_emb,
                )
            elif self.cpu_offload_checkpointing:
                # Standard cpu offload: blocking transfers
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        # Determine original device from first tensor input
                        device = next(t.device for t in inputs if isinstance(t, torch.Tensor))
                        device_inputs = to_device(inputs, device)
                        outputs = func(*device_inputs)
                        return to_cpu(outputs)

                    return custom_forward

                return torch_checkpoint(
                    create_custom_forward(self._forward),
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    use_fp32,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D,
                    extra_per_block_pos_emb,
                    use_reentrant=False,
                )
            else:
                # Standard gradient checkpointing (no offload)
                return torch_checkpoint(
                    self._forward,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    use_fp32,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D,
                    extra_per_block_pos_emb,
                    use_reentrant=False,
                )
        else:
            return self._forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                use_fp32,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_per_block_pos_emb,
            )


# Main DiT Model: MiniTrainDIT (renamed to Anima)
class Anima(nn.Module):
    """Cosmos-Predict2 DiT model for image/video generation.

    28 transformer blocks with AdaLN-LoRA modulation, 3D RoPE, and optional LLM Adapter.
    """

    LATENT_CHANNELS = 16

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        use_llm_adapter: bool = False,
        attn_mode: str = "torch",
        split_attn: bool = False,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.use_llm_adapter = use_llm_adapter

        self.attn_mode = attn_mode
        self.split_attn = split_attn

        # Block swap support
        self.blocks_to_swap = None
        self.offloader: Optional[custom_offloading_utils.ModelOffloader] = None

        self.build_patch_embed()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        if self.use_llm_adapter:
            self.llm_adapter = LLMAdapter(
                source_dim=1024,
                target_dim=1024,
                model_dim=1024,
                num_layers=6,
                self_attn=True,
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)
        self.init_weights()

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()
        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()
        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

    def enable_gradient_checkpointing(self, cpu_offload: bool = False, unsloth_offload: bool = False):
        for block in self.blocks:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload, unsloth_offload=unsloth_offload)

    def disable_gradient_checkpointing(self):
        for block in self.blocks:
            block.disable_gradient_checkpointing()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def build_patch_embed(self) -> None:
        in_channels = self.in_channels + 1 if self.concat_padding_mask else self.in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_channels,
            out_channels=self.model_channels,
        )

    def build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(**kwargs)

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        from torchvision import transforms

        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1)
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)

        return x_B_T_H_W_D, None, extra_pos_emb

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap = num_blocks

        assert (
            self.blocks_to_swap <= self.num_blocks - 2
        ), f"Cannot swap more than {self.num_blocks - 2} blocks. Requested: {self.blocks_to_swap} blocks."

        self.offloader = custom_offloading_utils.ModelOffloader(self.blocks, self.blocks_to_swap, device)
        logger.info(f"Anima: Block swap enabled. Swapping {num_blocks} blocks, total blocks: {self.num_blocks}, device: {device}.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # Move all modules to device except blocks (which are managed by offloader)
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None  # Use None to skip .to() on blocks (consistent with flux_models.py)

        self.to(device)

        if self.blocks_to_swap:
            self.blocks = save_blocks

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()
        print(f"Anima: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()
        print(f"Anima: Block swap set to forward and backward.")

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def forward_mini_train_dit(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        t5_input_ids: Optional[torch.Tensor] = None,
        t5_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_B_C_T_H_W: (B, C, T, H, W) noisy latents
            timesteps_B_T: (B,) or (B, T) timesteps
            crossattn_emb: (B, N, D) cross-attention embeddings (or raw Qwen3 prompt_embeds if t5_input_ids provided)
            fps: Optional frames per second
            padding_mask: Optional padding mask
            source_attention_mask: Optional attention mask for Qwen3 embeddings (used with LLM adapter)
            t5_input_ids: Optional T5 token IDs (triggers LLM adapter when provided)
            t5_attn_mask: Optional T5 attention mask
        """

...14757 bytes truncated...

The tool call succeeded but the output was truncated. Full output saved to: /home/kuroko/.local/share/opencode/tool-output/tool_e80a24a47001Zc6W2svLiz9rc2
Use the Task tool to have explore agent process this file with Grep and Read (with offset/limit). Do NOT read the full file yourself - delegate to save context.
```

---

## Assistant (Build · MiMo V2.5 Free · 8.4s)

_Thinking:_

Let me now also look at the ComfyUI Anima model implementation to understand how it's loaded and used there, since this is a ComfyUI custom node.

**Tool: websearch**

**Input:**
```json
{
  "query": "ComfyUI Anima model implementation attn1_patch reference_latents ComfyUI nodes",
  "numResults": 5
}
```

**Output:**
```
{
  "search_id": "search_b01329933c1f4345bf46653bf1f168ba",
  "results": [
    {
      "url": "https://github.com/Comfy-Org/ComfyUI/actions/runs/24363723119",
      "title": "feat: Add reference latent support for Anima · Comfy-Org/ComfyUI@312adfc · GitHub",
      "publish_date": null,
      "excerpts": [
        "## Navigation Menu\n\nToggle navigation\n\nAppearance settings\n\nSearch or jump to...\n\n# Search code, repositories, users, issues, pull requests...\n\nAppearance settings\n\nResetting focus\n\nPull Request CI Workflow Runs\n\n# feat: Add reference latent support for Anima \\\n\nSign in to view logs * Sign in to view logs\n\n## Summary\n\nTriggered via pull request April 13, 2026 19:53\n\n@Kosinkadink Kosinkadink\n\nlabeled \\\n\nStatus Skipped\n\nTotal duration 1s\n\nArtifacts –\n\n## pullrequest-ci-run.yml\n\non: pull\\_request\\_target\n\ncomment comment\n\nMatrix: pr-test-stable\n\n1\njob completed\n\nShow all jobs\n\nOh hello! Nice to see you.\n\nMade with ❤️ by [humans.txt](https://actions.github.io/humans.txt)\n\nFit to window\n\nZoom out\n\nZoom in"
      ]
    },
    {
      "url": "https://docs.comfy.org/built-in-nodes/ReferenceLatent",
      "title": "ReferenceLatent - ComfyUI Built-in Node Documentation",
      "publish_date": null,
      "excerpts": [
        "> ## Documentation Index\n> Fetch the complete documentation index at: https://docs.comfy.org/llms.txt\n> Use this file to discover all available pages before exploring further.\n\n# ReferenceLatent - ComfyUI Built-in Node Documentation\n\n> Complete documentation for the ReferenceLatent node in ComfyUI. Learn its inputs, outputs, parameters and usage.\n\n> This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! [Edit on GitHub](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/ReferenceLatent/en.md)\n\nThis node sets the guiding latent for an edit model. It takes conditioning data and an optional latent input, then modifies the conditioning to include reference latent information. If the model supports it, you can chain multiple ReferenceLatent nodes to set multiple reference images.\n\n## Inputs\n| Parameter      | Data Type    | Required | Range | Description                                                            |\n| -------------- | ------------ | -------- | ----- | ---------------------------------------------------------------------- |\n| `conditioning` | CONDITIONING | Yes      | -     | The conditioning data to be modified with reference latent information |\n| `latent`       | LATENT       | No       | -     | Optional latent data to use as reference for the edit model            |\n\n## Outputs\n\n| Output Name | Data Type    | Description                                                            |\n| ----------- | ------------ | ---------------------------------------------------------------------- |\n| `output`    | CONDITIONING | The modified conditioning data containing reference latent information |"
      ]
    },
    {
      "url": "https://github.com/fulletLab/comfyui-anima-style-nodes",
      "title": "GitHub - fulletLab/comfyui-anima-style-nodes · GitHub",
      "publish_date": null,
      "excerpts": [
        "master\n\nBranches Tags\n\n \n\nGo to file\n\nCode\n\nOpen more actions menu\n\n## Folders and files\n\n|Name |Name |Last commit message |Last commit date |\n| --- | --- | --- | --- |\n|## Latest commit\n\n## History\n\n16 Commits\n\n 16 Commits |\n|assets |assets | | |\n|data |data | | |\n|js |js | | |\n|.gitignore |.gitignore | | |\n|DATA\\_SOURCE.md |DATA\\_SOURCE.md | | |\n|LICENSE |LICENSE | | |\n|README.md |README.md | | |\n|\\_\\_init\\_\\_.py |\\_\\_init\\_\\_.py | | |\n|artist\\_data.py |artist\\_data.py | | |\n|nodes.py |nodes.py | | |\n|pyproject.toml |pyproject.toml | | |\n|requirements.txt |requirements.txt | | |\n|routes.py |routes.py | | |\n|routes\\_core.py |routes\\_core.py | | |\n|routes\\_favorites.py |routes\\_favorites.py | | |\n|routes\\_fullet.py |routes\\_fullet.py | | |\n|View all files |\n\n## Repository files navigation\n\n* README\n* MIT license\n\n# Anima Style Explorer (Anima-2b)\nA quality-of-life node for ComfyUI that adds artist and style browsing, visual selection, and prompt autocomplete directly inside the workflow.\n\n> This is an independent community tool.\n> It does **not** connect to or make requests to the original website.\n> \n> \n\n* * *\n\n## Installation\n\n1. Download or clone this repository\n2. Place the folder inside:\n\n```\nComfyUI/custom_nodes/\n```\n\n1. Restart ComfyUI\n\nThe node will appear as:\n\n```\nAnima Style Explorer\n```\n\n* * *\n\n## What it does\n\nThis node replaces the usual “type and guess” prompt workflow with an interactive style selector.\n\nIt works similarly to **CLIP Text Encode** :\n\n* connect a CLIP model\n* write your prompt\n* send conditioning directly to KSampler\n\nThe node automatically injects the selected `@artist` tag into the prompt and encodes everything in a single step.\n\n* * *\n\n## Basic Workflow\n\n```\nCheckpointLoader\n      |\n   (clip) ──► Anima Style Explorer ──► (conditioning) ──► KSampler (positive)\n|\n                 prompt\n```\n\nNo extra nodes required.\n\n* * *\n\n## Features\n\n### Visual Style Browser\n\nOpen a gallery of 5000+ artists with thumbnails and preview references.\nClick any artist to instantly apply it to the prompt.\n\n### Prompt Autocomplete\n\nType `@` in the prompt box and a live suggestion list appears with previews.\n\n### Random Style\n\nInstantly injects a random artist into your prompt — useful for exploration and inspiration.\n\n### Auto Cycle\n\nAutomatically queues generations while cycling artists one by one.\nGreat for discovering new styles hands-free.\n\n### Update Styles\n\nThe node includes a built-in database of 5,000+ artists. You can manually refresh this data by clicking the **Update Styles** button on the node or the refresh icon in the Browser. This avoids automatic downloads on startup.\n\n* * *\n\n## Credits\n\nStyle explorer and dataset concept by @ThetaCursed <https://thetacursed.github.io/Anima-Style-Explorer>"
      ]
    },
    {
      "url": "https://github.com/GumGum10/comfyui-qwen35-anima",
      "title": "GitHub - GumGum10/comfyui-qwen35-anima: ComfyUI custom node: Qwen 3.5 4B hybrid text encoder for Anima 2B · GitHub",
      "publish_date": null,
      "excerpts": [
        "- [Premium Support Enterprise-grade 24/7 support](https://github.com/premium-support)\n* [Pricing](https://github.com/pricing)\n\nSearch or jump to...\n\n# Search code, repositories, users, issues, pull requests...\n\nSearch\n\nClear\n\n[Search syntax tips](https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax)\n\nAppearance settings\n\nResetting focus\n\nYou signed in with another tab or window. Reload to refresh your session. You signed out in another tab or window. Reload to refresh your session. You switched accounts on another tab or window. Reload to refresh your session. Dismiss alert\n\nGumGum10 / **comfyui-qwen35-anima** Public\n\n* Notifications You must be signed in to change notification settings\n* Fork 1\n* Star 8\n\n* Code\n* Issues 1\n* Pull requests 0\n* Actions\n* Projects\n* Security 0\n* Insights\n\nAdditional navigation options * Code\n* Issues\n* Pull requests\n* Actions\n* Projects\n* Security\n* Insights\n\n# GumGum10/comfyui-qwen35-anima\n\nmaster\nBranches Tags\n\n \n\nGo to file\n\nCode\n\nOpen more actions menu\n\n## Folders and files\n\n|Name |Name |Last commit message |Last commit date |\n| --- | --- | --- | --- |\n|## Latest commit\n\n## History\n\n7 Commits\n\n 7 Commits |\n|.gitignore |.gitignore | | |\n|README.md |README.md | | |\n|\\_\\_init\\_\\_.py |\\_\\_init\\_\\_.py | | |\n|View all files |\n\n## Repository files navigation\n\n* README\n\n# ComfyUI Qwen 3.5 4B Text Encoder for Anima 2B\n\nA custom ComfyUI node that adds support for the **Qwen 3.5 4B** hybrid (Mamba2 + Attention) text encoder for use with the **Anima 2B** diffusion model.\n\nThe base Anima 2B ships with a Qwen 3 0.6B text encoder. This node enables the larger Qwen 3.5 4B variant from [cosmos-qwen3.5](https://huggingface.co/nightknocker/cosmos-qwen3.5/tree/main/4b) , which uses a hybrid SSM/attention architecture for improved text understanding.\n\n## Architecture\n\nQwen 3.\n5 4B is **not** a standard transformer — it's a hybrid model alternating between Mamba2-style selective state space (SSM) blocks and gated self-attention:\n\n* **32 layers** total: 24 SSM + 8 self-attention (at positions 3, 7, 11, 15, 19, 23, 27, 31)\n* **Hidden size** : 2560, **Output dim** : 1024 (matching Anima's expected embedding size)\n* **Vocab** : 248,320 tokens\n* **FP8 quantized** (F8\\_E4M3) weights with BF16 norms\n\n## Installation\n\nAll files are available at: **[lylogummy/anima2b-qwen-3.5-4b](https://huggingface.co/lylogummy/anima2b-qwen-3.5-4b)**\n\n1. Clone this repo into your ComfyUI `custom_nodes` directory:\n   \n   ```\n   cd ComfyUI/custom_nodes\n   git clone https://github.com/GumGum10/comfyui-qwen35-anima.git\n   ```\n2. Download `qwen35_4b.safetensors` from [text\\_encoders/](https://huggingface.co/lylogummy/anima2b-qwen-3.5-4b/tree/main/text_encoders) and place it in:\n   \n   ```\n   ComfyUI/models/text_encoders/qwen35_4b.safetensors\n   ```\n3. Download `calibration_params."
      ]
    },
    {
      "url": "https://github.com/shootthesound/comfyui-ReferenceLatentPlus",
      "title": "GitHub - shootthesound/comfyui-ReferenceLatentPlus: Improved ReferenceLatent for ComfyUI — per-image strength, timestep gating, MediaPipe auto-masks, 1-4 image inputs. Works with Flux, Klein, Lumina, Wan, Hunyuan, Qwen-Image. · GitHub",
      "publish_date": null,
      "excerpts": [
        "- [Premium Support Enterprise-grade 24/7 support](https://github.com/premium-support)\n* [Pricing](https://github.com/pricing)\n\nSearch or jump to...\n\n# Search code, repositories, users, issues, pull requests...\n\nSearch\n\nClear\n\n[Search syntax tips](https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax)\n\nAppearance settings\n\nResetting focus\n\nYou signed in with another tab or window. Reload to refresh your session. You signed out in another tab or window. Reload to refresh your session. You switched accounts on another tab or window. Reload to refresh your session. Dismiss alert\n\nshootthesound / **comfyui-ReferenceLatentPlus** Public\n\n* Notifications You must be signed in to change notification settings\n* Fork 2\n* Star 18\n\n* Code\n* Issues 1\n* Pull requests 0\n* Actions\n* Projects\n* Security and quality 0\n* Insights\n\nAdditional navigation options * Code\n* Issues\n* Pull requests\n* Actions\n* Projects\n* Security and quality\n* Insights\n# shootthesound/comfyui-ReferenceLatentPlus\n\nmain\n\nBranches Tags\n\n \n\nGo to file\n\nCode\n\nOpen more actions menu\n\n## Folders and files\n\n|Name |Name |Last commit message |Last commit date |\n| --- | --- | --- | --- |\n|## Latest commit\n\n## History\n\n10 Commits\n\n 10 Commits |\n|examples |examples | | |\n|web |web | | |\n|.gitignore |.gitignore | | |\n|README.md |README.md | | |\n|\\_\\_init\\_\\_.py |\\_\\_init\\_\\_.py | | |\n|mediapipe\\_masker.py |mediapipe\\_masker.py | | |\n|nodes.py |nodes.py | | |\n|screenshot.png |screenshot.png | | |\n|View all files |\n\n## Repository files navigation\n\n* README\n\n# Reference Latent+ for ComfyUI\n\nA drop-in replacement for ComfyUI's stock `ReferenceLatent` node, with the controls I always wished it had.  \nPer-image strength, per-image timestep gating, per-image MediaPipe auto-masks, megapixel cap on attention cost, and 1–4 image inputs in a single node.\n\n**Works with Flux, Flux2/Klein, Lumina2, Z-Image, Wan, Hunyuan, Qwen-Image** — anything that consumes `reference_latents` .\n[](https://buymeacoffee.com/lorasandlenses)\n\n* * *\n\n### Why I built this\n\nThe stock `ReferenceLatent` is a tiny wrapper — it just attaches a latent to the conditioning's `reference_latents` key and walks away. That's fine for the obvious case (drop in a ref, get an output), but the moment I tried to push it I hit walls:\n\n* No way to dial strength up or down — it's binary on/off.\n* No timestep gating — the ref is active for the whole denoising schedule whether I want it or not.\n* The ref's _whole image_ leaks into output: pose, framing, background, even crop. There's no \"use the face only, ignore the rest\" knob.\n* Multi-ref means wiring multiple `LoadImage → VAE Encode → ReferenceLatent` chains by hand.\n* Bigger refs = quadratic attention cost, with no easy way to cap it.\n\nI built **Reference Latent+** as the node I wished existed. Each image gets its own strength, its own timestep range, and its own auto-mask config. Slots reveal progressively as you connect them."
      ]
    },
    {
      "url": "https://ltdrdata.github.io/",
      "title": "ComfyUI Nodes Info",
      "publish_date": null,
      "excerpts": [
        "**ComfyUI Nodes Info**  \n**GitHub Repositories:** [ [](https://github.com/comfyanonymous/ComfyUI) ComfyUI]\n \n[ [](https://github.com/ltdrdata/ComfyUI-Manager) ComfyUI-Manager]\n \n[ [](https://github.com/ltdrdata/ltdrdata.github.io) ltdrdata.github.io]\n\nType: All ID Author Title Description Nodes  \nToggle Theme  \n  \n|ID |Author |Title |Reference |Description |\n| --- | --- | --- | --- | --- |"
      ]
    },
    {
      "url": "https://comfyai.run/documentation/ReferenceLatentList",
      "title": "ReferenceLatentList Node Documentation (ComfyUI-FramePackWrapper_PlusOne)",
      "publish_date": null,
      "excerpts": [
        "ComfyAI Platform Update: You can now run ComfyUI workflows instantly on [ComfyAI Cloud](https://comfyai.run)\n\n|Run Community Workflow - no setup needed |Host and Share Your Workflow - 24/7 Uptime |\n| --- | --- |\n|**SD:** [txt2img](https://comfyai.run/workflow/01915eb0-3e97-48f1-b549-e6a4d41c9f09) , [img2img](https://comfyai.run/workflow/0f9ff6a0-eb06-42e3-ab60-34c894fecf53) , [ControlNet](https://comfyai.run/workflow/b7379603-432b-475e-9291-0b9d57e551c2) , [Lora](https://comfyai.run/workflow/d9f5483f-ef4c-4914-87b1-5f01d71943b7)  \n**Flux:** [txt2img](https://comfyai.run/workflow/5545f035-499b-4048-9fef-edb7055ebc4f) , [ControlNet](https://comfyai.run/workflow/4bf52d3d-2425-48f3-9419-9e36dd78cbc5) , [Lora](https://comfyai.run/workflow/ecc8faeb-4efe-4022-b4ce-f490933c0a88)  \n**WAN:** [txt2vid](https://comfyai.run/workflow/f436cb23-f0be-40ef-95cc-3a64c2ed3d63) , [img2vid](https://comfyai.run/workflow/ed642f50-1602-4b33-bd34-31b80ec4b696) |[Start for Free Today](https://comfyai.run) |\nComfyUI-FramePackWrapper\\_PlusOne\n\n# Reference Latent List Node in ComfyUI\n\n## What is this node?\n\nThe Reference Latent List node in ComfyUI is a versatile node designed to combine up to eight reference latents into a single list. This functionality is particularly useful for users engaging in creative tasks like style transfer, where multiple latent images are required. By inputting various latents, users can generate a comprehensive reference list, thereby streamlining workflows for multi-reference Kisekaeichi modes or other image generation processes.\n\n## Real-world Use-Cases\n\nThis node is especially crucial for artists and developers who work with machine learning models for image synthesis. For instance, users aiming to merge different artistic styles into a single output can leverage the Reference Latent List node. Another use case is for AI-powered design studios that process multiple latent attributes to generate customized content consistently.\n## Example Input: High-level explanation on what this node tries to accomplish with these example inputs\n\nThe Reference Latent List node compiles multiple latent inputs to create a comprehensive reference list for further model processing. The individual latent images supplied should each offer unique aspects for style transfer help.\n\n|Field |Example Values |Explanation |\n| --- | --- | --- |\n|reference\\_1 |Landscape\\_Style |The first reference latent. |\n|reference\\_2 |Portrait\\_Style |The second reference latent. |\n|reference\\_3 |Cartoon\\_Style |The third reference latent. |\n|reference\\_4 |Abstract\\_Style |The fourth reference latent. |\n|reference\\_5 |Vintage\\_Style |The fifth reference latent. |\n|reference\\_6 |Modern\\_Art\\_Style |The sixth reference latent. |\n|reference\\_7 |Grunge\\_Style |The seventh reference latent. |\n|reference\\_8 |SciFi\\_Style |The eighth reference latent. |\n\n## Example Output: High-level explanation, focusing on key output"
      ]
    },
    {
      "url": "https://civitai.com/models/128556/comfyui-inpaint-workflow",
      "title": "ComfyUI inpaint workflow - Anima v0.9 | Other Workflows | Civitai",
      "publish_date": "2023-08-14",
      "excerpts": [
        "Pro\n\nCreate\n\nCreate\n\n# ComfyUI inpaint workflow\n\n866\n\n18\\.2k\n\n930\n\n1\\.9k\n\n[](https://education.civitai.com/civitais-guide-to-resource-types/)\n\nUpdated: May 26, 2026\n\ntool inpaint comfyui comfy\n\nAnima v0.9\n\nXL V2.15-maskopacity\n\nXL V2.12-colormatch\n\nXL V2fix\n\nXL V2\n\nXL 1.1\n\n1\\.5 V3.1\n\n1\\.5 V3.05\n\n1\\.5 V2.02\n\nv2.02\\_ALT\\_V2\n\nv2.02\\_ALT\n\nv2.0\n\nv1.09\n\nv1.08\n\nV1.0\n\nDownload\n\n1 variant available\n\nArchive Other\n\n9\\.62 KB\n\nVerified: a day ago\n\nDownload (9.62 KB)\n\nDetails\n\nType\n\nWorkflows\n\nStats\n\n93 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9\n\nReviews\n\nPositive\n\n(14)\n\nPublished\n\nMay 26, 2026\n\nBase Model\n\nOther\n\nHash\n\nAutoV2\n\n516974DB33\n\nAbout this version\n\n\\-Initial version, converted XL inpainting workflow into Anima\n\nFirst Birthday Badge\n\nPotatCat's Avatar PotatCat\n\nFollow\n\n# Anima workflow\n\n26/05/2026 - v0.9 initial release\n\n* Converted XL workflow to work with anima. It works pretty much same as XL one, guide is relevant on that part.\n\n# Inpaint workflow V2\n## GUIDE FOR V2 WORKFLOW IS LOCATED [HERE](https://civitai.com/articles/9971)\n\n06/12/2025\n\n\\-Added ability to control mask opacity.\n\n\\-Changed sampler to prevent problems with generation.\n\n\\-Adjusted guide.\n\n04/12/2025\n\n\\-Added ability to do colormatch. Changing colormatch setting to 1 will try to match the color of the image.This is good if using VAE that changes color between encode/decode cycle or using different lora/model.\n\n* * *\n\nOlder:\n\n\\-Added differential diffusion\n\n\\-Cleaned the workflow\n\n\\-Created guide which can be found [HERE](https://civitai.com/articles/9971)\n\n\\-This should work with 1.5 too, did not test.\n\n# Inpaint workflow XL V1.1\n\nDue to request updated to work with XL. It is somewhat barebones compared to 1.5. Does not have segmentation or controlnet models currently.\n\nI have a bit outdated comfyui, let me know if it is throwing some errors.\n\n*[a day ago]: 2026-05-26T14:24:42+00:00"
      ]
    },
    {
      "url": "https://civitai.com/models/2502969/anima-lora-trainer-for-comfyui",
      "title": "Anima LoRA Trainer for ComfyUI - v1.0 | Anima Workflows | Civitai",
      "publish_date": "2026-05-15",
      "excerpts": [
        "Pro\n\nCreate\n\nCreate\n\n# Anima LoRA Trainer for ComfyUI\n\n54\n\n489\n\n18\n\n100\n\n[](https://education.civitai.com/civitais-guide-to-resource-types/)\n\nUpdated: May 15, 2026\n\ntool training comfyui lora anima kohya \\+ 1\n\nv1.0\n\nDownload\n\n1 variant available\n\nArchive Other\n\n50\\.95 KB\n\nVerified: 13 days ago\n\nDownload (50.95 KB)\n\nDetails\n\nType\n\nWorkflows\n\nStats\n\n489 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9\n\nReviews\n\nVery Positive\n\n(54)\n\nPublished\n\nMay 15, 2026\n\nBase Model\n\nAnima\n\nHash\n\nAutoV2\n\n8AC92CA779\n\nAbout this version\n\nAdded LoKR & LoHA parameters\n\ndefault creator card background decoration\n\nSt Patrick's Day Badge\n\n7err4's Avatar 7err4\n\nFollow\n\nThe Anima Model is licensed by CircleStone Labs LLC. Copyright CircleStone Labs LLC. IN NO EVENT SHALL CIRCLESTONE LABS LLC BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.\n\nBuilt on NVIDIA Cosmos\nA ComfyUI custom node pack for training LoRA / LoKR / LoHA models on the Anima architecture using kohya-ss/sd-scripts. Train Anima LoRAs directly from ComfyUI with zero command-line knowledge required.\n\n**Installation**\n\nDownload and extract the zip into your `ComfyUI/custom_nodes/` folder, then restart ComfyUI. The nodes will appear under the \"Anima Training\" category.\n\n**Features**\n\n* One-click model downloading from HuggingFace (always fetches the latest Anima base model)\n* Two workflow modes: Wizard (questionnaire-driven auto-config) and Manual (full control over all parameters)\n* Automatic sd-scripts installation and dependency management\n* TOML config generation with inline annotations explaining every setting\n* Smart VRAM presets (12GB, 16GB, 24GB) with automatic memory optimization\n* Live config preview inside ComfyUI before launching training\n* Cross-platform support (Windows, Linux, macOS)\n* Zero pip dependencies — pure Python stdlib\n\n**Included Nodes**\n* Model Downloader — Downloads Anima models from HuggingFace\n* SD-Scripts Manager — Installs/updates kohya-ss/sd-scripts\n* Model Selector — Finds and validates model files\n* Training Wizard — Questionnaire-driven auto-configuration\n* Basic/Advanced/Memory Params — Manual parameter control\n* Training Launcher — Generates config and launches training\n* Config Preview — Shows annotated TOML config inline\n\n**Requirements**\n\n* ComfyUI (any recent version)\n* Python 3.10+\n* kohya-ss/sd-scripts (auto-installed by the SD-Scripts Manager node)\n* Anima base model files (auto-downloaded by the Model Downloader node)\n\n**Dataset Preparation**\n\nYour training images should follow the sd-scripts convention: place images in a folder named `<repeats>_<concept>` inside your `train_data_dir` , with matching `.txt` caption files for each image.\n\n*[13 days ago]: 2026-05-15T06:28:38+00:00"
      ]
    },
    {
      "url": "https://www.comfyonline.app/comfyui-nodes/nodes/StyleAlignedSampleReferenceLatents",
      "title": "StyleAlignedSampleReferenceLatents | ComfyOnline",
      "publish_date": null,
      "excerpts": [
        "Implementation of the a/StyleAligned paper for ComfyUI. This node allows you to apply a consistent style to all images in a batch; by default it will use the first image in the batch as the style reference, forcing all other images to be consistent with it."
      ]
    }
  ],
  "warnings": null,
  "usage": [
    {
      "name": "sku_search",
      "count": 1
    }
  ],
  "session_id": "ses_17f67f5beffeKq0EJMHZjfupbo"
}
```

---

## Assistant (Build · MiMo V2.5 Free · 7.0s)

_Thinking:_

Now I have a comprehensive understanding of both the Flux.2 Untwisting RoPE implementation and the Anima model architecture. Let me now look at how ComfyUI handles Anima's attention and whether there's an `attn1_patch` hook available for Anima models, and how reference latents work with Anima.

**Tool: websearch**

**Input:**
```json
{
  "query": "ComfyUI Anima attn1_patch transformer_options model_function_wrapper diffusion_models.py",
  "numResults": 5
}
```

**Output:**
```
{
  "search_id": "search_ede76e3f542d46ab9930617ab58d9509",
  "results": [
    {
      "url": "https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_inference_low_vram/anima-preview.py",
      "title": "DiffSynth-Studio/examples/anima/model_inference_low_vram/anima-preview.py at main · modelscope/DiffSynth-Studio · GitHub",
      "publish_date": null,
      "excerpts": [
        "## Navigation Menu\n\nToggle navigation\n\nAppearance settings\n\nSearch or jump to...\n\n# Search code, repositories, users, issues, pull requests...\n\nAppearance settings\n\nResetting focus\n\nmodelscope / **DiffSynth-Studio** Public\n\n* Notifications You must be signed in to change notification settings\n* Fork 1\\.2k\n* Star 12\\.5k\n\n## Files Expand file tree\n\nmain\n\n## Breadcrumbs\n\n1. DiffSynth-Studio\n2. / examples\n3. / anima\n4. / model\\_inference\\_low\\_vram\n\n/\n\n# anima-preview.py\n\nCopy path\n\nMore file actions\n\nMore file actions\n\n## Latest commit\n\n## History\n\nHistory\n\n History\n\n30 lines (28 loc) · 1.55 KB\n\nmain\n\n## Breadcrumbs\n\n1. DiffSynth-Studio\n2. / examples\n3. / anima\n4. / model\\_inference\\_low\\_vram\n\n/\n\n# anima-preview.py\n\nTop\n\n## File metadata and controls\n\n* Code\n* Blame\n\n30 lines (28 loc) · 1.55 KB\n\n[Raw](https://github.com/modelscope/DiffSynth-Studio/raw/refs/heads/main/examples/anima/model_inference_low_vram/anima-preview.py)\n\nCopy raw file\n\nDownload raw file\n\nOpen symbols panel\nEdit and raw actions\n\n1\n\n2\n\n3\n\n4\n\n5\n\n6\n\n7\n\n8\n\n9\n\n10\n\n11\n\n12\n\n13\n\n14\n\n15\n\n16\n\n17\n\n18\n\n19\n\n20\n\n21\n\n22\n\n23\n\n24\n\n25\n\n26\n\n27\n\n28\n\n29\n\n30\n\nfrom diffsynth . pipelines . anima\\_image import AnimaImagePipeline , ModelConfig\n\nimport torch\n\nvram\\_config = {\n\n\"offload\\_dtype\" : \"disk\" ,\n\n\"offload\\_device\" : \"disk\" ,\n\n\"onload\\_dtype\" : \"disk\" ,\n\n\"onload\\_device\" : \"disk\" ,\n\n\"preparing\\_dtype\" : torch . bfloat16 ,\n\n\"preparing\\_device\" : \"cuda\" ,\n\n\"computation\\_dtype\" : torch . bfloat16 ,\n\n\"computation\\_device\" : \"cuda\" ,\n\n}\n\npipe = AnimaImagePipeline . from\\_pretrained (\n\ntorch\\_dtype = torch . bfloat16 ,\n\ndevice = \"cuda\" ,\n\nmodel\\_configs = [\n\nModelConfig ( model\\_id = \"circlestone-labs/Anima\" , origin\\_file\\_pattern = \"split\\_files/diffusion\\_models/anima-preview.safetensors\" , \\*\\* vram\\_config ),\n\nModelConfig ( model\\_id = \"circlestone-labs/Anima\" , origin\\_file\\_pattern = \"split\\_files/text\\_encoders/qwen\\_3\\_06b\\_base.safetensors\" , \\*\\* vram\\_config ),\nModelConfig ( model\\_id = \"circlestone-labs/Anima\" , origin\\_file\\_pattern = \"split\\_files/vae/qwen\\_image\\_vae.safetensors\" , \\*\\* vram\\_config ),\n\n],\n\ntokenizer\\_config = ModelConfig ( model\\_id = \"Qwen/Qwen3-0.6B\" , origin\\_file\\_pattern = \"./\" ),\n\ntokenizer\\_t5xxl\\_config = ModelConfig ( model\\_id = \"stabilityai/stable-diffusion-3.5-large\" , origin\\_file\\_pattern = \"tokenizer\\_3/\" ),\n\nvram\\_limit = torch . cuda . mem\\_get\\_info ( \"cuda\" )[ 1 ] / ( 1024 \\*\\* 3 ) \\- 0\\.5 ,\n\n)\n\nprompt = \"Masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait.\"\n\nnegative\\_prompt = \"worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,\"\n\nimage = pipe ( prompt , seed = 0 , num\\_inference\\_steps = 50 )\n\nimage . save ( \"image.jpg\" )\n\n## Footer"
      ]
    },
    {
      "url": "https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-preview.safetensors",
      "title": "split_files/diffusion_models/anima-preview.safetensors · circlestone-labs/Anima at main",
      "publish_date": null,
      "excerpts": [
        "# \n\ncirclestone-labs\n\n/\n\nAnima\n\nlike 1\\.46k\n\nFollow\n\nCircleStone Labs 713\n\nDiffusion Single File comfyui\n\nLicense: circlestone-labs-non-commercial-license\n\nModel card Files Files and versions xet Community 165\n\nCopy to bucket new\n\nUse this model\n\n### Instructions to use circlestone-labs/Anima with libraries, inference providers, notebooks, and local apps. Follow these links to get started.\n\n* Libraries\n* Diffusion Single File\n  \n  How to use circlestone-labs/Anima with Diffusion Single File:\n  \n  ```\n  # No code snippets available yet for this library.\n  \n  # To use this model, check the repository files and the library's documentation.\n  \n  # Want to help? PRs adding snippets are welcome at:\n  # https://github.com/huggingface/huggingface.js\n  ```\n* Notebooks\n* Google Colab\n* Kaggle\n\nmain\n\nAnima / split\\_files / diffusion\\_models / anima-preview.safetensors\n\ntdrussell's picture\n\ntdrussell\n\nRename anima-preview.safetensors to split\\_files/diffusion\\_models/anima-preview.safetensors\n2a33fa2 verified 4 months ago\n\ndownload\n\nCopy download link\n\nhistory blame contribute delete\n\nSafe\n\n4\\.18 GB\n\nThis file is stored with [Xet](https://huggingface.co/docs/hub/xet/index) . It is too big to display, but you can still download it.\n\n### Xet Pointer Details\n\n( Raw pointer file )\n\nXet hash:\n    7b5f82c130fd06400951ee77bbfc6cfb8e12b4f62602ae925880926c0f1ddb19\n\nSize of remote file:\n    4\\.18 GB\n\n·\n\nSHA256:\n    41fa7b78613dfe0d888b3647f70c7fb8cdfda2ff177e78d2e16e06dc810d9dcc\n\nXet efficiently stores Large Files inside Git, intelligently splitting files into unique chunks and\naccelerating uploads and downloads. More info .\n\n| | |\n| --- | --- |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |\n| | |"
      ]
    },
    {
      "url": "https://huggingface.co/spaces/multimodalart/flux-style-shaping/blob/main/comfy/model_patcher.py",
      "title": "comfy/model_patcher.py · multimodalart/flux-style-shaping at main",
      "publish_date": null,
      "excerpts": [
        "model\\_options, pre\\_cfg\\_function, disable\\_cfg1\\_optimization) |\n| | |\n| |def set\\_model\\_unet\\_function\\_wrapper ( self, unet\\_wrapper\\_function: UnetWrapperFunction ): |\n| |self.model\\_options[ \"model\\_function\\_wrapper\" ] = unet\\_wrapper\\_function |\n| | |\n| |def set\\_model\\_denoise\\_mask\\_function ( self, denoise\\_mask\\_function ): |\n| |self.model\\_options[ \"denoise\\_mask\\_function\" ] = denoise\\_mask\\_function |\n| | |\n| |def set\\_model\\_patch ( self, patch, name ): |\n| |to = self.model\\_options[ \"transformer\\_options\" ] |\n| |if \"patches\" not in to: |\n| |to[ \"patches\" ] = {} |\n| |to[ \"patches\" ][name] = to[ \"patches\" ].get(name, []) + [patch] |\n| | |\n| |def set\\_model\\_patch\\_replace ( self, patch, name, block\\_name, number, transformer\\_index= None ): |\n| |self.model\\_options = set\\_model\\_options\\_patch\\_replace(self.model\\_options, patch, name, block\\_name, number, transformer\\_index=transformer\\_index) |\n| | |\n| |def set\\_model\\_attn1\\_patch ( self, patch ): |\n| |self.\nset\\_model\\_patch(patch, \"attn1\\_patch\" ) |\n| | |\n| |def set\\_model\\_attn2\\_patch ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"attn2\\_patch\" ) |\n| | |\n| |def set\\_model\\_attn1\\_replace ( self, patch, block\\_name, number, transformer\\_index= None ): |\n| |self.set\\_model\\_patch\\_replace(patch, \"attn1\" , block\\_name, number, transformer\\_index) |\n| | |\n| |def set\\_model\\_attn2\\_replace ( self, patch, block\\_name, number, transformer\\_index= None ): |\n| |self.set\\_model\\_patch\\_replace(patch, \"attn2\" , block\\_name, number, transformer\\_index) |\n| | |\n| |def set\\_model\\_attn1\\_output\\_patch ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"attn1\\_output\\_patch\" ) |\n| | |\n| |def set\\_model\\_attn2\\_output\\_patch ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"attn2\\_output\\_patch\" ) |\n| | |\n| |def set\\_model\\_input\\_block\\_patch ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"input\\_block\\_patch\" ) |\n| | |\n| |def set\\_model\\_input\\_block\\_patch\\_after\\_skip ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"input\\_block\\_patch\\_after\\_skip\" ) |\n| | |\n| |def set\\_model\\_output\\_block\\_patch ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"output\\_block\\_patch\" ) |\n| | |\n| |def add\\_object\\_patch ( self, name, obj ): |\n| |self.object\\_patches[name] = obj |\n| | |\n| |def get\\_model\\_object ( self, name ): |\n| |if name in self.object\\_patches: |\n| |return self.object\\_patches[name] |\n| |else : |\n| |if name in self.object\\_patches\\_backup: |\n| |return self.object\\_patches\\_backup[name] |\n| |else : |\n| |return comfy.utils.get\\_attr(self.model, name) |\n| | |\n| |def model\\_patches\\_to ( self, device ): |\n| |to = self.model\\_options[ \"transformer\\_options\" ] |\n| |if \"patches\" in to: |\n| |patches = to[ \"patches\" ] |\n| |for name in patches: |\n| |patch\\_list = patches[name] |\n| |for i in range ( len (patch\\_list)): |\n| |if hasattr (patch\\_list[i], \"to\" ): |\n| |patch\\_list[i] = patch\\_list[i]."
      ]
    },
    {
      "url": "https://huggingface.co/spideyrim/ComfyUI/blob/main/comfy/model_patcher.py",
      "title": "comfy/model_patcher.py · spideyrim/ComfyUI at main - Hugging Face",
      "publish_date": null,
      "excerpts": [
        "model\\_options[ \"sampler\\_cfg\\_function\" ] = lambda args: sampler\\_cfg\\_function(args[ \"cond\" ], args[ \"uncond\" ], args[ \"cond\\_scale\" ]) \\ way |\n| |else : |\n| |self.model\\_options[ \"sampler\\_cfg\\_function\" ] = sampler\\_cfg\\_function |\n| | |\n| |def set\\_model\\_unet\\_function\\_wrapper ( self, unet\\_wrapper\\_function ): |\n| |self.model\\_options[ \"model\\_function\\_wrapper\" ] = unet\\_wrapper\\_function |\n| | |\n| |def set\\_model\\_patch ( self, patch, name ): |\n| |to = self.model\\_options[ \"transformer\\_options\" ] |\n| |if \"patches\" not in to: |\n| |to[ \"patches\" ] = {} |\n| |to[ \"patches\" ][name] = to[ \"patches\" ].get(name, []) + [patch] |\n| | |\n| |def set\\_model\\_patch\\_replace ( self, patch, name, block\\_name, number ): |\n| |to = self.model\\_options[ \"transformer\\_options\" ] |\n| |if \"patches\\_replace\" not in to: |\n| |to[ \"patches\\_replace\" ] = {} |\n| |if name not in to[ \"patches\\_replace\" ]: |\n| |to[ \"patches\\_replace\" ][name] = {} |\n| |to[ \"patches\\_replace\" ][name][(block\\_name, number)] = patch |\n| | |\n| |def set\\_model\\_attn1\\_patch ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"attn1\\_patch\" ) |\n| | |\n| |def set\\_model\\_attn2\\_patch ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"attn2\\_patch\" ) |\n| | |\n| |def set\\_model\\_attn1\\_replace ( self, patch, block\\_name, number ): |\n| |self.set\\_model\\_patch\\_replace(patch, \"attn1\" , block\\_name, number) |\n| | |\n| |def set\\_model\\_attn2\\_replace ( self, patch, block\\_name, number ): |\n| |self.set\\_model\\_patch\\_replace(patch, \"attn2\" , block\\_name, number) |\n| | |\n| |def set\\_model\\_attn1\\_output\\_patch ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"attn1\\_output\\_patch\" ) |\n| | |\n| |def set\\_model\\_attn2\\_output\\_patch ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"attn2\\_output\\_patch\" ) |\n| | |\n| |def set\\_model\\_output\\_block\\_patch ( self, patch ): |\n| |self.set\\_model\\_patch(patch, \"output\\_block\\_patch\" ) |\n| | |\n| |def model\\_patches\\_to ( self, device ): |\n| |to = self.model\\_options[ \"transformer\\_options\" ] |\n| |if \"patches\" in to: |\n| |patches = to[ \"patches\" ] |\n| |for name in patches: |\n| |patch\\_list = patches[name] |\n| |for i in range ( len (patch\\_list)): |\n| |if hasattr (patch\\_list[i], \"to\" ): |\n| |patch\\_list[i] = patch\\_list[i].to(device) |\n| |if \"patches\\_replace\" in to: |\n| |patches = to[ \"patches\\_replace\" ] |\n| |for name in patches: |\n| |patch\\_list = patches[name] |\n| |for k in patch\\_list: |\n| |if hasattr (patch\\_list[k], \"to\" ): |\n| |patch\\_list[k] = patch\\_list[k].to(device) |\n| | |\n| |def model\\_dtype ( self ): |\n| |if hasattr (self.model, \"get\\_dtype\" ): |\n| |return self.model.get\\_dtype() |\n| | |\n| |def add\\_patches ( self, patches, strength\\_patch= 1\\.0 , strength\\_model= 1\\.0 ): |\n| |p = set () |\n| |for k in patches: |\n| |if k in self.model\\_keys: |\n| |p.add(k) |\n| |current\\_patches = self.patches.get(k, []) |\n| |current\\_patches."
      ]
    },
    {
      "url": "https://gigazine.net/gsc_news/en/20260515-anima-image-generation-ai",
      "title": "The official version of 'Anima,' an AI image generation tool specializing in anime and illustrations, has finally been released. It supports both tags and natural language processing and can easily run locally on any PC that can handle SDXL or Illustrious-type models. - GIGAZINE",
      "publish_date": null,
      "excerpts": [
        "' Now that the official version has been released, we've put together a summary of the steps for actually generating images with ComfyUI and some tips for prompting.  \n  \n**circlestone-labs/Anima · Hugging Face**\n\n<https://huggingface.co/circlestone-labs/Anima>  \n  \nAnima is an image generation model specializing in anime and illustrations, developed in collaboration with CircleStone Labs and Comfy Org. 'Anime Preview' was released in January 2026, followed by 'Preview2' in March 2026, and 'Preview3 Base' in April 2026, and it has been steadily updated ever since. Even in the preview stage, it was praised for features such as 'the ability to enter prompts in natural language,' 'high prompt tracking performance,' 'works on PCs that can handle SDXL models,' and ' **[unrestricted](https://dic.nicovideo.jp/t/a/%E5%8F%A1%E6%99%BA)** expressiveness,' and it became a hot topic in the local image generation community.\nThe official version of this model, 'Anima-Base,' was released on May 15, 2026.  \n  \nThe procedure for generating images using Anima-Base is as follows. This time, we will generate them using **[ComfyUI](https://gigazine.net/news/20260201-comfyui-image-video-generation-ai/)** .  \n  \nFirst, download the main model file ' **anima-base-v1.0.safetensors** ', the text encoder ' **qwen\\_3\\_06b\\_base.safetensors** ' which handles the bridging of text and images, and ' **qwen\\_image\\_vae.safetensors** ' which handles the conversion process between human images and AI images. Each file is stored in 'diffusion\\_models', 'text\\_encoders', and 'vae' at the following links.  \n  \n**circlestone-labs/Anima at main**  \n**[https://huggingface.co/circlestone-labs/Anima/tree/main/split\\_files](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files)**  \n  \nOnce the file download is complete, open the Downloads folder and the ComfyUI data folder, and move 'anima-base-v1.0.\nsafetensors' into 'diffusion\\_models', 'qwen\\_3\\_06b\\_base.safetensors' into 'text\\_encoders', and 'qwen\\_image\\_vae.safetensors' into 'vae'. The ComfyUI data folder will vary depending on where you installed ComfyUI; if you installed it in 'C:\\\\comfyui\\\\', it will be in 'C:\\\\comfyui\\\\models'.  \n[](https://i.gzn.jp/img/2026/05/15/anima-image-generation-ai/01.png)\n\nNext, launch ComfyUI, open the Anima\n\n[sample image page](https://huggingface.co/circlestone-labs/Anima/blob/main/example.png) in your browser, and drag and drop the sample image into the ComfyUI window to load the workflow.  \n[](https://i.gzn.jp/img/2026/05/15/anima-image-generation-ai/02.png)\n\nThe workflow is for the initial preview version, so we will change the settings. First, in the model selection field within the 'Load Diffusion Model' node, select 'anima-base-v1.0.safetensors'.\n\n[](https://i.gzn.jp/img/2026/05/15/anima-image-generation-ai/03.png)\n\nNext, double-click on an empty area.  \n[](https://i.gzn."
      ]
    },
    {
      "url": "https://github.com/T8mars/comfyui-anima-t8",
      "title": "GitHub - T8mars/comfyui-anima-t8 · GitHub",
      "publish_date": null,
      "excerpts": [
        "> Anima 动漫提示词生成器 · ComfyUI 自定义节点\n> \n> 风格库三段式（52 条预设 / 15 分类）+ 1000+ 画师库 + Danbooru 四类（画师 / 作品 IP / 角色 IP / 风格·meta）+ Civitai 一键抓取 + 实时风格预览图\n> \n> \n\nversion [ComfyUI](https://github.com/comfyanonymous/ComfyUI) license\n\n* * *\n\n## ✨ 简介\n\n**comfyui-anima-t8** 是为 [Anima 动漫文生图模型](https://huggingface.co/circlestone-labs/Anima) 设计的 ComfyUI 提示词工作站，把\"风格库 + 画师库 + IP 库\"完整搬进 ComfyUI 主面板，让你无需离开画布就能完成提示词组装。\n\n**核心特色** ：\n\n* 📚 **风格库 52 条 Pony 兼容预设 × 15 分类** ：画质 / 媒介 / 镜头 / 构图 / 光影 / 服装 / 表情 / 季节 / 时代 / 场景 / 风格 / 情绪 / 角色 / NSFW / 测试，开箱即用\n* 🌐 **一键 Civitai 抓取** ：输入模型 ID 自动拉高赞图 prompt，按 14 组关键词自动归类（兼容 Civitai 直 prompt / 嵌套 / ComfyUI workflow 三种 meta 结构）\n* 🎨 **5 个素材库 Tab 一键切换** ：mooshieblob 画师 + Danbooru 画师 / 作品 IP / 角色 IP / 风格·meta\n* 🖼️ **实时风格预览图** ：选画师后运行节点即可在 PreviewImage 看到代表作首图\n* 🚀 **本地缓存 + 并发拉取** ：首次切 Tab 仅 2~5 秒可用，后台异步补全 30000+ 标签\n* 🧬 **增量种子机制** ：升级版本会按 name / title 增量补入新预设， **不覆盖** 用户已编辑的内容\n* 📌 **Pin 收藏 / 仅固定 / A-Z 字母筛选 / 关键字搜索** 一应俱全\n* 🛡️ **图片代理** ：所有 Danbooru CDN 图片走后端同源代理，绕开浏览器防盗链/CSP\n\n* * *\n\n## 📦 节点清单\n\n|节点 |说明 |主要输入 |输出 |\n| --- | --- | --- | --- |\n|**Anima Prompt T8** |三段式提示词组装（正向 / 负向 / 风格） |positive / negative / style |POSITIVE / NEGATIVE |\n|**Anima Artist Style T8** |画师风格输出 + 实时预览 |artist\\_tags（多选）/ default\\_weight / use\\_artist\\_prefix |STYLE\\_PROMPT / **PREVIEW\\_IMAGES** |\n|**Anima Prompt Combiner T8** |把多段提示词合并为单段 |text\\_a / text\\_b / separator |COMBINED |\n|**Anima Saved Prompt Loader T8** |从风格库一键加载已保存提示词 |preset\\_id |POSITIVE / NEGATIVE / STYLE |\n\n* * *\n\n## 🚀 安装\n\n### 方式 1：手动 clone\n\n```\ncd ComfyUI/custom_nodes\ngit clone https://github.com/T8mars/comfyui-anima-t8\n cd comfyui-anima-t8\npip install -r requirements.txt\n```\n\n### 方式 2：通过 ComfyUI Manager\n\n在 ComfyUI Manager 搜索 `comfyui-anima-t8` 后一键安装（待发布到 manager 列表）。\n\n### 依赖\n\n* Python ≥ 3.10\n* ComfyUI 主分支（任意近期版本）\n* `Pillow` / `numpy` / `torch` （ComfyUI 自带，无需额外）\n* `requests` （已写入 `requirements.txt` ）\n\n* * *\n\n## 🎯 使用流程\n\n### 1\\. 拖入节点\n\n在 ComfyUI 画布右键 → `Add Node` → `Anima/T8` → 选择需要的节点。\n\n### 2\\. 打开素材库\n\n每个 Anima 节点上方都有按钮：\n\n* **📚 风格库** ：浏览/搜索/收藏已保存的三段式提示词，支持 📥 导入 / 📤 导出 / 🌐 Civitai 抓取\n* **🎨 艺术家 / IP 库** ：5 个 Tab 切换\n  \n    + 👤 **Danbooru 画师** —— 30000+ Danbooru 真实标注画师标签\n    + 📚 **作品 IP** —— 5000+ 动画/游戏/漫画作品名\n    + 🧑‍🎤 **角色 IP** —— 30000+ 角色名\n    + 🎭 **风格·meta** —— Danbooru meta tag（画风 / 媒介 / 题材描述）\n    + 🎨 **画师库·mooshieblob** —— 1000+ 精选画师 + 高质量预览图\n\n### 3\\. 选择 → 添加\n\n* 在卡片上点击 → 进入选中状态\n* 点击右上角\"➜ 添加 N 个\" → 自动写入对应节点的 widget\n  \n    + 画师类 → 写入 `artist_tags` widget（Anima Artist Style T8）或 `style` widget（Anima Prompt T8）\n    + 作品 / 角色 IP → 写入 `positive` widget\n\n### 4\\. 运行查看预览\n\n把 Anima Artist Style T8 的 `PREVIEW_IMAGES` 端口连到 `PreviewImage` 节点，运行后会自动从 Danbooru 拉每个画师的代表作首图（≤16 张，6 路并发，约 2~5 秒）。"
      ]
    },
    {
      "url": "https://comfyanonymous.github.io/ComfyUI_examples/cosmos",
      "title": "Original Nvidia Cosmos Models | ComfyUI_examples - GitHub Pages",
      "publish_date": null,
      "excerpts": [
        "Original Nvidia Cosmos Models | ComfyUI\\_examples\n\n# [ComfyUI\\_examples](https://comfyanonymous.github.io/ComfyUI_examples/)\n\n# Original Nvidia Cosmos Models\n\nFor the newer Cosmos models see Cosmos Predict2\n\n[Nvidia Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) is a family of “World Models”. ComfyUI currently supports specifically the 7B and 14B text to video diffusion models and the 7B and 14B image to video diffusion models.\n\n## Files to Download\n\nYou will first need:\n\n#### Text encoder and VAE:\n\n[oldt5\\_xxl\\_fp8\\_e4m3fn\\_scaled.safetensors](https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/tree/main/text_encoders) goes in: ComfyUI/models/text\\_encoders/\n\n[cosmos\\_cv8x8x8\\_1.0.safetensors](https://huggingface.co/comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI/blob/main/vae/cosmos_cv8x8x8_1.0.safetensors) goes in: ComfyUI/models/vae/\n\nNote: oldt5\\_xxl is not the same as the t5xxl used in flux and other models. \noldt5\\_xxl is t5xxl 1.\n0 while the one used in flux and others is t5xxl 1.1\n\n#### Video Models\n\nThe video models can be found [in safetensors format here.](https://huggingface.co/mcmonkey/cosmos-1.0/tree/main)\n\nThe workflows on this page use [Cosmos-1\\_0-Diffusion-7B-Text2World.safetensors](https://huggingface.co/mcmonkey/cosmos-1.0/blob/main/Cosmos-1_0-Diffusion-7B-Text2World.safetensors) and [Cosmos-1\\_0-Diffusion-7B-Video2World.safetensors](https://huggingface.co/mcmonkey/cosmos-1.0/blob/main/Cosmos-1_0-Diffusion-7B-Video2World.safetensors)\n\nThese files go in: ComfyUI/models/diffusion\\_models\n\nNote: “Text to World” means Text to video and “Video to World” means image/video to video.\n\nIf you want the original diffusion models in .pt format instead of the repacked safetensors the official links are: [7B-Text2World](https://huggingface.co/nvidia/Cosmos-1.0-Diffusion-7B-Text2World) [7B-Video2World](https://huggingface.co/nvidia/Cosmos-1.0-Diffusion-7B-Video2World) [14B-Text2World](https://huggingface."
      ]
    },
    {
      "url": "https://github.com/kohya-ss/ComfyUI-Anima-LLLite/blob/main/nodes.py",
      "title": "ComfyUI-Anima-LLLite/nodes.py at main · kohya-ss/ComfyUI-Anima-LLLite · GitHub",
      "publish_date": null,
      "excerpts": [
        "43\n\n44\n\n45\n\n46\n\n47\n\n48\n\n49\n\n50\n\n51\n\n52\n\n53\n\n54\n\n55\n\n56\n\n57\n\n58\n\n59\n\n60\n\n61\n\n62\n\n63\n\n64\n\n65\n\n66\n\n67\n\n68\n\n69\n\n70\n\n71\n\n72\n\n73\n\n74\n\n75\n\n76\n\n77\n\n78\n\n79\n\n80\n\n81\n\n82\n\n83\n\n84\n\n85\n\n86\n\n87\n\n88\n\n89\n\n90\n\n91\n\n92\n\n93\n\n94\n\n95\n\n96\n\n97\n\n98\n\n99\n\n100\n\n101\n\n102\n\n103\n\n104\n\n105\n\n106\n\n107\n\n108\n\n109\n\n110\n\n111\n\n112\n\n113\n\n114\n\n115\n\n116\n\n117\n\n118\n\n119\n\n120\n\n121\n\n122\n\n123\n\n124\n\n125\n\n126\n\n127\n\n128\n\n129\n\n130\n\n131\n\n132\n\n133\n\n134\n\n135\n\n136\n\n137\n\n138\n\n139\n\n140\n\n141\n\n142\n\n143\n\n144\n\n145\n\n146\n\n147\n\n148\n\n149\n\n150\n\n151\n\n152\n\n153\n\n154\n\n155\n\n156\n\n157\n\n158\n\n159\n\n160\n\n161\n\n162\n\n163\n\n164\n\n165\n\n166\n\n167\n\n168\n\n169\n\n170\n\n171\n\n172\n\n173\n\n174\n\n175\n\n176\n\n177\n\n178\n\n179\n\n180\n\n181\n\n182\n\n183\n\n184\n\n185\n\n186\n\n187\n\n188\n\n189\n\n190\n\n191\n\n192\n\n193\n\n194\n\n195\n\n196\n\n197\n\n198\n\n199\n\n200\n\n201\n\n202\n\n203\n\n204\n\n205\n\n206\n\n207\n\n208\n\n209\n\n210\n\n211\n\n212\n\n213\n\n214\n\n215\n\n216\n\n217\n\n218\n\n219\n\n220\n\n221\n\n222\n\n223\n\n224\n\n225\n\n226\n\n227\n\n228\n\n229\n\n230\n\n231\n\n232\n\n233\n\n234\n\n235\n\n236\n\n237\n\n238\n\n239\n\n240\n\n241\n\n242\n\n243\n\n244\n\n245\n\n246\n\n247\n\n248\n\n249\n\n250\n\n251\n\n252\n\n253\n\n254\n255\n\n256\n\n257\n\n258\n\n259\n\n260\n\n261\n\n262\n\n263\n\n264\n\n265\n\n266\n\n\"\"\"ComfyUI node for Anima ControlNet-LLLite.\n\nSingle LoRA-style node: takes a MODEL, an LLLite weights file, a control IMAGE\n\nand a strength; returns the patched MODEL. Integration is done via\n\n``set\\_model\\_unet\\_function\\_wrapper`` so the LLLite contribution is fully scoped\n\nto this model clone — no global monkey-patching that could leak into other\n\nsamplers in the same workflow.\n\n\"\"\"\n\nfrom \\_\\_future\\_\\_ import annotations\n\nimport logging\n\nimport os\n\nfrom typing import Optional\n\nimport torch\n\nimport torch . nn . functional as F\n\nimport folder\\_paths\n\nfrom . control\\_net\\_lllite\\_anima import (\n\nASPP\\_DEFAULT\\_DILATIONS ,\n\nControlNetLLLiteDiT ,\n\nload\\_lllite\\_weights ,\n\nread\\_lllite\\_metadata ,\n\n)\n\nlogger = logging . getLogger ( \\_\\_name\\_\\_ )\n\ndef \\_get\\_inner\\_dit ( model ) \\-> torch . nn . Module :\n\n\"\"\"Reach the underlying Anima DiT (nn.Module) from a ComfyUI ModelPatcher.\"\"\"\n\ninner = getattr ( model , \"model\" , None )"
      ]
    },
    {
      "url": "https://github.com/CompVis/stable-diffusion",
      "title": "GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model · GitHub",
      "publish_date": null,
      "excerpts": [
        "ai/blog/laion-aesthetics/) (a subset of laion2B-en with estimated aesthetics score `> 5.0` , and additionally\n  filtered to images with an original size `>= 512x512` , and an estimated watermark probability `< 0.5` . The watermark estimate is from the [LAION-5B](https://laion.ai/blog/laion-5b/) metadata, the aesthetics score is estimated using the [LAION-Aesthetics Predictor V2](https://github.com/christophschuhmann/improved-aesthetic-predictor) ).\n* `sd-v1-3.ckpt` : Resumed from `sd-v1-2.ckpt` . 195k steps at resolution `512x512` on \"laion-aesthetics v2 5+\" and 10% dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598) .\n* `sd-v1-4.ckpt` : Resumed from `sd-v1-2.ckpt` . 225k steps at resolution `512x512` on \"laion-aesthetics v2 5+\" and 10% dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598) .\n\nEvaluations with different classifier-free guidance scales (1.5, 2.0, 3.\n0, 4.0,\n5\\.0, 6.0, 7.0, 8.0) and 50 PLMS sampling\nsteps show the relative improvements of the checkpoints: sd evaluation results\n\n### Text-to-Image with Stable Diffusion\n\ntxt2img-stable2 txt2img-stable2\n\nStable Diffusion is a latent diffusion model conditioned on the (non-pooled) text embeddings of a CLIP ViT-L/14 text encoder.\nWe provide a reference script for sampling , but\nthere also exists a diffusers integration , which we\nexpect to see more active community development.\n\n#### Reference Sampling Script\n\nWe provide a reference sampling script, which incorporates\n\n* a [Safety Checker Module](https://github.com/CompVis/stable-diffusion/pull/36) ,\n  to reduce the probability of explicit outputs,\n* an [invisible watermarking](https://github.com/ShieldMnt/invisible-watermark) of the outputs, to help viewers identify the images as machine-generated .\n\nAfter obtaining the `stable-diffusion-v1-*-original` weights , link them\n\n```\nmkdir -p models/ldm/stable-diffusion-v1/"
      ]
    },
    {
      "url": "https://github.com/kohya-ss/ComfyUI-Anima-LLLite",
      "title": "GitHub - kohya-ss/ComfyUI-Anima-LLLite · GitHub",
      "publish_date": null,
      "excerpts": [
        "- [Premium Support Enterprise-grade 24/7 support](https://github.com/premium-support)\n* [Pricing](https://github.com/pricing)\n\nSearch or jump to...\n\n# Search code, repositories, users, issues, pull requests...\n\nSearch\n\nClear\n\n[Search syntax tips](https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax)\n\nAppearance settings\n\nResetting focus\n\nYou signed in with another tab or window. Reload to refresh your session. You signed out in another tab or window. Reload to refresh your session. You switched accounts on another tab or window. Reload to refresh your session. Dismiss alert\n\nkohya-ss / **ComfyUI-Anima-LLLite** Public\n\n* Notifications You must be signed in to change notification settings\n* Fork 4\n* Star 53\n\n* Code\n* Issues 0\n* Pull requests 0\n* Actions\n* Projects\n* Security and quality 0\n* Insights\n\nAdditional navigation options * Code\n* Issues\n* Pull requests\n* Actions\n* Projects\n* Security and quality\n* Insights\n# kohya-ss/ComfyUI-Anima-LLLite\n\nmain\n\nBranches Tags\n\n \n\nGo to file\n\nCode\n\nOpen more actions menu\n\n## Folders and files\n\n|Name |Name |Last commit message |Last commit date |\n| --- | --- | --- | --- |\n|## Latest commit\n\n## History\n\n10 Commits\n\n 10 Commits |\n|.gitignore |.gitignore | | |\n|LICENSE |LICENSE | | |\n|README.md |README.md | | |\n|\\_\\_init\\_\\_.py |\\_\\_init\\_\\_.py | | |\n|control\\_net\\_lllite\\_anima.py |control\\_net\\_lllite\\_anima.py | | |\n|nodes.py |nodes.py | | |\n|View all files |\n\n## Repository files navigation\n\n* README\n* Apache-2.0 license\n\n# ComfyUI-Anima-LLLite\n\nComfyUI custom node for **ControlNet-LLLite for Anima** (DiT-based).\n\nLLLite is a lightweight ControlNet variant that injects a low-rank correction\ninto the attention (and optionally MLP) projections of the DiT. This node\nloads weights trained with [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) and applies them\nto the ComfyUI Anima model at inference time."
      ]
    }
  ],
  "warnings": null,
  "usage": [
    {
      "name": "sku_search",
      "count": 1
    }
  ],
  "session_id": "ses_17f67f5beffeKq0EJMHZjfupbo"
}
```

---

## Assistant (Build · MiMo V2.5 Free · 5.2s)

_Thinking:_

Now I have a very comprehensive understanding. Let me now look at how ComfyUI's Anima implementation handles reference latents and what patches are available. Let me check the ComfyUI source for how it handles Anima specifically.

Now I have enough information to create a comprehensive instruction document. Let me also quickly check how ComfyUI integrates Anima's attention patching.

**Tool: webfetch**

**Input:**
```json
{
  "format": "text",
  "url": "https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/diffusion_models/anima.py",
  "timeout": 15
}
```

**Error:**
```
StatusCode: non 2xx status code (404 GET https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/diffusion_models/anima.py)
```

---

## Assistant (Build · MiMo V2.5 Free · 9.0s)

_Thinking:_

Let me try to find the correct path for the ComfyUI Anima model implementation.

**Tool: websearch**

**Input:**
```json
{
  "query": "site:github.com/comfyanonymous/ComfyUI anima.py diffusion_models ref_latents",
  "numResults": 5
}
```

**Output:**
```
{
  "search_id": "search_b5e13a82e54d4792af1e7d84a51db68b",
  "results": [
    {
      "url": "https://docs.comfy.org/tutorials/image/anima/anima",
      "title": "Anima Base v1 ComfyUI workflow example",
      "publish_date": null,
      "excerpts": [
        "> ## Documentation Index\n> Fetch the complete documentation index at: https://docs.comfy.org/llms.txt\n> Use this file to discover all available pages before exploring further.\n\n# Anima Base v1 ComfyUI workflow example\n\n> Anima is a 2B-parameter text-to-image model by CircleStone Labs / Comfy Org, optimized for anime and non-photorealistic illustration generation.\n\n<Tip>\n  <Tabs>\n    <Tab title=\"Portable or self deployed users\">\n      Make sure your ComfyUI is updated.\n\n      * [Download ComfyUI](https://www.comfy.org/download)\n      * [Update Guide](/installation/update_comfyui)\n\n      Workflows in this guide can be found in the [Workflow Templates](/interface/features/template).\n      If you can't find them in the template, your ComfyUI may be outdated. (Desktop version's update will delay sometime)\n\n      If nodes are missing when loading a workflow, possible reasons:\n\n      1. You are not using the latest ComfyUI version (Nightly version)\n      2. Some nodes failed to import at startup\n    </Tab>\n<Tab title=\"Desktop or Cloud users\">\n      * The Desktop is base on ComfyUI stable release, it will auto-update when there is a new Desktop stable release available.\n      * [Cloud](https://cloud.comfy.org) will update after ComfyUI stable release.\n\n      So, if you find any core node missing in this document, it might be because the new core nodes have not yet been released in the latest stable version. Please wait for the next stable release.\n    </Tab>\n  </Tabs>\n</Tip>\n\n**Anima** is an open text-to-image model by [CircleStone Labs](https://huggingface.co/circlestone-labs/Anima) in partnership with Comfy Org. With **2 billion** parameters, it is designed to generate high-quality **anime and non-photorealistic** images — ideal for characters, scenes, and concept art.\n\n**Model highlights**:\n\n* **Anime and artistic illustration focus** — tuned for non-photorealistic art styles\n* **ComfyUI native support** — loaded with standard UNETLoader, CLIPLoader, and VAELoader nodes\n* **Qwen-3 0.\n6B text encoder** — provides strong language understanding and prompt adherence\n* **No input image needed** — pure text-to-image generation\n* **Compact footprint** — 2B diffusion model, runs comfortably on consumer GPUs\n\n**Related links**:\n\n* [Hugging Face (Anima)](https://huggingface.co/circlestone-labs/Anima)\n\n## Available workflows\n\nAnima provides two workflows — the base version for standard use and a preview version for early access.\n\nBoth workflows use a **Subgraph** node to manage the text-to-image generation pipeline. You can open the subgraph to inspect or customize the internal nodes.\n\n<Card title=\"Learn about Subgraph\" icon=\"book-open\" href=\"/interface/features/subgraph\">\n  This workflow uses Subgraph nodes for modular processing. Check out the Subgraph documentation to learn how to customize and extend the workflow.\n</Card>\n\n### Anima Base v1: Text to Image\n\n<CardGroup cols={2}>\n  <Card title=\"Download Workflow\" icon=\"download\" href=\"https://github."
      ]
    },
    {
      "url": "https://comfy.org/workflows/image_anima_preview-cc053bb55df6",
      "title": "Anima Anime Text-to-Image Generation - ComfyUI Workflow",
      "publish_date": "2025-05-07",
      "excerpts": [
        "Comfy Comfy\n\n`/`\n\n[Apply to Join](https://links.comfy.org/hubearlyaccess) [Try Comfy Cloud](https://cloud.comfy.org/?utm_source=workflow_hub&utm_medium=site_CTA&utm_campaign=hub_preview&utm_content=site_button)\n\nFilter\n\nTop Creators\n\nPOPULAR CATEGORIES\n\nPartner Nodes Video Image Text to Image Image Edit\n\nPOPULAR MODELS\n\nGrok Google Flux LTX-2.3 Z-Image-Turbo\n\nClear All Done\n\nBack\n\nAnima Anime Text-to-Image Generation\n\n* * *\n\nThe Anima Anime Text-to-Image Generation workflow in ComfyUI is designed to transform text prompts into captivating anime-style images. This workflow leverages the Anima model, a specialized text-to-image model trained on anime and non-photorealistic art styles, ensuring that the generated images maintain the distinct aesthetics of anime. The workflow involves several key nodes, including CLIPTextEncode for processing the text prompt, and VAEDecode for decoding the latent space into a visual output.\nBy adjusting parameters such as steps and CFG scale within the KSampler node, users can fine-tune the artistic style and detail of the generated image.\n\nTechnically, this workflow is structured into three main steps: loading the necessary models, setting the image size, and inputting the prompt. The VAELoader and UNETLoader nodes are crucial for preparing the model components, while the EmptyLatentImage node initializes the latent space for image generation. The SaveImage node ensures that the final output is stored for user access. This workflow is particularly useful for artists and content creators who wish to quickly generate anime-style visuals from textual descriptions, providing a creative tool for concept art, storyboarding, and digital illustration.\n\nWorkflow\n\n# Anima Anime Text-to-Image Generation\n\nInput a text prompt to generate an anime-style image using the Anima model. Configure settings like steps and CFG scale to control the output.\nThe workflow produces a non-photorealistic illustration based on your description.\n\nCREATED BY\n\nComfyUI\n\nComfyUI 1 year ago\n\n[Try on Comfy Cloud](https://cloud.comfy.org/?share=cc053bb55df6&utm_source=workflow_hub&utm_medium=site_CTA&utm_campaign=hub_preview&utm_content=cc053bb55df6&utm_term=hero) Download Workflow\n\nCATEGORIES\n\nImage Generation Anime Text to Image\n\nModels\n\nAnima"
      ]
    },
    {
      "url": "https://docs.comfy.org/tutorials/video/ltx/ltx-2-3",
      "title": "LTX-2.3 ComfyUI Workflow Examples - ComfyUI - docs.comfy.org",
      "publish_date": null,
      "excerpts": [
        "Learn how to use LTX-2.3 in ComfyUI for enhanced audio-video generation with improved details and portrait support"
      ]
    },
    {
      "url": "https://comfy.icu/node/ReferenceLatent",
      "title": "ReferenceLatent - ComfyUI Cloud",
      "publish_date": null,
      "excerpts": [
        "The most powerful and modular diffusion model GUI, api and backend with a graph/nodes interface. Authored by Comfy-Org. Run ComfyUI workflows in the Cloud! No downloads or installs are required. Pay only for active GPU usage, not idle time. No complex setups and dependency issues."
      ]
    },
    {
      "url": "https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/anima.py",
      "title": "ComfyUI/comfy/text_encoders/anima.py at master · Comfy-Org/ComfyUI · GitHub",
      "publish_date": null,
      "excerpts": [
        "## Navigation Menu\n\nToggle navigation\n\nAppearance settings\n\nSearch or jump to...\n\n# Search code, repositories, users, issues, pull requests...\n\nAppearance settings\n\nResetting focus\n\n## Files Expand file tree\n\nmaster\n\n## Breadcrumbs\n\n1. ComfyUI\n2. / comfy\n3. / text\\_encoders\n\n/\n\n# anima.py\n\nCopy path\n\nMore file actions\n\nMore file actions\n\n## Latest commit\n\n## History\n\nHistory\n\n History\n\n63 lines (50 loc) · 3.77 KB\n\nmaster\n\n## Breadcrumbs\n\n1. ComfyUI\n2. / comfy\n3. / text\\_encoders\n\n/\n\n# anima.py\n\nTop\n\n## File metadata and controls\n\n* Code\n* Blame\n\n63 lines (50 loc) · 3.77 KB\n\n[Raw](https://github.com/Comfy-Org/ComfyUI/raw/refs/heads/master/comfy/text_encoders/anima.py)\n\nCopy raw file\n\nDownload raw file\n\nOpen symbols panel\n\nEdit and raw actions\n\n1\n\n2\n\n3\n\n4\n\n5\n\n6\n\n7\n\n8\n\n9\n\n10\n\n11\n\n12\n\n13\n\n14\n\n15\n\n16\n\n17\n\n18\n\n19\n\n20\n\n21\n\n22\n\n23\n\n24\n\n25\n\n26\n\n27\n\n28\n\n29\n\n30\n\n31\n\n32\n\n33\n\n34\n\n35\n\n36\n\n37\n\n38\n\n39\n\n40\n\n41\n\n42\n\n43\n\n44\n\n45\n\n46\n\n47\n\n48\n\n49\n\n50\n\n51\n\n52\n\n53\n\n54\n\n55\n\n56\n\n57\n\n58\n\n59\n\n60\n\n61\n\n62\n\n63\nfrom transformers import Qwen2Tokenizer , T5TokenizerFast\n\nimport comfy . text\\_encoders . llama\n\nfrom comfy import sd1\\_clip\n\nimport os\n\nimport torch\n\nclass Qwen3Tokenizer ( sd1\\_clip . SDTokenizer ):\n\ndef \\_\\_init\\_\\_ ( self , embedding\\_directory = None , tokenizer\\_data = {}):\n\ntokenizer\\_path = os . path . join ( os . path . dirname ( os . path . realpath ( \\_\\_file\\_\\_ )), \"qwen25\\_tokenizer\" )\n\nsuper (). \\_\\_init\\_\\_ ( tokenizer\\_path , pad\\_with\\_end = False , embedding\\_directory = embedding\\_directory , embedding\\_size = 1024 , embedding\\_key = 'qwen3\\_06b' , tokenizer\\_class = Qwen2Tokenizer , has\\_start\\_token = False , has\\_end\\_token = False , pad\\_to\\_max\\_length = False , max\\_length = 99999999 , min\\_length = 1 , pad\\_token = 151643 , tokenizer\\_data = tokenizer\\_data )\n\nclass T5XXLTokenizer ( sd1\\_clip . SDTokenizer ):\n\ndef \\_\\_init\\_\\_ ( self , embedding\\_directory = None , tokenizer\\_data = {}):\n\ntokenizer\\_path = os . path . join ( os . path . dirname ( os . path .\nrealpath ( \\_\\_file\\_\\_ )), \"t5\\_tokenizer\" )\n\nsuper (). \\_\\_init\\_\\_ ( tokenizer\\_path , embedding\\_directory = embedding\\_directory , pad\\_with\\_end = False , embedding\\_size = 4096 , embedding\\_key = 't5xxl' , tokenizer\\_class = T5TokenizerFast , has\\_start\\_token = False , pad\\_to\\_max\\_length = False , max\\_length = 99999999 , min\\_length = 1 , tokenizer\\_data = tokenizer\\_data )\n\nclass AnimaTokenizer :\n\ndef \\_\\_init\\_\\_ ( self , embedding\\_directory = None , tokenizer\\_data = {}):\n\nself . qwen3\\_06b = Qwen3Tokenizer ( embedding\\_directory = embedding\\_directory , tokenizer\\_data = tokenizer\\_data )\n\nself . t5xxl = T5XXLTokenizer ( embedding\\_directory = embedding\\_directory , tokenizer\\_data = tokenizer\\_data )\n\ndef tokenize\\_with\\_weights ( self , text : str , return\\_word\\_ids = False , \\*\\* kwargs ):\n\nout = {}\n\nqwen\\_ids = self . qwen3\\_06b . tokenize\\_with\\_weights ( text , return\\_word\\_ids , \\*\\* kwargs )\n\nout [ \"qwen3\\_06b\" ] = [[( k [ 0 ], 1\\."
      ]
    },
    {
      "url": "https://www.runcomfy.com/comfyui-nodes/style_aligned_comfy/StyleAlignedSampleReferenceLatents",
      "title": "StyleAligned Sample Reference Latents - RunComfy",
      "publish_date": "2025-03-24",
      "excerpts": [
        "The ref_latents output parameter provides the reference latents that have been sampled and aligned with the specified style. These latents can be used in subsequent stages of the generation process to ensure consistency and coherence in the final output."
      ]
    },
    {
      "url": "https://github.com/shootthesound/comfyui-ReferenceLatentPlus",
      "title": "GitHub - shootthesound/comfyui-ReferenceLatentPlus: Improved ...",
      "publish_date": "2026-05-07",
      "excerpts": [
        "I tested all of this against Flux2/Klein 9B (the model I'm building Fizgig for), but the underlying mechanism is generic — it works on every model in ComfyUI that consumes reference_latents."
      ]
    },
    {
      "url": "https://huggingface.co/circlestone-labs/Anima/discussions",
      "title": "circlestone-labs/Anima · Discussions - Hugging Face",
      "publish_date": null,
      "excerpts": [
        "# \n\ncirclestone-labs\n\n/\n\nAnima\n\nlike 1\\.59k\n\nFollow\n\nCircleStone Labs 750\n\nDiffusion Single File comfyui\n\nLicense: circlestone-labs-non-commercial-license\n\nModel card Files Files and versions xet Community 180\n\nCopy to bucket new\n\nUse this model\n\n### Instructions to use circlestone-labs/Anima with libraries, inference providers, notebooks, and local apps. Follow these links to get started.\n\n* Libraries\n* Diffusion Single File\n  \n  How to use circlestone-labs/Anima with Diffusion Single File:\n  \n  ```\n  # No code snippets available yet for this library.\n  \n  # To use this model, check the repository files and the library's documentation.\n  \n  # Want to help? PRs adding snippets are welcome at:\n  # https://github.com/huggingface/huggingface.js\n  ```\n* Notebooks\n* Google Colab\n* Kaggle\n\nNew discussion\n\nNew pull request\n\nResources\n\n* PR & discussions documentation\n* Code of Conduct\n* Hub documentation\n\n* All\n* Discussions\n* Pull requests\n\nView closed (14)\n\nSort: \nRecently created\n#### Explanation of the license pinned 👍 ❤️ 56 39\n\n\\ opened 4 months ago\nby tdrussell\n\n#### Added a better ComfyUI sampler based on the Cosmos-style RF sampling 👍 1\n\n\\ opened about 10 hours ago\nby Aizenzz\n\n#### Anima Support in KNX Trainer! (and thank you!) 🔥 1\n\n\\ opened about 12 hours ago\nby Duskfallcrew\n\n#### L2P for Anima?\n\n\\ opened about 21 hours ago\nby PokeGeek\n\n#### Add base\\_model metadata (nvidia/Cosmos-Predict2-2B-Text2Image) 👍 1\n\n\\ opened 1 day ago\nby victor\n\n#### Question for Everyone. 😔 2\n\n\\ opened 1 day ago\nby Darkknight535\n\n#### Anima in tensor art 5\n\n\\ opened 2 days ago\nby zak01010101\n\n#### NVFP4 / MXFP4 / FP8 quantizations for faster inference 👍 🔥 2 1\n\n\\ opened 2 days ago\nby Iwaku-Real\n\n#### Anima can edit images without inpainting. (In case anyone missed it) 👍 6 6\n\n\\ opened 3 days ago\nby Nuke1229\n\n#### Should the backslash (`\\\\`) be removed from an artist name that contains parentheses `()`? 1\n\n\\ opened 3 days ago\nby daijava\n#### Anima LoRA Stage Scheduler and Layer Control Plugin 👍 4\n\n\\ opened 5 days ago\nby kongbai-84\n\n#### This is just a thank you! I have no complaints. 🤗 1\n\n\\ opened 5 days ago\nby PizmoSF\n\n#### Anima-Artist-Mixer has now been uploaded to GitHub! 👍 8 3\n\n\\ opened 7 days ago\nby R3DeK\n\n#### Regarding artist mixing through the cross-attention layer of hook Anima ❤️ 👍 4 5\n\n\\ opened 8 days ago\nby R3DeK\n\n#### Upload LICENSE\\_CIRCLESTONE\\_NON\\_COMMERCIAL.md\n\n\\ opened 8 days ago\nby oron1208\n\n#### Optimal sampler / scheduler settings for Anima (+ comparison site) 👍 10 3\n\n\\ opened 10 days ago\nby darask0\n\n#### After adding weights, how should the @ prefix be added before artist tag? 1\n\n\\ opened 10 days ago\nby WallBreakerNO4\n\n#### Had a question about Hires.fix 1\n\n\\ opened 10 days ago\nby earthboundAI\n\n#### Lora character training 83\n\n\\ opened 10 days ago\nby Shadyman961\n\n#### Please tell me if there is a complete list of characters supported by this model? 3\n\n\\ opened 11 days ago\nby JarvisS666\n\n...\n\ncom/huggingface/) \n\nWebsite\n\nModels Datasets Spaces Pri"
      ]
    },
    {
      "url": "https://comfyai.run/documentation/StyleAlignedSampleReferenceLatents",
      "title": "StyleAligned Sample Reference Latents",
      "publish_date": null,
      "excerpts": [
        "It enables users to generate new images by employing a reference latent image as the basis for style alignment. This node comes in handy when you want to infuse a specific stylistic element into newly generated images, balancing between reference style and image content."
      ]
    },
    {
      "url": "https://www.youtube.com/watch?v=0trUwzli5G0",
      "title": "Animate ANY Reference Image with a Video | WAN 2.2 ... - YouTube",
      "publish_date": "2025-10-03",
      "excerpts": [
        "Today I'll be sharing a workflow I made for ComfyUI that allows you to animate any reference image using a video as a driver for the movment with WAN 2.2 ANIMATE."
      ]
    }
  ],
  "warnings": null,
  "usage": [
    {
      "name": "sku_search",
      "count": 1
    }
  ],
  "session_id": "ses_17f67f5beffeKq0EJMHZjfupbo"
}
```

---

## Assistant (Build · MiMo V2.5 Free · 58.9s)

_Thinking:_

Now I have enough information to create a comprehensive instruction document for adapting the Untwisting RoPE implementation for the Anima model. Let me now write this instruction document.

Key differences between Flux.2 and Anima that need to be addressed:

1. **Architecture**: Anima is a Cosmos-Predict2-based DiT, not a Flux/Flux.2 MMDiT. It has:
   - 28 transformer blocks (AnimaBlock) instead of Flux's single/double blocks
   - 3D RoPE (Temporal, Height, Width) instead of Flux's 2D RoPE
   - Separate self-attention and cross-attention within each block
   - QK-norm (RMSNorm on Q/K before RoPE)
   - LLMAdapter bridge for text conditioning
   - Uses `bshd` tensor format internally, while Flux uses a different layout

2. **ComfyUI integration**: 
   - Anima uses UNETLoader instead of the separate double/single block structure
   - Reference latents may or may not have native support via `ref_latents`
   - The `attn1_patch` hook may or may not work the same way for Anima
   - Anima's self-attention applies RoPE directly within the Attention module, so I need to understand how the patching mechanism interacts with this

3. **RoPE structure**: 
   - Flux uses 2D RoPE decomposed as [image_ref_index, y, x] across axes_dim
   - Anima uses 3D RoPE decomposed as [temporal, height, width] with specific dimension allocation: dim_h = dim // 6 * 2, dim_w = dim_h, dim_t = dim - 2 * dim_h

4. **Tensor layout**: 
   - Flux uses attention in single-stream blocks with a specific layout
   - Anima uses `bshd` format (batch, sequence, heads, dim) with separate self-attention and cross-attention

5. **Reference latent mechanism**: Flux has native support for reference latents via `ref_latents` in conditioning, but Anima doesn't have this built-in—I'd need to implement it myself by appending reference image tokens to the sequence, scaling reference keys using the frequency scale vector in self-attention, and handling the denoising schedule.

The main implementation challenge is figuring out how to intercept Anima's self-attention to apply the RoPE scaling to reference tokens, since it doesn't have the same hook system that Flux provides. I could use a model wrapper to inject the behavior, or I might need to monkey-patch the Attention module directly.

Now I have all the information needed. Here's the comprehensive instruction document for an AI agent:

---

# Instruction: Implement Untwisting RoPE for the Anima Model

## 1. Goal

Adapt the existing ComfyUI-Flux.2-Untwisting-RoPE custom node (frequency-aware RoPE reference-attention modulation) to work with the **Anima** model (2B-parameter Cosmos-Predict2-based DiT by CircleStone Labs / Comfy Org).

The core paper mechanism to preserve:

- RoPE decomposes into frequency components with different positional sensitivity.
- High-frequency RoPE components cause reference spatial copying; low-frequency components preserve broader style association.
- The intervention modulates **reference image keys only** in self-attention, not text keys, not values.
- The scaling is smooth across the RoPE spectrum and schedulable over denoising.

## 2. Critical Architecture Differences (Flux.2 vs. Anima)

### 2.1 Model Architecture

| Aspect | Flux / Flux.2 | Anima |
|---|---|---|
| Architecture | MMDiT (double + single blocks) | Cosmos-Predict2 DiT (AnimaBlock) |
| Block count | 19 double + 38 single (Flux.1) | 28 identical blocks |
| Block types | `double_blocks` (joint attn) + `single_blocks` (self-attn only) | Each block has: self-attention + cross-attention + MLP, all modulated by AdaLN |
| RoPE type | 2D (applied in ComfyUI's attention kernel) | **3D RoPE** (temporal T, height H, width W), applied inside `Attention.compute_qkv()` |
| QK norm | None (Flux.2) | **RMSNorm** on Q and K *before* RoPE application |
| Tensor format | Custom Flux layout | `bshd` (batch, sequence, heads, dim) |
| Text encoder | CLIP-L + T5xxl | **Qwen3-0.6B** (+ optional T5xxl via LLMAdapter) |
| Reference latents | Native `ref_latents` path in ComfyUI conditioning | **No native `ref_latents` path** — must be implemented manually |
| Model loader | `UNETLoader` or equivalent | `UNETLoader` (ComfyUI loads as diffusion model) |

### 2.2 Anima Block Forward Pass (simplified)

From `library/anima_models.py`, each `Block._forward()`:

```
x = x + extra_per_block_pos_emb  (optional absolute pos emb)

# 1. Self-attention with AdaLN + RoPE
normed_x = adaln_fn(x, layer_norm_self_attn, scale, shift)
result = self_attn(rearrange(normed_x, "b t h w d -> b (t h w) d"), attn_params, None, rope_emb=rope_emb)
x = x + gate_self_attn * result

# 2. Cross-attention with AdaLN (NO RoPE on cross-attn keys)
normed_x = adaln_fn(x, layer_norm_cross_attn, scale, shift)
result = cross_attn(rearrange(normed_x, "b t h w d -> b (t h w) d"), attn_params, crossattn_emb, rope_emb=rope_emb)
x = result * gate_cross_attn + x

# 3. MLP with AdaLN
normed_x = adaln_fn(x, layer_norm_mlp, scale, shift)
result = mlp(normed_x)
x = x + gate_mlp * result
```

Key: RoPE is applied to Q and K **inside** `Attention.compute_qkv()` for self-attention only (when `context is None`). The `rope_emb` is a `(L, 1, 1, D)` tensor of 3D rotary frequencies.

### 2.3 Anima 3D RoPE Structure

From `VideoRopePosition3DEmb.generate_embeddings()`:

```python
dim_h = head_dim // 6 * 2   # e.g., for head_dim=128: dim_h = 42
dim_w = dim_h                 # dim_w = 42
dim_t = head_dim - 2*dim_h   # dim_t = 44
# Total: 42 + 42 + 44 = 128 = head_dim

# Frequencies:
h_spatial_freqs = 1.0 / (10000.0 * h_ntk_factor) ** dim_spatial_range
w_spatial_freqs = 1.0 / (10000.0 * w_ntk_factor) ** dim_spatial_range
temporal_freqs = 1.0 / (10000.0 * t_ntk_factor) ** dim_temporal_range

# The embedding is constructed as:
# [temporal_freqs | h_spatial_freqs | w_spatial_freqs] repeated twice
# then reshaped to (L, 1, 1, head_dim)
```

The RoPE embedding is structured as 3 axis chunks, each containing pairs of frequencies (repeated twice for cos/sin), concatenated along the last dimension. The order is: `[temporal, height, width]`, each occupying `dim_t`, `dim_h`, `dim_w` dimensions respectively, with each axis's pair scales repeated interleaved (x1, x2, x1, x2, ...).

## 3. Integration Strategy

### 3.1 Approach: `model_function_wrapper` + Self-Attention Monkey-Patch

Since Anima does NOT have ComfyUI's `attn1_patch` hook (which is Flux-specific), use the **`model_function_wrapper`** mechanism to:

1. Inject configuration into `transformer_options` per call.
2. Monkey-patch the `Attention.compute_qkv` method on each self-attention module to intercept Q/K before the attention kernel, apply reference-key frequency scaling, then delegate to the original.

### 3.2 What to Intercept

**Target: Self-attention Q/K after RoPE is applied.**

The Flux implementation scales reference keys *before* RoPE (pre-RoPE scaling), which is equivalent because scalar multiplication commutes with 2D rotation. For Anima's 3D RoPE, the same commutativity holds — scaling a key channel by a scalar is equivalent before or after rotation.

However, since Anima applies RoPE inside `Attention.compute_qkv()`, the simplest approach is:

**Option A (Recommended): Patch `Attention.compute_qkv` to:**
1. Call original `compute_qkv` to get Q, K, V (with RoPE already applied).
2. Identify reference token ranges from `extra_options` (via `transformer_options`).
3. Scale reference K channels using the frequency scale vector.
4. Return modified Q, K, V.

**Option B: Patch `Attention.forward` to intercept after `compute_qkv`.**

Option A is cleaner because it keeps the patching at the lowest level.

### 3.3 Reference Token Injection

Anima does NOT have a native `ref_latents` path. You must implement reference token injection manually:

1. Encode reference image with the Qwen-Image VAE → get latent tensor.
2. Patchify the reference latent into tokens using the same `PatchEmbed` as the target.
3. Concatenate reference tokens to the target token sequence at the appropriate point (before or after the target image tokens, matching the spatial grid layout).
4. Track which token positions are reference tokens using `img_slice`-style metadata in `transformer_options`.

The concatenation should happen before the transformer blocks process the sequence. Use the `model_function_wrapper` to prepend/append reference tokens to the latent, and pass token range metadata through `transformer_options`.

### 3.4 Model Detection

Replace the Flux-specific `_looks_like_flux()` check with an Anima detection function:

```python
def _looks_like_anima(obj: Any) -> bool:
    return (
        hasattr(obj, "blocks")  # nn.ModuleList of AnimaBlock
        and hasattr(obj, "x_embedder")  # PatchEmbed
        and hasattr(obj, "pos_embedder")  # VideoRopePosition3DEmb
        and hasattr(obj, "t_embedder")  # TimestepEmbedding
        and hasattr(obj, "params")  # config params
    )
```

Or detect via the class name: `type(obj).__name__ == "Anima"`.

### 3.5 Axes Dimension Mapping

Flux uses `axes_dim` to describe its 2D RoPE structure: `[image_ref_index, y, x]` (3 axes). Anima's 3D RoPE is always `[temporal, height, width]` with fixed dimension allocation based on `head_dim`:

```python
dim_h = head_dim // 6 * 2
dim_w = dim_h
dim_t = head_dim - 2 * dim_h
axes_dim = (dim_t, dim_h, dim_w)
```

For Anima (head_dim=128): `axes_dim = (44, 42, 42)`.

## 4. Implementation Steps

### 4.1 New File Structure

```
ComfyUI-Flux.2-Untwisting-RoPE/  (or rename to ComfyUI-Untwisting-RoPE)
  __init__.py
  nodes.py                     # Add AnimaUntwistRoPE node class
  flux_untwist/
    config.py                  # Reuse; add Anima-specific axes_dim detection
    patches.py                 # Add anima_untwist_self_attn_patch
    utils.py                   # Add Anima model detection, latent processing
  tests/
    test_frequency_scale.py    # Existing
    test_anima_*.py            # New Anima-specific tests
```

### 4.2 `config.py` Changes

- Add `anima_axes_dim_from_head_dim(head_dim: int) -> Tuple[int, int, int]` helper.
- The `UntwistConfig` dataclass is reusable as-is.

### 4.3 `patches.py` — New `anima_untwist_self_attn_patch`

This function will be installed as a monkey-patch on `Attention.compute_qkv`. It must:

```python
def anima_untwist_self_attn_patch(original_compute_qkv, self, x, context=None, rope_emb=None):
    """Monkey-patch for Anima Attention.compute_qkv (self-attention only)."""
    # Only intercept self-attention (context is None)
    if context is not None:
        return original_compute_qkv(self, x, context, rope_emb)

    # Get config from transformer_options (stored by model_function_wrapper)
    # ... read anima_untwist_rope from thread-local or self._anima_untwist_cfg ...

    q, k, v = original_compute_qkv(self, x, context, rope_emb)

    # Identify reference token ranges
    # ...

    # Build frequency scale vector
    scale_vec = build_frequency_scale_vector(
        head_dim=self.head_dim,
        axes_dim=anima_axes_dim_from_head_dim(self.head_dim),
        high_scale=..., low_scale=..., beta=...,
        device=k.device, dtype=k.dtype,
    )

    # Scale reference keys
    for start, end in ref_ranges:
        k[:, :, start:end, :] = k[:, :, start:end, :] * scale_vec.view(1, 1, 1, -1)

    return q, k, v
```

**Challenge: Passing configuration to the monkey-patch.** Options:
- Store config on the `Attention` module instance (e.g., `self._anima_untwist_cfg = cfg`).
- Use a thread-local or global dict keyed by module id.
- Use `functools.partial` to bind config when creating the patch.

Recommended: use `functools.partial`:

```python
from functools import partial

patch_fn = partial(
    anima_untwist_self_attn_patch,
    ref_ranges=ref_ranges,
    scale_vec=scale_vec,
    adain_strength=adain_strength,
    target_range=target_range,
)
# Then monkey-patch each self_attn module's compute_qkv
```

### 4.4 `nodes.py` — New `AnimaUntwistRoPE` Node

Inputs are nearly identical to `Flux2UntwistRoPE` but with Anima-specific adjustments:

- Remove `reference_latents_method` (Anima doesn't have Flux's native ref_latents placement methods).
- Remove `start_single_block` / `end_single_block` (Anima has uniform blocks, not double/single).
- Add `start_block` / `end_block` (0–27 for 28-block Anima).
- Keep all frequency scaling parameters unchanged.

The `patch()` method:

1. Clone the model.
2. Detect Anima model via `_looks_like_anima()`.
3. Extract `head_dim` and compute `axes_dim`.
4. Register reference latent tokens in `model_function_wrapper`.
5. Monkey-patch all `self_attn.compute_qkv` methods in the DiT blocks.

### 4.5 Reference Token Injection (model_function_wrapper)

```python
def model_function_wrapper(apply_model, args):
    input_x = args["input"]
    timestep = args["timestep"]
    c = dict(args["c"])

    # 1. Process reference latent
    ref_tokens = process_reference_for_anima(ref_samples_cpu, anima_model, input_x)

    # 2. Concatenate to input latent
    if ref_tokens is not None and active:
        # Append reference tokens to the sequence dimension
        # After patchification, tokens are (B, T*H*W, D)
        input_x_with_ref = torch.cat([input_x_flat, ref_tokens], dim=1)
        # Record ref range in transformer_options
        c["transformer_options"]["anima_untwist_rope"] = {
            "ref_range": (target_len, target_len + ref_len),
            ...
        }

    # 3. Apply model
    return apply_model(input_x_with_ref, timestep, **c)
```

**Note:** This is the most complex part. The reference tokens must be:
- Patchified with the same `PatchEmbed` as the target.
- Placed in the sequence such that their spatial positions make sense relative to the target.
- Un-patchified after the transformer blocks (or the model must handle variable-length sequences).

Consider using a simpler approach: **append reference tokens at the end of the sequence** and let the self-attention attend to them. Track the boundary with `ref_range` metadata.

### 4.6 Token Range Detection in the Patch

Unlike Flux, Anima does not provide `img_slice` or `reference_image_num_tokens` in `extra_options`. You must pass this information yourself:

- In `model_function_wrapper`, record the reference token range as metadata in `transformer_options["anima_untwist_rope"]["ref_ranges"]`.
- In the attention patch, read `ref_ranges` from `extra_options.get("anima_untwist_rope", {})`.

### 4.7 `utils.py` Changes

Add:
- `safe_get_anima_model(model_patcher)` — find the Anima DiT.
- `process_reference_for_anima(ref_latent, anima_model, target_latent)` — encode reference to tokens.
- `anima_axes_dim_from_head_dim(head_dim)` — compute the 3-axis dimension tuple.

## 5. `build_frequency_scale_vector` Adaptation

The existing function works with `axes_dim` as a sequence of axis dimensions. For Anima:

```python
# Anima axes are [temporal, height, width]
# The paper's insight: spatial axes (H, W) are responsible for spatial copying.
# The temporal axis preserves broader/global association.

axes = [dim_t, dim_h, dim_w]  # e.g., [44, 42, 42]

for axis_index, axis_dim in enumerate(axes):
    if axis_index == 0:  # Temporal axis → keep low_scale (reference-preserving)
        pair_scales = torch.full((n_pairs,), low_scale, ...)
    else:  # Spatial axes (H, W) → interpolate high_scale to low_scale
        d = torch.linspace(0, 1, n_pairs)
        pair_scales = high_scale + (low_scale - high_scale) * d.pow(beta)
```

This is the same logic as the Flux version — the first axis (non-spatial) gets `low_scale`, and spatial axes get the polynomial interpolation. The semantic mapping is:

| Flux axes | Anima axes | Role |
|---|---|---|
| `axes[0]` = image/ref index | `axes[0]` = temporal | Non-spatial → `low_scale` |
| `axes[1]` = height | `axes[1]` = height | Spatial → polynomial |
| `axes[2]` = width | `axes[2]` = width | Spatial → polynomial |

## 6. Defaults

```python
high_scale_start = 0.25   # Reduce high-freq spatial copying
high_scale_end   = 0.75   # Gradually restore
low_scale_start  = 1.00   # No change to low-freq initially
low_scale_end    = 1.40   # Boost low-freq for style pull
beta             = 2.0    # Paper default
start_percent    = 0.0
end_percent      = 1.0
start_block      = 0
end_block        = 27     # All 28 blocks (0-indexed)
```

Anima-specific tuning notes:
- Anima is a smaller (2B) model trained on anime. Style transfer effects may be more pronounced.
- The LLMAdapter processes text before cross-attention. This means cross-attention is text-conditioned and should NOT be modulated — only self-attention keys.
- Anima's QK-norm (RMSNorm before RoPE) means the frequency scale vector interacts with normalized Q/K. This should be fine since scaling commutes with normalization of the *reference* keys only.

## 7. Validation Plan

1. **Import test**: ComfyUI loads `Anima Untwist RoPE` without errors when Anima model is loaded.
2. **Baseline parity**: No reference connected → output matches unpatched model.
3. **No-op patch**: Reference connected but `start_percent/end_percent` excluding all steps → no effect.
4. **Reference injection**: With `verbose=True`, confirm reference token counts are detected during self-attention blocks.
5. **A/B visual tests** at fixed seed:
   - No reference
   - Reference connected (native if available, or node-injected)
   - With untwisting at defaults
   - Lower `high_scale_end` for copying-heavy references
   - Higher `low_scale_end` for weak style transfer
6. **Resolution sweep**: Test at 512², 768², 1024² — reference token count scales with resolution.

## 8. Known Risks

- Anima has **no native `ref_latents` path** in ComfyUI (as of v1.0). Reference token injection is entirely custom and must be implemented correctly.
- The monkey-patching of `Attention.compute_qkv` is more invasive than Flux's `attn1_patch` hook. It may break if ComfyUI changes Anima's module structure.
- Anima's 3D RoPE means reference tokens receive temporal + spatial frequencies. Scaling reference keys affects all 3 axes — the `axes_dim` mapping must be correct.
- Anima's QK-norm means the frequency scale vector is applied to *normalized* Q/K. This is mathematically fine but may interact differently with the model's trained behavior.
- Anima uses `bshd` format (batch, sequence, heads, dim) while Flux uses a different layout. All tensor indexing must be adjusted accordingly.

## 9. Code References

| What | Where |
|---|---|
| Flux reference implementation | `flux_untwist/patches.py` — `flux_untwist_attn1_patch`, `build_frequency_scale_vector` |
| Anima model source | `kohya-ss/sd-scripts/library/anima_models.py` — `Anima`, `Block`, `Attention`, `VideoRopePosition3DEmb` |
| Anima Attention.forward | `anima_models.py` lines ~280-310 (compute_qkv + attention call) |
| Anima Block._forward | `anima_models.py` lines ~978-1036 |
| Anima 3D RoPE | `anima_models.py` `VideoRopePosition3DEmb.generate_embeddings()` |
| ComfyUI ModelPatcher hooks | `comfy/model_patcher.py` — `set_model_unet_function_wrapper`, `set_model_patch` |
| ComfyUI Anima text encoder | `comfy/text_encoders/anima.py` — `AnimaTokenizer` |

---

This instruction provides everything needed for an AI agent to implement the adaptation. The agent should start by reading the full Flux reference implementation (`nodes.py`, `flux_untwist/patches.py`, `flux_untwist/config.py`, `flux_untwist/utils.py`) and the Anima source (`anima_models.py`) before writing code.

---

