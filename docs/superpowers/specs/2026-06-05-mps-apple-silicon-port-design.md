# PiD on MPS / Apple Silicon — Port Design

**Date:** 2026-06-05
**Status:** Approved (pending spec review)
**Scope:** Make PiD inference (`from_ldm` and `from_clean`) run on Apple Silicon via the
PyTorch MPS backend, with CPU as a universal fallback, without regressing the existing
CUDA path.

## Goal

PiD is NVIDIA's Pixel Diffusion Decoder — a drop-in replacement for VAE/RAE decoders that
denoises directly in high-resolution pixel space. The codebase is CUDA-only by hardcoded
convention, not by hard dependency: it is stock PyTorch + diffusers with **no custom CUDA
kernels, no flash-attn / triton / xformers / apex**, and attention is already
`scaled_dot_product_attention` (MPS-supported). This makes the port a matter of device,
dtype, RNG, and distributed plumbing rather than kernel work.

**Primary target:** M5 Max / 128 GB unified memory (dev box). Memory is not the binding
constraint there.
**Secondary (nice-to-have):** lower-spec Apple Silicon — addressed via a documented
`--dtype bf16` opt-in and a future tiling note, *not* code in this pass.

**First green path:** `from_clean` + `flux` (lightest real path), then `from_ldm` + `flux`.

## Non-Goals (YAGNI)

- No tiling / low-memory streaming / sequential offload in this pass.
- No torch.compile on MPS (Inductor MPS backend is immature).
- No multi-GPU / context-parallel support on Mac (single-process only).
- No checkpoint-download automation — weights are fetched separately via
  `hf download nvidia/PiD`. The port is code-only.

## Key Decision: fp32 default on MPS

MPS bf16 has historically diverged from CUDA bf16 in two ways that matter for a
**pixel-space** decoder: (1) patchy op coverage causing silent CPU fallbacks mid-graph,
and (2) different rounding/accumulation than CUDA, which compounds across multi-step
diffusion and lands directly in output pixels (no re-encode to wash it out). fp16 is worse
— its narrow dynamic range overflows in diffusion.

Decision:
- **Default dtype on MPS = fp32** (maximum CUDA parity, no silent-fallback surprises).
- Expose `--dtype {fp32,bf16,fp16}` to A/B bf16 for speed *after* fp32 is proven green.
- **CUDA keeps its bf16 default** (upstream parity).
- This is a runtime flag, not a code fork: `resolve_dtype()` picks fp32 when device is mps
  and the user did not override.

## Architecture

### New module: `pid/_src/utils/device_utils.py`

Single source of truth for device / dtype / RNG / backend setup.

| Function | Behavior |
|---|---|
| `get_device(prefer=None)` | Resolve order `mps → cuda → cpu`; honor explicit `prefer` (from `--device`). Result cached. |
| `resolve_dtype(device, requested=None)` | `requested` wins; else fp32 on mps, bf16 on cuda, fp32 on cpu. Maps string names (`fp32`/`bf16`/`fp16`) to torch dtypes. |
| `make_generator(device, seed)` | MPS generators are unreliable → build a **CPU** `torch.Generator` for mps (consumers move sampled noise to device); native generator on cuda/cpu. |
| `empty_cache(device)` | `torch.mps.empty_cache()` / `torch.cuda.empty_cache()` / no-op on cpu. |
| `setup_backends(device)` | Guard `cudnn.*` + `matmul.allow_tf32` behind cuda; set `PYTORCH_ENABLE_MPS_FALLBACK=1` early on mps. |
| `synchronize(device)` | Dispatch to the correct device sync (used for timing). |

The module is the only place that branches on backend identity. Every consumer asks it for
a device/dtype/generator and stays backend-agnostic.

### Edit sites (seams)

- `pid/_src/utils/model_loader.py`
  - backend flags → `setup_backends(device)`
  - `instantiate(config.model).cuda()` → `.to(device)`
  - `torch.cuda.empty_cache()` → `empty_cache(device)`
- `pid/_src/inference/decoder.py`
  - `device="cuda"` (latent + sigma tensors) → resolved `device` / `clean_latent.device`
- `pid/_src/inference/from_clean.py`
  - `torch.cuda.set_device(rank)` → guard to cuda only
  - `device="cuda"` literals (input tensor, sigma) → resolved device
  - `torch.Generator(device="cuda")` → `make_generator(device, seed)`
- `pid/_src/inference/from_ldm.py`
  - `torch.Generator(device="cuda")` → `make_generator(device, seed)`
- `pid/_src/inference/pipeline_registry.py` (`load_pipeline`)
  - `device="cuda"` default → resolved device
  - `enable_model_cpu_offload(gpu_id=...)` only on cuda; else `pipeline.to(device)`
- `pid/_src/models/pixeldit_model.py`
  - `device="cuda"` method defaults + text-encoder `torch_dtype` → resolved device/dtype
- `pid/_src/inference/cli_utils.py`
  - add `--device {auto,mps,cuda,cpu}` (default `auto`) and `--dtype {auto,fp32,bf16,fp16}`
    (default `auto`) to both entrypoints' parsers

### Distributed / pynvml neutralization

- `cli_utils.maybe_init_distributed()` — when `world_size == 1` (no `torchrun`), **skip
  entirely**: no NCCL, no pynvml, no `libcudart.so`. The single-process Mac path never
  touches CUDA-only libraries.
- `inference_utils.get_rank_and_world_size()` — already env-based; returns `(0, 1)` on Mac.
  Confirm only; no change expected.
- `pid/_ext/imaginaire/utils/distributed.py` `init()` and `device.py` pynvml imports are
  already lazy (function-local) and are simply never called on the single-process path.
- `pid/_src/utils/context_parallel.py` `.cuda()` sites live behind a CP process group that
  only exists under multi-GPU; unreachable on Mac, left as-is.

### torch.compile gate

`pid/_src/models/pixeldit_model.py::_maybe_compile_net` — skip compile when device is not
cuda; `--compile` becomes a no-op with a single warning line on mps/cpu. CUDA behavior
unchanged.

## Verification Plan

1. **Import / device smoke:** `import pid...` plus a tiny tensor round-trip on mps through
   `device_utils` (allocate, op, `.cpu()`), asserting finite output.
2. **`from_clean` + flux (first green gate):** decode a small clean image at low `--scale`
   on mps/fp32; assert finite output tensor and a written image file.
3. **`from_ldm` + flux:** short run (few LDM steps) → PiD decode at 2K on mps/fp32; eyeball
   the saved image.
4. **CUDA non-regression:** confirm by inspection that on a cuda host `get_device()` →
   cuda, `resolve_dtype()` → bf16, and the effective literals/flags match pre-port behavior.
5. fp32 is the gating dtype. A bf16 A/B on MPS is recorded as a follow-up note, not a
   blocker.

## Risks & Mitigations

- **Silent MPS op fallback to CPU** mid-graph (perf cliff): `PYTORCH_ENABLE_MPS_FALLBACK=1`
  keeps correctness; perf is a later concern. fp32 default avoids the worst bf16-coverage
  gaps.
- **A diffusers backbone hardcodes cuda internally** (outside our seams): out of scope for
  the first pass; flux is the validated path. Other backbones are best-effort.
- **Weights are bf16 on disk:** loading into fp32 upcasts (more memory, fine on 128 GB);
  `model.to(dtype)` already exists in the loader.

## Out of scope / follow-ups

- bf16-on-MPS speed validation and A/B numbers.
- Tiling / low-memory paths for ≤32 GB Macs.
- Backbones beyond flux (sd3, sdxl, qwenimage, flux2, zimage, dinov2, siglip).
