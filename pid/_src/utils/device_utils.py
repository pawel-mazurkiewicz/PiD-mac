# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend-agnostic device / dtype / RNG helpers.

This is the single place that branches on the compute backend. Inference code asks
here for a device, dtype, or generator and stays backend-agnostic, so the same code
runs on CUDA, Apple Silicon (MPS), or CPU.

Defaults are chosen for correctness/parity:
  * device resolution order: mps -> cuda -> cpu
  * dtype: bf16 on CUDA (upstream parity), fp32 on MPS and CPU
    (MPS bf16 has patchy op coverage and diverges numerically from CUDA, which
    compounds across a pixel-space diffusion decode; fp32 is the safe default).
The dtype can be overridden globally via `init_dtype(...)` / per call via `resolve_dtype`.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device, None]

_RESOLVED_DEVICE: Optional[torch.device] = None
_REQUESTED_DTYPE: Optional[str] = None

_DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
}


def _enable_mps_fallback() -> None:
    # Keep correctness when an op lacks an MPS kernel: fall back to CPU for that op.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _resolve_device(prefer: DeviceLike = None) -> torch.device:
    if prefer is not None and str(prefer) != "auto":
        dev = torch.device(prefer)
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    if dev.type == "mps":
        _enable_mps_fallback()
    return dev


def init_device(prefer: DeviceLike = None) -> torch.device:
    """Resolve and cache the process-wide device. Call once from the entrypoint."""
    global _RESOLVED_DEVICE
    _RESOLVED_DEVICE = _resolve_device(prefer)
    return _RESOLVED_DEVICE


def get_device(prefer: DeviceLike = None) -> torch.device:
    """Return the active device.

    An explicit `prefer` always wins. Otherwise return the cached process-wide device
    (resolving and caching it on first use).
    """
    if prefer is not None and str(prefer) != "auto":
        return _resolve_device(prefer)
    if _RESOLVED_DEVICE is not None:
        return _RESOLVED_DEVICE
    return init_device(None)


def init_dtype(requested: Optional[str]) -> None:
    """Set the process-wide requested dtype (from `--dtype`); None/"auto" = backend default."""
    global _REQUESTED_DTYPE
    _REQUESTED_DTYPE = requested if (requested and requested != "auto") else None


def resolve_dtype(device: DeviceLike = None, requested: Optional[str] = None) -> torch.dtype:
    """Resolve the compute dtype.

    Precedence: process-wide `init_dtype` (explicit user `--dtype`) > per-call `requested`
    (e.g. a per-experiment `config.precision`, used as the CUDA default for back-compat) >
    backend default (bf16 on CUDA, fp32 on MPS/CPU).
    """
    if isinstance(requested, torch.dtype):
        if not _REQUESTED_DTYPE:
            return requested
        requested = None
    name = _REQUESTED_DTYPE or (requested if (requested and requested != "auto") else None)
    if name:
        try:
            return _DTYPE_ALIASES[name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown dtype '{name}'. Valid: {sorted(_DTYPE_ALIASES)}") from exc
    dev = device if isinstance(device, torch.device) else get_device(device)
    return torch.bfloat16 if dev.type == "cuda" else torch.float32


def make_generator(device: DeviceLike = None, seed: Optional[int] = None) -> torch.Generator:
    """Create a seeded generator.

    MPS does not support device-resident generators reliably, so use a CPU generator on
    MPS (diffusers / `torch.randn(..., generator=g)` then place the sample on-device).
    """
    dev = device if isinstance(device, torch.device) else get_device(device)
    gen_device = "cpu" if dev.type == "mps" else dev
    generator = torch.Generator(device=gen_device)
    if seed is not None:
        generator.manual_seed(int(seed))
    return generator


def empty_cache(device: DeviceLike = None) -> None:
    dev = device if isinstance(device, torch.device) else get_device(device)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        torch.mps.empty_cache()


def synchronize(device: DeviceLike = None) -> None:
    dev = device if isinstance(device, torch.device) else get_device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def setup_backends(device: DeviceLike = None) -> None:
    """Enable backend-specific perf/correctness knobs that only apply to that backend."""
    dev = device if isinstance(device, torch.device) else get_device(device)
    if dev.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    elif dev.type == "mps":
        _enable_mps_fallback()


def mps_safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """scaled_dot_product_attention that stays correct on Apple Silicon (MPS).

    PyTorch's fused MPS SDPA kernel returns *increasingly wrong* outputs past a few
    thousand key tokens (a long-standing Metal kernel bug), which shows up as a regular
    grid/tile artifact in high-resolution outputs. On MPS we therefore compute attention
    explicitly — matmul -> softmax -> matmul in fp32 — chunked over the query dim to
    bound peak memory to [B, H, q_chunk, S] instead of the full [B, H, S, S]. This is
    exact at any sequence length. Off MPS we defer to the fused kernel.

    q/k/v are [..., S, head_dim] (the standard SDPA layout). dropout is ignored on the
    MPS path (inference only; callers pass dropout_p=0.0). is_causal is not used by PiD
    and is unsupported here.
    """
    import math

    import torch.nn.functional as F

    if query.device.type != "mps":
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
    if is_causal:
        raise NotImplementedError("mps_safe_sdpa does not implement is_causal (unused by PiD).")

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    seq_q = query.shape[-2]
    key_t = key.transpose(-2, -1)
    out = torch.empty(*query.shape[:-1], value.shape[-1], device=query.device, dtype=query.dtype)
    q_chunk = 1024
    for start in range(0, seq_q, q_chunk):
        end = min(start + q_chunk, seq_q)
        scores = (query[..., start:end, :].float() @ key_t.float()) * scale
        if attn_mask is not None:
            m = attn_mask
            if m.dim() == scores.dim() and m.shape[-2] not in (1, scores.shape[-2]):
                m = m[..., start:end, :]
            if m.dtype == torch.bool:
                scores = scores.masked_fill(~m, float("-inf"))
            else:
                scores = scores + m
        out[..., start:end, :] = (scores.softmax(dim=-1) @ value.float()).to(out.dtype)
        del scores
    return out
