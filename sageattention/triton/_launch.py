"""
Triton launch parameters for the attention kernels.

The defaults were tuned on NVIDIA GPUs. On AMD GPUs (ROCm) deep software
pipelining (num_stages > 2) tends to fail to compile or spill registers,
so different defaults are used there.

Both values can be overridden with the environment variables
``SAGEATTN_NUM_WARPS`` and ``SAGEATTN_NUM_STAGES`` (useful for tuning).
"""

import os

import torch

IS_ROCM = getattr(torch.version, "hip", None) is not None

# Defaults for AMD GPUs, chosen by benchmarking on RDNA2 (gfx1030, RX 6800 XT):
# 8 warps / 2 stages was best or within ~10% of best for head_dim 64 and 128,
# causal and non-causal. 4 warps / 1 stage was 2-4x slower.
_ROCM_NUM_WARPS = 8
_ROCM_NUM_STAGES = 2


def _env_int(name):
    value = os.getenv(name)
    return int(value) if value else None


def launch_params(head_dim, cuda_num_warps, cuda_num_stages):
    """Return ``(num_warps, num_stages)`` for the current platform."""
    if IS_ROCM:
        num_warps, num_stages = _ROCM_NUM_WARPS, _ROCM_NUM_STAGES
    else:
        num_warps, num_stages = cuda_num_warps, cuda_num_stages
    num_warps = _env_int("SAGEATTN_NUM_WARPS") or num_warps
    num_stages = _env_int("SAGEATTN_NUM_STAGES") or num_stages
    return num_warps, num_stages
