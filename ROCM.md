# SageAttention and FlashAttention on AMD GPUs (ROCm) for ComfyUI

This fork makes SageAttention work on AMD Radeon GPUs under Linux with ROCm. It also installs a tuned FlashAttention build. Both plug into ComfyUI's existing `--use-sage-attention` and `--use-flash-attention` flags.

## What works

| Feature | Status on AMD |
|---|---|
| SageAttention Triton kernel (INT8 QK, FP16 PV) | Works, with fp16, bf16 and fp32 inputs |
| FlashAttention 2, Triton AMD backend | Works |
| SageAttention2 CUDA kernels | Not possible. They use NVIDIA-only PTX instructions. |
| SageAttention3 | Not possible. It needs FP4 tensor cores that only NVIDIA Blackwell GPUs have. |

The Triton kernel ships in the SageAttention2 package, but algorithmically it is close to the original SageAttention. SageAttention2's finer-grained INT8 quantization and FP8 PV only exist in the CUDA kernels, which need NVIDIA GPUs.

On RDNA2, PyTorch has no fast attention kernel. It falls back to a slow kernel that uses a lot of memory. Both SageAttention and FlashAttention are faster, and they run sequence lengths that make PyTorch run out of memory.

## Tested setup

- RX 6800 XT (gfx1030), 16 GB
- Ubuntu 24.04, ROCm 7.2
- PyTorch 2.14.0+rocm7.2, triton-rocm 3.8.0, Python 3.12
- ComfyUI 0.37 in a conda env

Other RDNA2 cards should work. For RX 6700 XT, 6600 and similar (gfx1031, gfx1032), set `HSA_OVERRIDE_GFX_VERSION=10.3.0`. RDNA3 and RDNA4 cards should also work, but their speed has not been tested or tuned.

## Install

Activate your ComfyUI Python environment first. Then run:

```bash
git clone https://github.com/Aceeton/SageAttention_RDNA2_GFX1030.git
cd SageAttention_RDNA2_GFX1030
bash rocm/install_comfyui_rocm.sh
```

The script does four things:

1. It checks that PyTorch is a ROCm build and can see your GPU.
2. It repairs Triton if it is broken. See the troubleshooting section below.
3. It installs SageAttention. The CUDA extensions are skipped automatically on ROCm.
4. It installs FlashAttention 2.8.0 from [ROCm/flash-attention](https://github.com/ROCm/flash-attention) with the RDNA2 tuning patch from `rocm/flash-attention-rdna.patch`, then checks both against PyTorch.

Options:

- `--no-flash` installs SageAttention only.
- `--editable` installs SageAttention in editable mode, for development.

Nothing is compiled, so the install takes about a minute.

## Use it in ComfyUI

SageAttention, which is faster on RDNA2 in most cases:

```bash
python main.py --use-sage-attention
```

FlashAttention needs an environment variable set before launch:

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
python main.py --use-flash-attention
```

The ComfyUI log should say `Using sage attention` at startup. If a model's attention call fails, ComfyUI logs `Error running sage attention` and falls back to PyTorch.

## Benchmark

These are RX 6800 XT numbers for fp16, batch 1, 24 heads, non-causal. Times are milliseconds per attention call, so lower is better.

| Head dim | Seq len | PyTorch | SageAttention | FlashAttention |
|---|---|---|---|---|
| 64 | 4096 | 37.6 | 10.0 | 12.2 |
| 64 | 8192 | 164 | 39.7 | 48.9 |
| 128 | 4096 | 42.2 | 26.1 | 23.8 |
| 128 | 8192 | 164 | 102 | 95.3 |
| 128 | 16384 | out of memory | 364 | 386 |

Run the benchmark and accuracy tests on your own card:

```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python bench/test_rocm.py --flash
```

## Troubleshooting

**`RuntimeError: Should never be installed` when importing triton.** Installing PyTorch from PyPI instead of the PyTorch ROCm index can pull in a placeholder package called `pytorch-triton-rocm 0.0.1`. It overwrites Triton with a file that only raises this error. The install script fixes it. To fix it by hand, with the ROCm version matching your PyTorch:

```bash
pip uninstall -y pytorch-triton-rocm
pip install --force-reinstall --no-deps triton-rocm==3.8.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

**`No module named 'flash_attn_2_cuda'`.** `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` is not set in the shell that launched ComfyUI.

**Masked attention is not accelerated.** ComfyUI sends attention calls that carry a mask to PyTorch instead of SageAttention. Most image and video models do not use masks.

**Tuning for a different GPU.** SageAttention's launch settings can be overridden with environment variables. Try values such as 4 or 8 warps and 1 or 2 stages, then rerun the benchmark:

```bash
SAGEATTN_NUM_WARPS=8 SAGEATTN_NUM_STAGES=2 python bench/test_rocm.py --skip-accuracy
```

## What changed compared to upstream SageAttention

- The package imports without the compiled CUDA extensions, and `sageattn()` always uses the Triton kernel on ROCm.
- The Triton kernel accepts fp32 inputs. On RDNA2, ComfyUI runs models without fp16 support in fp32, because it avoids bf16 on these cards. Upstream rejected those calls, and ComfyUI silently fell back to PyTorch attention.
- The Triton launch settings on ROCm are 8 warps and 2 pipeline stages. The NVIDIA settings were 2 to 4 times slower on RDNA2. The settings are in `sageattention/triton/_launch.py`.
- `setup.py` skips the CUDA extensions automatically when PyTorch is a ROCm build.
- The FlashAttention patch sets the same tuned block configuration on RDNA2. It also fixes a crash when autotuning is enabled on non-CDNA GPUs.

NVIDIA behavior is unchanged.

Vibe Coded with Claude Opus 5
