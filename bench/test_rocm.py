"""
Accuracy and speed check for SageAttention (and optionally FlashAttention) on AMD GPUs.

Compares against torch.nn.functional.scaled_dot_product_attention, which on
RDNA2 (gfx1030) falls back to the MATH backend.

Usage:
    python bench/test_rocm.py                 # accuracy + timings
    python bench/test_rocm.py --quick         # small subset
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python bench/test_rocm.py --flash
"""

import argparse
import itertools
import warnings

import torch
import torch.nn.functional as F

from sageattention import sageattn

warnings.filterwarnings("ignore")


def to_hnd(x, layout):
    return x.transpose(1, 2) if layout == "NHD" else x


def reference(q, k, v, layout, is_causal, mask=None):
    q, k, v = (to_hnd(t, layout) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
    return to_hnd(o, layout)


def make_qkv(b, h, s, d, dtype, layout):
    shape = (b, s, h, d) if layout == "NHD" else (b, h, s, d)
    return [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]


def compare(o, ref):
    o, ref = o.float().flatten(), ref.float().flatten()
    cos = F.cosine_similarity(o, ref, dim=0).item()
    return cos, (o - ref).abs().max().item()


def bench(fn, iters=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def flash_fn():
    from flash_attn import flash_attn_func
    return flash_attn_func


def accuracy(args):
    print("== Accuracy (cosine similarity vs SDPA; target > 0.99)")
    seqs = [1024] if args.quick else [1024, 4096]
    failures = 0
    for dtype, d, layout, causal, s in itertools.product(
            [torch.float16, torch.bfloat16], [64, 128], ["HND", "NHD"], [False, True], seqs):
        q, k, v = make_qkv(1, 8, s, d, dtype, layout)
        ref = reference(q, k, v, layout, causal)
        o = sageattn(q, k, v, tensor_layout=layout, is_causal=causal)
        cos, err = compare(o, ref)
        ok = cos > 0.99
        failures += not ok
        line = f"sage  {str(dtype):15s} d={d:3d} {layout} causal={causal!s:5s} s={s:5d}  cos={cos:.5f} maxerr={err:.4f}"
        if args.flash and layout == "NHD":
            of = flash_fn()(q, k, v, causal=causal)
            fcos, ferr = compare(of, ref)
            failures += not fcos > 0.99
            line += f" | flash cos={fcos:.5f} maxerr={ferr:.4f}"
        print(("OK   " if ok else "FAIL ") + line)

    # Boolean mask, as passed by ComfyUI (non-causal, NHD layout).
    q, k, v = make_qkv(1, 8, 1024, 64, torch.float16, "NHD")
    mask = torch.rand(1, 1, 1024, 1024, device="cuda") > 0.2
    ref = reference(q, k, v, "NHD", False, mask)
    o = sageattn(q, k, v, tensor_layout="NHD", attn_mask=mask, smooth_k=False)
    cos, err = compare(o, ref)
    failures += not cos > 0.99
    print(("OK   " if cos > 0.99 else "FAIL ") + f"sage  bool mask  cos={cos:.5f} maxerr={err:.4f}")
    return failures


def speed(args):
    print("\n== Speed (fp16, batch 1, 24 heads, non-causal, ms per call)")
    seqs = [4096] if args.quick else [1024, 4096, 8192, 16384]
    header = f"{'d':>4} {'seq':>6} {'sdpa':>9} {'sage':>9} {'speedup':>8}"
    if args.flash:
        header += f" {'flash':>9} {'speedup':>8}"
    print(header)
    for d, s in itertools.product([64, 128], seqs):
        q, k, v = make_qkv(1, 24, s, d, torch.float16, "HND")
        try:
            t_ref = bench(lambda: F.scaled_dot_product_attention(q, k, v))
        except torch.OutOfMemoryError:
            t_ref = float("nan")
        torch.cuda.empty_cache()
        t_sage = bench(lambda: sageattn(q, k, v, tensor_layout="HND"))
        line = f"{d:4d} {s:6d} {t_ref:9.2f} {t_sage:9.2f} {t_ref / t_sage:7.2f}x"
        if args.flash:
            qn, kn, vn = (t.transpose(1, 2).contiguous() for t in (q, k, v))
            t_flash = bench(lambda: flash_fn()(qn, kn, vn))
            line += f" {t_flash:9.2f} {t_ref / t_flash:7.2f}x"
        print(line)
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--flash", action="store_true", help="also test flash_attn_func")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    args = parser.parse_args()
    print(torch.__version__, "HIP", torch.version.hip, torch.cuda.get_device_name(0))
    failures = 0 if args.skip_accuracy else accuracy(args)
    if not args.skip_speed:
        speed(args)
    if failures:
        raise SystemExit(f"\n{failures} accuracy check(s) failed")
    print("\nAll accuracy checks passed" if not args.skip_accuracy else "")
