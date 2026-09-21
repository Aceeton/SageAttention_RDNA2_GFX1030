#!/usr/bin/env bash
# Install SageAttention (Triton) and FlashAttention (Triton AMD backend) into the
# active Python environment for ComfyUI on AMD GPUs with ROCm.
#
# Usage (inside your activated ComfyUI conda env or venv):
#   bash rocm/install_comfyui_rocm.sh             # SageAttention + FlashAttention
#   bash rocm/install_comfyui_rocm.sh --no-flash  # SageAttention only
#   bash rocm/install_comfyui_rocm.sh --editable  # editable SageAttention install (for development)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
FLASH_TAG="v2.8.0-cktile"
FLASH_REPO="https://github.com/ROCm/flash-attention"
INSTALL_FLASH=1
EDITABLE=""

for arg in "$@"; do
    case "$arg" in
        --no-flash) INSTALL_FLASH=0 ;;
        --editable) EDITABLE="-e" ;;
        -h|--help) sed -n 2,9p "$0"; exit 0 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*"; exit 1; }
pip() { "$PYTHON" -m pip "$@"; }

step "Checking PyTorch"
"$PYTHON" - <<'EOF' || die "A ROCm build of PyTorch with a visible GPU is required."
import torch
assert torch.version.hip, f"PyTorch {torch.__version__} is not a ROCm build"
assert torch.cuda.is_available(), "No GPU visible to PyTorch"
print(f"PyTorch {torch.__version__}, HIP {torch.version.hip}, {torch.cuda.get_device_name(0)}")
EOF

step "Checking Triton"
# Installing PyTorch from PyPI instead of the PyTorch ROCm index can pull in the
# placeholder package 'pytorch-triton-rocm 0.0.1'. It overwrites triton/__init__.py
# with 'raise RuntimeError("Should never be installed")'.
if "$PYTHON" -c "import triton" 2>/dev/null; then
    echo "Triton OK: $("$PYTHON" -c 'import triton; print(triton.__version__)')"
else
    echo "Triton is broken. Repairing it."
    read -r TRITON_PKG TRITON_SPEC INDEX_URL < <("$PYTHON" - <<'EOF'
import importlib.metadata as md
import torch
from packaging.requirements import Requirement

req = next(Requirement(r) for r in md.requires("torch") or [] if "triton" in r.split(";")[0])
try:
    installed = md.version(req.name)
except md.PackageNotFoundError:
    installed = None
# Keep the installed version unless it is the placeholder; otherwise use torch's pin.
spec = f"=={installed}" if installed and installed != "0.0.1" else str(req.specifier)
major, minor = torch.version.hip.split(".")[:2]
print(req.name, spec, f"https://download.pytorch.org/whl/rocm{major}.{minor}")
EOF
)
    PLACEHOLDER_INFO="$(pip show pytorch-triton-rocm 2>/dev/null || true)"
    if grep -q '^Version: 0.0.1$' <<<"$PLACEHOLDER_INFO"; then
        pip uninstall -y pytorch-triton-rocm
    fi
    pip install --force-reinstall --no-deps "${TRITON_PKG}${TRITON_SPEC}" --index-url "$INDEX_URL"
    "$PYTHON" -c "import triton; print('Triton OK:', triton.__version__)" \
        || die "Triton still does not import. Reinstall PyTorch from $INDEX_URL."
fi

step "Installing SageAttention from $REPO_DIR"
# setup.py detects ROCm and skips the CUDA extensions; only Triton kernels are used.
pip install --no-build-isolation --no-deps $EDITABLE "$REPO_DIR"

if [ "$INSTALL_FLASH" = 1 ]; then
    step "Installing FlashAttention ($FLASH_TAG, Triton AMD backend)"
    command -v git >/dev/null || die "git is required to install FlashAttention (or pass --no-flash)."
    "$PYTHON" -c "import einops" 2>/dev/null || pip install einops
    if pip show flash_attn >/dev/null 2>&1; then
        pip uninstall -y flash_attn
    fi
    BUILD_DIR="$(mktemp -d)"
    trap 'rm -rf "$BUILD_DIR"' EXIT
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$FLASH_TAG" "$FLASH_REPO" "$BUILD_DIR/flash-attention"
    git -C "$BUILD_DIR/flash-attention" apply "$REPO_DIR/rocm/flash-attention-rdna.patch"
    (cd "$BUILD_DIR/flash-attention" && \
        FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE "$PYTHON" -m pip install --no-build-isolation --no-deps .)
fi

step "Verifying"
cd /  # import the installed packages, not the source tree
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE INSTALL_FLASH="$INSTALL_FLASH" "$PYTHON" - <<'EOF'
import os, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F
from sageattention import sageattn

q, k, v = (torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.float16) for _ in range(3))
ref = F.scaled_dot_product_attention(q, k, v)
cos = lambda o: F.cosine_similarity(o.float().flatten(), ref.float().flatten(), dim=0).item()

c = cos(sageattn(q, k, v, tensor_layout="HND"))
print(f"SageAttention  cosine vs PyTorch: {c:.5f}")
assert c > 0.99, "SageAttention output is wrong"

if os.environ["INSTALL_FLASH"] == "1":
    from flash_attn import flash_attn_func
    o = flash_attn_func(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
    c = cos(o)
    print(f"FlashAttention cosine vs PyTorch: {c:.5f}")
    assert c > 0.99, "FlashAttention output is wrong"
EOF

step "Done"
cat <<'EOF'
Launch ComfyUI with SageAttention:
    python main.py --use-sage-attention

Or with FlashAttention (the environment variable is required):
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    python main.py --use-flash-attention
EOF
