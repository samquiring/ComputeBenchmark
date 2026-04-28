#!/usr/bin/env bash
set -euo pipefail

# ── ensure pip installs go to container disk, not network volume ──────────────
unset PYTHONUSERBASE
unset PIP_TARGET

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing computebenchmark from ${REPO_DIR}"
pip install -e "${REPO_DIR}" --no-build-isolation -q

# ── detect versions ───────────────────────────────────────────────────────────
SM=$(python3 -c "import torch; print(torch.cuda.get_device_capability()[0])" 2>/dev/null || echo "0")
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0].replace('.', ''))" 2>/dev/null || echo "")

echo "==> SM=${SM}  CUDA=${CUDA_VER}  PyTorch=${TORCH_VER}"

# ── install flash-attention ───────────────────────────────────────────────────
if [ "${SM}" -ge 10 ]; then
    echo "==> Blackwell (SM${SM}) — installing flash-attn-4"
    pip install "flash-attn-4[cu13]" -q

elif [ "${SM}" -ge 9 ]; then
    echo "==> Hopper (SM${SM}) — installing flash_attn_3 for cu${CUDA_VER} torch${TORCH_VER}"
    FIND_LINKS="https://windreamer.github.io/flash-attention3-wheels/cu${CUDA_VER}_torch${TORCH_VER}"
    echo "    find-links: ${FIND_LINKS}"

    if pip install flash_attn_3 --find-links "${FIND_LINKS}" -q 2>/dev/null; then
        echo "    installed via windreamer"
    else
        echo "⚠  FA3 prebuilt not found for cu${CUDA_VER}_torch${TORCH_VER}."
        echo "   Browse available indexes at: https://windreamer.github.io/flash-attention3-wheels/"
        echo "   Then run: pip install flash_attn_3 --find-links <matching-url>"
    fi

elif [ "${SM}" -ge 8 ]; then
    echo "==> Ampere (SM${SM}) — installing flash-attn-2"
    pip install flash-attn --no-build-isolation -q

else
    echo "⚠  Could not detect GPU (SM=${SM}). Skipping flash-attn."
fi

# ── verify ────────────────────────────────────────────────────────────────────
echo ""
echo "==> Environment"
python3 - <<'EOF'
import torch
print(f"  torch      {torch.__version__}")
print(f"  cuda       {torch.version.cuda}")
print(f"  GPU        {torch.cuda.get_device_name(0)}")
print(f"  SM         {torch.cuda.get_device_capability()}")

for pkg in ("flash_attn", "flash_attn_3", "flash_attn_4"):
    try:
        m = __import__(pkg)
        version = getattr(m, "__version__", "installed")
        print(f"  {pkg:<14} {version}")
    except ImportError:
        pass

import computebenchmark
print(f"  computebenchmark ✓")
EOF

echo ""
echo "==> Ready. Commands:"
echo "    computebenchmark algorithms race --model-id <id> --output-dir /workspace/results/race"
echo "    computebenchmark compute run --model-id <id> --output /workspace/results/compute.json"
