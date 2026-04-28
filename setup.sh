#!/usr/bin/env bash
set -euo pipefail

# ── ensure pip installs go to container disk, not network volume ──────────────
unset PYTHONUSERBASE
unset PIP_TARGET

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing computebenchmark from ${REPO_DIR}"
pip install -e "${REPO_DIR}" --no-build-isolation -q

# ── detect GPU architecture and install the right flash-attention ─────────────
SM=$(python3 -c "import torch; print(torch.cuda.get_device_capability()[0])" 2>/dev/null || echo "0")

if [ "${SM}" -ge 10 ]; then
    echo "==> Blackwell (SM${SM}) detected — installing flash-attn-4"
    pip install "flash-attn-4[cu13]" -q

elif [ "${SM}" -ge 9 ]; then
    echo "==> Hopper (SM${SM}) detected — installing flash-attn-3"
    pip install flash-attn-3 \
        --find-links https://windreamer.github.io/flash-attention3-wheels/ \
        --no-build-isolation -q

elif [ "${SM}" -ge 8 ]; then
    echo "==> Ampere (SM${SM}) detected — installing flash-attn-2"
    pip install flash-attn --no-build-isolation -q

else
    echo "⚠  Could not detect GPU architecture (SM=${SM}). Skipping flash-attn install."
    echo "   Run manually: pip install flash-attn --no-build-isolation"
fi

# ── verify ────────────────────────────────────────────────────────────────────
echo ""
echo "==> Environment check"
python3 - <<'EOF'
import torch
print(f"  torch      {torch.__version__}")
print(f"  cuda       {torch.version.cuda}")
print(f"  GPU        {torch.cuda.get_device_name(0)}")
print(f"  SM         {torch.cuda.get_device_capability()}")

for pkg in ("flash_attn", "flash_attn_3", "flash_attn_4"):
    try:
        m = __import__(pkg)
        print(f"  {pkg:<14} {m.__version__}")
    except ImportError:
        pass

import computebenchmark
print(f"  computebenchmark installed ✓")
EOF

echo ""
echo "==> Done. Run benchmarks with:"
echo "    computebenchmark compute run --model-id <model-id> --output /workspace/results/compute.json"
echo "    computebenchmark algorithms train grpo --model-id <model-id> --output-dir /workspace/checkpoints"
