#!/usr/bin/env bash
# start.sh — activate the venv and launch the Gradio app.

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .venv/bin/activate ]; then
    echo "Run ./install.sh first." >&2
    exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

# Mac (MPS) memory tuning. Disable the default upper watermark (which assumes
# isolated GPU memory) and let torch.mps.set_per_process_memory_fraction() in
# the Python process be the single source of truth for the MPS cap.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3

# Allow PyTorch to fall back to CPU for ops not yet implemented on MPS.
# Avoids hard-failing mid-training on a single missing kernel.
export PYTORCH_ENABLE_MPS_FALLBACK=1

exec python CoppyLora_webUI.py "$@"
