#!/usr/bin/env bash
# install.sh — Apple Silicon (macOS, MPS) installer for CoppyLora_webUI.
# Mirror of CoppyLora_webUI_install.ps1, adapted for Mac:
#   - PyTorch stable (Mac wheels carry MPS, no --index-url).
#   - sd-scripts pinned to the same upstream commit.
#   - Skips xformers, triton-windows, bitsandbytes (no Apple Silicon builds).
#   - Patches diffusers imports the same way the PowerShell installer does.
#   - Writes a Mac-friendly default accelerate config.
#
# Usage: ./install.sh
# Re-running is idempotent.

set -euo pipefail

cd "$(dirname "$0")"

export PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# 0. Prereq checks (fail fast with actionable messages)
# ---------------------------------------------------------------------------

if [ "$(uname -s)" != "Darwin" ]; then
    echo "ERROR: install.sh is for macOS. For Windows, use CoppyLora_webUI_install.ps1." >&2
    exit 1
fi

if [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: This installer targets Apple Silicon (arm64). Detected: $(uname -m)." >&2
    exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
    echo "ERROR: Xcode Command Line Tools are not installed." >&2
    echo "Run: xcode-select --install" >&2
    exit 1
fi

if ! command -v python3.11 >/dev/null 2>&1; then
    echo "ERROR: python3.11 not found on PATH." >&2
    echo "Install with: brew install python@3.11" >&2
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git not found on PATH." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Install uv if missing
# ---------------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# 2. Create Python 3.11 virtual environment
# ---------------------------------------------------------------------------

if [ ! -d .venv ]; then
    echo "Creating Python 3.11 virtual environment with uv..."
    uv venv --python 3.11 .venv
fi

# Activate venv for the rest of the script.
# shellcheck source=/dev/null
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 3. PyTorch (stable, MPS-enabled wheels)
# ---------------------------------------------------------------------------
# Mac wheels published to the default PyPI index already include MPS support.
# Do NOT pass --index-url here; the cu128 channel has no Apple Silicon wheels.

echo "Installing PyTorch (stable, with MPS)..."
uv pip install torch torchvision torchaudio

# ---------------------------------------------------------------------------
# 4. Clone sd-scripts at the upstream-pinned commit
# ---------------------------------------------------------------------------

SD_SCRIPTS_COMMIT="8b5ce3e641f0cf0775546922353105cb2d3a6895"

if [ ! -d sd-scripts ]; then
    git clone https://github.com/kohya-ss/sd-scripts.git
fi

(
    cd sd-scripts
    # Only checkout if we are not already on the pinned commit.
    current="$(git rev-parse HEAD)"
    if [ "$current" != "$SD_SCRIPTS_COMMIT" ]; then
        git fetch --quiet origin "$SD_SCRIPTS_COMMIT" 2>/dev/null || true
        git checkout "$SD_SCRIPTS_COMMIT"
    fi
)

# ---------------------------------------------------------------------------
# 5. sd-scripts requirements
# ---------------------------------------------------------------------------

echo "Installing sd-scripts requirements..."
(
    cd sd-scripts
    uv pip install -r requirements.txt
)

# ---------------------------------------------------------------------------
# 6. (Skipped on Mac)
#    xformers          — no Apple Silicon build; PyTorch SDPA is the fallback.
#    triton-windows    — Windows only.
#    bitsandbytes      — no native Apple Silicon support; AdamW is used instead.
#    onnxruntime-gpu   — no Apple Silicon GPU build; plain onnxruntime (CPU) is enough.
#    pyinstaller       — .app bundling deferred to v2.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7. Pin numpy and install other deps (mirroring install.ps1, minus skipped)
# ---------------------------------------------------------------------------

echo "Installing numpy 1.26.4..."
uv pip install numpy==1.26.4

echo "Installing additional packages..."
uv pip install wandb==0.17.3
# Upstream pin (gradio==4.3.0) was Nov 2023 and is incompatible with current
# starlette/pydantic releases. We bump to gradio 4.44.1 (Sep 2024) and pin its
# contemporaries explicitly, since unpinned installs resolve to incompatible
# 2025+ versions and fail at first request:
#   - pydantic >= 2.10 emits additionalProperties: True (bool); gradio_client's
#     get_type() does `if "const" in schema` and crashes on bool.
#   - starlette >= 0.40 / fastapi >= 0.116 changed TemplateResponse to require
#     (request, name, ctx); gradio 4.44.1 still calls (name, ctx).
uv pip install "gradio==4.44.1" "pydantic<2.10" "fastapi==0.115.0" "starlette==0.38.6"
uv pip install huggingface-hub==0.34.3
uv pip install onnx==1.15.0 onnxruntime==1.17.1
uv pip install toml

# ---------------------------------------------------------------------------
# 8. Patch diffusers imports in sd-scripts (same patch as install.ps1)
# ---------------------------------------------------------------------------

OLD_IMPORT="from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker"
NEW_IMPORT_1="from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput"
NEW_IMPORT_2="from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker"

patch_diffusers_import() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "Skipping patch — file not found: $file"
        return
    fi
    if ! grep -qF "$OLD_IMPORT" "$file"; then
        echo "No modification needed for $file."
        return
    fi
    # BSD sed: -i '' for in-place edit. The replacement uses python so we don't
    # have to escape regex metacharacters or wrestle with sed's newline handling.
    python3 - "$file" "$OLD_IMPORT" "$NEW_IMPORT_1" "$NEW_IMPORT_2" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
old = sys.argv[2]
new = sys.argv[3] + "\n" + sys.argv[4]
text = path.read_text()
path.write_text(text.replace(old, new))
PY
    echo "File $file has been modified."
}

patch_diffusers_import "sd-scripts/library/lpw_stable_diffusion.py"
patch_diffusers_import "sd-scripts/library/sdxl_lpw_stable_diffusion.py"

# ---------------------------------------------------------------------------
# 9. Default accelerate config (only if absent — never clobber user config)
# ---------------------------------------------------------------------------

ACCEL_DIR="$HOME/.cache/huggingface/accelerate"
ACCEL_FILE="$ACCEL_DIR/default_config.yaml"

if [ ! -f "$ACCEL_FILE" ]; then
    echo "Writing default accelerate config to $ACCEL_FILE..."
    mkdir -p "$ACCEL_DIR"
    cat > "$ACCEL_FILE" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: 'no'
use_cpu: false
debug: false
machine_rank: 0
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
YAML
else
    echo "Existing accelerate config found at $ACCEL_FILE — leaving untouched."
fi

# ---------------------------------------------------------------------------
# 10. Final summary
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Install summary"
echo "============================================================"
python -c "import torch; print(f'PyTorch     : {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
echo "sd-scripts  : $(git -C sd-scripts rev-parse --short HEAD)"
echo "venv        : $(pwd)/.venv"
echo ""
echo "Next steps:"
echo "  1. Download models manually per README_MAC.md."
echo "  2. Run ./start.sh to launch the Gradio app."
echo "============================================================"
