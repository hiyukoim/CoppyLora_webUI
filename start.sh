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

exec python CoppyLora_webUI.py "$@"
