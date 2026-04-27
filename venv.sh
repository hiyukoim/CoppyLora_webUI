#!/usr/bin/env bash
# venv.sh — drop into a subshell with the project venv activated.
# Mac equivalent of venv.cmd.

set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .venv/bin/activate ]; then
    echo "Run ./install.sh first." >&2
    exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

exec "${SHELL:-/bin/zsh}"
