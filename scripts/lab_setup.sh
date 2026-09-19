#!/usr/bin/env bash
# Offline kernel lab (WSL / any Linux box, no GPU): the platform's exact Triton + torch (CPU build).
#   bash scripts/lab_setup.sh      -> ~/lab venv
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
[ -x "$HOME/lab/bin/python" ] || uv venv -q --python 3.11 "$HOME/lab"
VIRTUAL_ENV="$HOME/lab" uv pip install -q torch==2.5.1 triton==3.1.0 transformers==4.51.3 \
  safetensors==0.5.3 tokenizers==0.21.1 numpy pytest setuptools \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
"$HOME/lab/bin/python" -c 'import torch, triton, transformers; print("lab:", torch.__version__, triton.__version__, transformers.__version__)'
