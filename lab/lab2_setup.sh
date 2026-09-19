#!/usr/bin/env bash
# numerics lab: a recent Triton whose CPU interpreter handles bf16, tensor loop bounds and libdevice
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
[ -x "$HOME/lab2/bin/python" ] || uv venv -q --python 3.11 "$HOME/lab2"
VIRTUAL_ENV="$HOME/lab2" uv pip install -q torch==2.8.0 triton==3.4.0 transformers==4.51.3 \
  safetensors tokenizers==0.21.1 numpy setuptools \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
"$HOME/lab2/bin/python" -c 'import torch, triton, transformers; print("lab2:", torch.__version__, triton.__version__, transformers.__version__)'
