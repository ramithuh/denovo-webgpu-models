#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p .cache weights fixtures
if [ ! -d .cache/upstream/.git ]; then
  git clone https://github.com/NVIDIA-BioNeMo/la-proteina.git .cache/upstream
fi
git -C .cache/upstream checkout cde5de3ead6e4d76f367da6dc5174be9913ef6ca
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python --link-mode hardlink --index-url https://download.pytorch.org/whl/cpu torch==2.9.0+cpu
uv pip install --python .venv/bin/python --link-mode hardlink -r tools/requirements-export.txt
.venv/bin/python tools/download_weights.py LD1 AE1
.venv/bin/python tools/export_laproteina.py
.venv/bin/python tools/dump_oracle.py --trace
.venv/bin/python tools/dump_oracle.py --lengths 47 --steps 7 --output extended_oracles.json
