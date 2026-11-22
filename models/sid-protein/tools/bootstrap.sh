#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache fixtures weights
if [ ! -d .cache/upstream/.git ]; then
  git clone https://github.com/LY-Xie/SiD_Protein.git .cache/upstream
fi
git -C .cache/upstream checkout dfef6a29d975451144e8b0449c12cbea28b4ae15
uv venv .venv
uv pip install --python .venv/bin/python -r tools/requirements-export.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --link-mode hardlink
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_sid_protein.py
.venv/bin/python tools/dump_oracle.py
