#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache weights fixtures
if [ ! -d .cache/upstream/.git ]; then
 git clone https://github.com/microsoft/protein-frame-flow.git .cache/upstream
fi
git -C .cache/upstream checkout --detach f50d8dbbdae827be291e9f73d732b61b195f8816
if [ ! -x .venv/bin/python ]; then uv venv .venv --python 3.12; fi
uv pip install --python .venv/bin/python -r tools/requirements-export.txt
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_frameflow.py
.venv/bin/python tools/dump_oracle.py
.venv/bin/python tools/dump_motif_oracle.py
.venv/bin/python tools/probe_modes.py
npm install --cache .cache/npm --ignore-scripts --no-audit --no-fund
