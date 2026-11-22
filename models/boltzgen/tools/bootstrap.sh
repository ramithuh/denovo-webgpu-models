#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/home/ruh/.local/bin:/home/ruh/.nvm/versions/node/v22.21.1/bin:$PATH"
export UV_CACHE_DIR="$PWD/.cache/uv"
export TMPDIR="$PWD/.cache/tmp"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
mkdir -p .cache/tmp
if [ ! -d .cache/upstream/.git ]; then
  git clone https://github.com/HannesStark/boltzgen.git .cache/upstream
fi
git -C .cache/upstream checkout 6472b44e8072187e6eaa51271ae6838245206013
if [ ! -x .venv/bin/python ]; then uv venv .venv --python 3.11; fi
uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match -r tools/requirements-export.txt
.venv/bin/python tools/audit.py
.venv/bin/python tools/export_boltzgen.py
node tools/prepare_inputs.mjs
for precision in fp32 fp16 int8; do
  .venv/bin/python tools/dump_oracle.py --precision "$precision"
done
.venv/bin/python tools/dump_oracle.py --variant adherence
.venv/bin/python tools/quantization_metrics.py
npm install --ignore-scripts --cache .cache/npm
