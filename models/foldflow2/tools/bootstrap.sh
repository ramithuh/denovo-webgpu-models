#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache weights fixtures
if [ ! -d .cache/upstream/.git ]; then
  git clone --filter=blob:none --no-checkout https://github.com/DreamFold/FoldFlow.git .cache/upstream
fi
git -C .cache/upstream checkout 9d2c260813da3c9a2bc944973953f63ea6d71203
if [ ! -x .venv/bin/python ]; then
  uv venv .venv --python 3.12
fi
uv pip install --python .venv/bin/python --link-mode hardlink -r tools/requirements-export.txt
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_foldflow2.py
for precision in fp32 fp16 int8; do
  .venv/bin/python tools/dump_oracle.py --precision "$precision"
done
