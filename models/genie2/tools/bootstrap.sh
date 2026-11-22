#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
mkdir -p .cache weights fixtures
if [[ ! -d .cache/upstream/.git ]]; then
 git clone https://github.com/aqlaboratory/genie2.git .cache/upstream
fi
git -C .cache/upstream checkout 9a954578f7b5a39552545eebc6d4794447794c87
uv venv .venv
uv pip install --python .venv/bin/python -r tools/requirements-export.txt
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_genie2.py
.venv/bin/python tools/dump_oracle.py
.venv/bin/python tools/check_sign_stability.py
