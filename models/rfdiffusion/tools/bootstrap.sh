#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
port_root="$PWD"
mkdir -p "$port_root/.cache/tmp"
export TMPDIR="$port_root/.cache/tmp"
export UV_PYTHON_INSTALL_DIR="$port_root/.cache/uv-python"
export UV_CACHE_DIR="$port_root/.cache/uv"
export XDG_CACHE_HOME="$port_root/.cache"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -d .cache/upstream/.git ]]; then
  git clone https://github.com/RosettaCommons/RFdiffusion.git .cache/upstream
fi
git -C .cache/upstream checkout 86507b6538f51fce57b5a72477165f03999ed7ae
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r tools/requirements-export.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_rfdiffusion.py
.venv/bin/python tools/export_constants.py
.venv/bin/python tools/export_diffusion.py
.venv/bin/python tools/build_fixtures.py
.venv/bin/python tools/dump_primitives.py
npm ci --ignore-scripts --cache .cache/npm
