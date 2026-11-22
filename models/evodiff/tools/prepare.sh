#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
mkdir -p .cache/upstream
for spec in 'evodiff 33206e99446f799ec11cf6e57d66ffaec837be91' 'protein-sequence-models af695772c4a1c056d930c95ec7e6428aa042f5cd'; do
  read -r repo commit <<< "$spec"
  if [[ ! -d ".cache/upstream/$repo/.git" ]]; then
    git clone "https://github.com/microsoft/$repo" ".cache/upstream/$repo"
  fi
  git -C ".cache/upstream/$repo" checkout --detach "$commit"
done
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r tools/requirements-export.txt
.venv/bin/python tools/download_weights.py
.venv/bin/python tools/export_evodiff.py
.venv/bin/python tools/dump_oracle.py
