#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

node evodiff/tokenizer.test.mjs
node rfdiffusion/symmetry.test.mjs
node esm3/check-input.mjs
python3 boltzgen/inspect_archive.py
