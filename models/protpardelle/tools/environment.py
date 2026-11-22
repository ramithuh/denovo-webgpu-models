"""Confine all export/oracle writes to this checkout before importing upstream."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'ee378400f25b801fa481028000f9060183d7fb4c'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
for key, value in {
    'PROJECT_ROOT_DIR': ROOT,
    'PROTPARDELLE_MODEL_PARAMS': ROOT / '.cache/model_params',
    'PROTPARDELLE_OUTPUT_DIR': ROOT / '.cache/output',
    'XDG_CACHE_HOME': ROOT / '.cache',
    'TORCH_HOME': ROOT / '.cache/torch',
    'HF_HOME': ROOT / '.cache/huggingface',
    'TMPDIR': ROOT / '.cache/tmp',
}.items():
    os.environ[key] = str(value)
    Path(value).mkdir(parents=True, exist_ok=True)
