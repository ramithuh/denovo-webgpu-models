"""Pinned paths and process-local caches. Import before third party modules."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.cache/upstream'
COMMIT = '6472b44e8072187e6eaa51271ae6838245206013'
REVISION = 'c1be29e1f82ffcc72264f64b993c43fb4e0d17f0'
CHECKPOINTS = Path('/home/ruh/.cache/huggingface/hub/models--boltzgen--boltzgen-1/snapshots') / REVISION
MOLS = Path('/home/ruh/.cache/huggingface/hub/datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip')
for key, name in [('TMPDIR', 'tmp'), ('MPLCONFIGDIR', 'matplotlib'), ('NUMBA_CACHE_DIR', 'numba'), ('XDG_CACHE_HOME', 'xdg')]:
    path = ROOT / '.cache' / name
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(UPSTREAM / 'src'))

def checkpoint(variant='diverse'):
    import torch
    return torch.load(CHECKPOINTS / f'boltzgen1_{variant}.ckpt', map_location='cpu', weights_only=False, mmap=True)

def load_model(variant='diverse'):
    import inspect
    import torch
    from boltzgen.model.models.boltz import Boltz
    ckpt = checkpoint(variant)
    hp = dict(ckpt['hyper_parameters'])
    hp = {k: v for k, v in hp.items() if k in inspect.signature(Boltz.__init__).parameters}
    hp.update(validate_structure=False, validators=None, use_kernels=False, checkpoint_diffusion_conditioning=False)
    # Do not initialize a second 1.9 GB parameter copy.
    model = Boltz(**hp).eval()
    model.on_load_checkpoint(ckpt)
    model.load_state_dict(ckpt['state_dict'], strict=True, assign=True)
    model.requires_grad_(False)
    return model, ckpt
