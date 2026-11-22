"""Strict, unchanged upstream models; all filesystem output stays in this repo."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
import sys
import subprocess
from download_weights import SOURCES, digest
VERIFIED = {}
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from antifold.esm import pretrained
from antifold.esm.inverse_folding.util import CoordBatchConverter
COMMIT = '9918344067f34a6b026f2634fdba212fc3e6e986'
torch.set_num_threads(8)
torch.set_grad_enabled(False)

def model_for(variant):
    subprocess.run(['git','diff','--exit-code',COMMIT,'--','antifold/'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    checkpoint_path=ROOT/'.cache'/f'{variant}.pt'
    stamp=(checkpoint_path.stat().st_mtime_ns,checkpoint_path.stat().st_size)
    if VERIFIED.get(variant)!=stamp:
        if digest(checkpoint_path)!=SOURCES[variant][1]:raise RuntimeError(f'Checkpoint checksum mismatch: {variant}')
        VERIFIED[variant]=stamp
    if variant == 'antifold':
        model, alphabet = pretrained._load_IF1_local()
        state = torch.load(ROOT / '.cache/antifold.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    elif variant == 'esmif1':
        checkpoint = torch.load(ROOT / '.cache/esmif1.pt', map_location='cpu', weights_only=False)
        model, alphabet, state = pretrained._load_model_and_alphabet_core_v1(checkpoint)
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError(variant)
    return model.float().eval(), alphabet

def batch(case, alphabet):
    coords = np.array(case['coords'], dtype=np.float32).reshape(-1,3,3)
    confidence = case.get('confidence')
    c, cf, _, t, p = CoordBatchConverter(alphabet)([(coords, confidence, case['sequence'])])
    return c, p, cf, t

def relrms(a,b):
    a,b = np.asarray(a, dtype=np.float64),np.asarray(b,dtype=np.float64)
    return float(np.sqrt(np.square(a-b).sum() / max(np.square(b).sum(), 1e-30)))
