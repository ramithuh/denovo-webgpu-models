"""Import the unmodified pinned model directly (avoids unrelated plotting/data imports)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from download_weights import PATH, verify

ROOT = Path(__file__).resolve().parents[1]
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!'
COMMITS = {'evodiff': '33206e99446f799ec11cf6e57d66ffaec837be91',
           'protein-sequence-models': 'af695772c4a1c056d930c95ec7e6428aa042f5cd'}
for repo, commit in COMMITS.items():
    path = ROOT / '.cache/upstream' / repo
    assert subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() == commit
sys.path.insert(0, str(ROOT / '.cache/upstream/protein-sequence-models'))
spec = importlib.util.spec_from_file_location('pinned_evodiff_model', ROOT / '.cache/upstream/evodiff/evodiff/model.py')
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

def model(state=None):
    m = upstream.ByteNetLMTime(31, 8, 1024, 16, 5, 128, padding_idx=28,
        final_ln=True, slim=True, activation='gelu', timesteps=None).eval()
    if state is None:
        verify(PATH)
        state = {k.removeprefix('module.'): v for k, v in torch.load(PATH, map_location='cpu', mmap=True, weights_only=True)['model_state_dict'].items()}
    m.load_state_dict(state, strict=True)
    return m

def reload_bundle(precision):
    folder = ROOT / 'weights/oa_dm_38m' / precision
    manifest = json.loads((folder / 'manifest.json').read_text())
    raw = {}
    for shard in manifest['shards']:
        raw.update(load_file(folder / shard['file']))
    state = {}
    for name, rec in manifest['tensors'].items():
        t = raw[name].float()
        if rec['dtype'] == 'I8':
            t *= raw[name + '.__scale'].reshape(-1, *([1] * (t.ndim - 1)))
        state[name] = t
    return state

def tokens(s):
    return torch.tensor([[ALPHABET.index(c) for c in s]], dtype=torch.long)

def stats(a, b):
    a, b = a.detach().double(), b.detach().double()
    return {'relRMS': ((a-b).square().sum()/b.square().sum()).sqrt().item(), 'maxAbs': (a-b).abs().max().item()}

A_CASES = [
    ('gfp', 'MSK#EELFTGVVPILVELDGDVNGH#FSVSGE'),
    ('ubiquitin', 'MQI#VKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'),
    ('specials', ALPHABET),
    ('padding_mask', 'MSK#EELF!!!!'),
    # Length 257 exercises both +/- 2*128 taps at the maximum dilation.
    ('long_dilation', ''.join('#' if i % 17 == 0 else ALPHABET[i % 20] for i in range(257))),
]

def input_mask(name, seq):
    return (tokens(seq) != 30).float().unsqueeze(-1) if name == 'padding_mask' else None
