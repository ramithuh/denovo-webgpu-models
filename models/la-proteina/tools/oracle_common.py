"""Pinned upstream modules, without training/evaluation dependency installation.

Only import-time adapters are installed for Lightning's Module base and the
unused torch_scatter import. No neural or sampling implementation is replaced.
"""
import sys
import types
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.cache/upstream'
COMMIT = 'cde5de3ead6e4d76f367da6dc5174be9913ef6ca'
sys.path.insert(0, str(UPSTREAM))
torch.set_num_threads(8)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False

def unused(*args, **kwargs):
    raise RuntimeError('Unused training/conditional dependency entered the oracle')

sys.modules.setdefault('torch_scatter', types.SimpleNamespace(scatter_mean=unused))
sys.modules.setdefault('lightning', types.SimpleNamespace(LightningModule=torch.nn.Module))

from omegaconf import OmegaConf
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer
from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher, get_schedule, get_gt
from download_weights import CHECKPOINTS

def checkpoint(model):
    return torch.load(ROOT / 'weights' / CHECKPOINTS[model][0], map_location='cpu', weights_only=False, mmap=True)

def stats(a, b):
    a, b = a.double(), b.double()
    diff = a-b
    return {'relRMS': (diff.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30)).item(), 'maxAbs': diff.abs().max().item()}

def quantize(t, precision):
    t = t.detach().float().contiguous()
    if t.ndim != 2 or precision == 'fp32': return t, None
    if precision == 'fp16': return t.half(), None
    if precision != 'int8': raise ValueError(precision)
    scale = t.abs().amax(1).clamp_min(1e-30) / 127
    return (t / scale[:, None]).round().clamp(-127,127).to(torch.int8), scale

def dequantize(t, scale):
    return t.float() if scale is None else t.float() * scale[:, None]

def model_state(model):
    ckpt = checkpoint(model)
    if model == 'LD1':
        cfg = OmegaConf.to_container(ckpt['hyper_parameters']['cfg_exp'].nn, resolve=True)
        cfg['latent_dim'] = 8
        prefix = 'nn.'
    else:
        hp = ckpt['hyper_parameters']
        cfg = OmegaConf.to_container(hp['cfg_ae'].nn_ae, resolve=True)
        prefix = 'decoder.'
    state = {k[len(prefix):]: v for k,v in ckpt['state_dict'].items() if k.startswith(prefix)}
    assert state
    return cfg, state

def make_model(model, precision='fp32', state=None):
    cfg, original = model_state(model)
    net = LocalLatentsTransformer(**cfg) if model == 'LD1' else DecoderTransformer(**cfg)
    if state is None:
        state = {k: dequantize(*quantize(v,precision)) for k,v in original.items()}
    net.load_state_dict(state, strict=True)
    return net.eval(), cfg

def reload_bundle(model, precision):
    from safetensors.torch import load_file
    folder = ROOT / 'weights' / model.lower() / precision
    manifest = json.loads((folder/'manifest.json').read_text())
    shards = {s['file']: load_file(folder/s['file']) for s in manifest['shards']}
    out = {}
    for name,entry in manifest['tensors'].items():
        parts=[]
        for part in entry['parts']:
            s=shards[part['shard']]
            parts.append(dequantize(s[part['key']],s.get(part.get('scale_tensor',''))))
        out[name]=torch.cat(parts,dim=0).reshape(entry['shape'])
    return out
