"""Execute the pinned upstream modules directly; no captured graph or rewritten network."""
import sys, types, json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.cache/upstream'
COMMIT = 'f50d8dbbdae827be291e9f73d732b61b195f8816'
sys.path.insert(0, str(UPSTREAM))
# scatter is used only by training's alignment routines, never by FlowModel or sampling.
# Make accidentally using it a hard failure rather than silently substituting arithmetic.
scatter = types.ModuleType('torch_scatter')
def unavailable(*a, **kw): raise RuntimeError('Training-only torch_scatter is excluded from inference export')
scatter.scatter_add = scatter.scatter = unavailable
sys.modules['torch_scatter'] = scatter
from models.flow_model import FlowModel
from data import all_atom, so3_utils
from openfold.utils.rigid_utils import Rigid, Rotation

torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)
def checkpoint():
    return torch.load(ROOT/'weights/published.ckpt', map_location='cpu', weights_only=False)
def model(state=None):
    ck = checkpoint()
    cfg = ck['hyper_parameters']['cfg']
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    net = FlowModel(cfg.model).eval()
    original = {k.removeprefix('model.'):v for k,v in ck['state_dict'].items()}
    net.load_state_dict(original if state is None else state, strict=True)
    return net, OmegaConf.to_container(cfg.model, resolve=True)
def quantize(state, precision):
    out, packed = {}, {}
    for n, v in state.items():
        v = v.detach().cpu().float().contiguous()
        scale = None
        if v.ndim == 2 and precision == 'fp16': p = v.half(); d = p.float()
        elif v.ndim == 2 and precision == 'int8':
            scale = v.abs().amax(1).clamp_min(1e-30)/127
            p = (v/scale[:,None]).round().clamp(-127,127).to(torch.int8)
            d = p.float()*scale[:,None]
        else: p = v; d = v
        out[n] = d; packed[n] = (p,scale)
    return out, packed

def reload_bundle(precision):
    folder = ROOT/'weights'/precision
    m = json.loads((folder/'manifest.json').read_text()); shards = {}
    for s in m['shards']: shards[s['file']] = load_file(folder/s['file'])
    out = {}
    for n,r in m['tensors'].items():
        parts = []
        for p in r.get('parts',[r]):
            sh = shards[p['shard']]; t = sh[p.get('key',n)].float()
            if p['dtype']=='I8': t = t*sh[p['quant']['scale_tensor']][:,None]
            parts.append(t)
        out[n] = torch.cat(parts,dim=0)
    return out

def stats(a,b):
    a,b=a.double(),b.double()
    return {'relRms': float(((a-b).square().sum()/b.square().sum().clamp_min(1e-30)).sqrt()), 'maxAbs': float((a-b).abs().max())}
