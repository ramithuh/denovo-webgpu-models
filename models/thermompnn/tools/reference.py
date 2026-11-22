"""Wrappers/hooks only. All inference executes unchanged pinned upstream modules."""
import sys
import subprocess
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from omegaconf import OmegaConf
from transfer_model import TransferModel
from protein_mpnn_utils import parse_PDB, tied_featurize
from types import SimpleNamespace
from download_weights import verify, COMMIT

torch.set_num_threads(4)
torch.manual_seed(0)
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

def load_model():
    verify()
    for source in ['transfer_model.py','protein_mpnn_utils.py','model_utils.py']:
        original=subprocess.check_output(['git','-C',str(ROOT),'show',f'{COMMIT}:{source}'])
        if hashlib.sha256(original).digest()!=hashlib.sha256((ROOT/source).read_bytes()).digest():
            raise ValueError(f'Upstream source changed: {source}')
    cfg = OmegaConf.create({'platform': {'thermompnn_dir': str(ROOT)}, 'model': {
        'hidden_dims': [64,32], 'subtract_mut': True, 'num_final_layers': 2,
        'freeze_weights': True, 'load_pretrained': True, 'lightattn': True}})
    model = TransferModel(cfg).eval()
    checkpoint = torch.load(ROOT/'models/thermoMPNN_default.pt', map_location='cpu', weights_only=False)
    model.load_state_dict({k.removeprefix('model.'): v for k,v in checkpoint['state_dict'].items()}, strict=True)
    return model

def specialized(state):
    result = {}
    for k,v in state.items():
        if k.startswith('light_attention.attention_convolution.'):
            continue
        if k == 'light_attention.feature_convolution.weight':
            v = v[:,:,4]
        result[k] = v.detach().float().contiguous()
    return result

def restore(model, tensors):
    state = model.state_dict()
    for k,v in tensors.items():
        if k == 'light_attention.feature_convolution.weight':
            state[k] = state[k].clone()
            state[k][:,:,4] = v
        else:
            state[k] = v
    model.load_state_dict(state, strict=True)

def run(model, case, full_ssm=False):
    pdb = parse_PDB(str(ROOT/'fixtures'/case['pdb']), case.get('chains'))[0]
    feat = tied_featurize([pdb], 'cpu', None)
    trace = dict(X=feat[0][0], S=feat[1][0].int(), mask=feat[2][0],
                 chain=feat[5][0].int(), residue_idx=feat[12][0].int())
    handles = []
    def hook(module, fn):
        handles.append(module.register_forward_hook(fn))
    def save(name, out):
        trace[name] = out.detach().squeeze(0).clone()
    mp = model.prot_mpnn
    hook(mp.features, lambda m,a,o: (save('features',o[0]),save('E_idx',o[1].int())) and None)
    hook(mp.features.edge_embedding, lambda m,a,o: save('geometry', a[0]))
    hook(mp.W_e, lambda m,a,o: save('edge_projection',o))
    hook(mp.W_s, lambda m,a,o: save('sequence_embedding',o))
    for i,l in enumerate(mp.encoder_layers):
        hook(l, lambda m,a,o,i=i: (save(f'encoder_{i}',o[0]), save(f'encoder_edges_{i}',o[1])) and None)
    for i,l in enumerate(mp.decoder_layers):
        hook(l, lambda m,a,o,i=i: save(f'decoder_{i}',o))
    hook(mp.W_out, lambda m,a,o: save('logits',o))
    head = {name: [] for name in ['light_attention','head_0','head_1','head_2','scores']}
    for name,mod in [('light_attention',model.light_attention),('head_0',model.both_out[1]),
                     ('head_1',model.both_out[3]),('head_2',model.both_out[5]),('scores',model.ddg_out)]:
        hook(mod, lambda m,a,o,name=name: head[name].append(o.detach().flatten().clone()))
    mutations = [SimpleNamespace(position=i, wildtype=ALPHABET[int(s)], mutation='A') for i,s in enumerate(feat[1][0])]
    try:
        with torch.no_grad():
            targeted,_ = model([pdb], mutations)
    finally:
        for h in handles: h.remove()
    for name, values in head.items():
        trace[name] = torch.stack(values)
    scores=trace['scores']
    trace['ddg'] = scores[:,:20] - scores.gather(1,feat[1][0,:,None])
    trace['targeted'] = torch.cat([p['ddG'] for p in targeted])
    torch.testing.assert_close(trace['targeted'],trace['ddg'][:,0],rtol=0,atol=0)
    if full_ssm:
        mutations = [SimpleNamespace(position=i,wildtype=ALPHABET[int(s)],mutation=aa)
                     for i,s in enumerate(feat[1][0]) for aa in ALPHABET[:20]]
        with torch.no_grad():
            direct,_ = model([pdb],mutations)
        torch.testing.assert_close(torch.cat([p['ddG'] for p in direct]).reshape(-1,20),trace['ddg'],rtol=0,atol=0)
    return {k: v.cpu().numpy().astype('<i4' if v.dtype in (torch.int32,torch.int64) else '<f4') for k,v in trace.items()}

def metric(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    return float(np.sqrt(np.sum((a-b)**2)/max(float(np.sum(b*b)),1e-30)))
