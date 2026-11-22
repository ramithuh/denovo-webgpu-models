from pathlib import Path
import sys, json, torch, numpy as np
ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'.cache/upstream'
COMMIT='9a954578f7b5a39552545eebc6d4794447794c87'
sys.path.insert(0,str(UPSTREAM))
from genie.config import Config
from genie.model.model import Denoiser
from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames
from genie.diffusion.schedule import get_betas
from safetensors.torch import load_file
from download_weights import download

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
def config():
    c=Config(str(UPSTREAM/'results/base/configuration'))
    return dict(**c.model,n_timestep=c.diffusion['n_timestep'],max_n_res=c.io['max_n_res'],max_n_chain=c.io['max_n_chain'])
def original_state():
    s=torch.load(download(),map_location='cpu',weights_only=False)['state_dict']
    assert all(k.startswith('model.') for k in s)
    return {k[6:]:v.contiguous() for k,v in s.items()}
def model(state=None):
    torch.manual_seed(1234);np.random.seed(1234)
    m=Denoiser(**config()).eval()
    m.load_state_dict(original_state() if state is None else state,strict=True)
    return m

def quantize(s,precision):
    out={};scales={}
    for k,x in s.items():
        if x.ndim==2 and precision=='fp16': out[k]=x.half()
        elif x.ndim==2 and precision=='int8':
            scale=x.abs().amax(1).clamp_min(1e-30)/127
            out[k]=(x/scale[:,None]).round().clamp(-127,127).to(torch.int8);scales[k]=scale
        else: out[k]=x.clone()
    decoded={k:(x.float()*scales[k][:,None] if k in scales else x.float()) for k,x in out.items()}
    return out,scales,decoded

def reload_bundle(precision):
    folder=ROOT/'weights'/precision;j=json.loads((folder/'manifest.json').read_text())
    shards={s['file']:load_file(str(folder/s['file'])) for s in j['shards']};state={}
    for k,r in j['tensors'].items():
        parts=[]
        for p in r.get('parts',[r]):
            t=shards[p['shard']][p.get('key',k)].float()
            if p['dtype']=='I8': t=t*shards[p['shard']][p['quant']['scale_tensor']][:,None]
            parts.append(t)
        state[k]=torch.cat(parts,dim=0)
    return state

def stats(a,b):
    a=a.double();b=b.double();d=a-b
    return {'relRMS':float(torch.sqrt((d*d).sum()/(b*b).sum().clamp_min(1e-30))),'maxAbs':float(d.abs().max())}

def features(L,mode):
    f={'aatype':torch.zeros(1,L,20),'atom_positions':torch.zeros(1,L,3),'residue_mask':torch.ones(1,L),'residue_index':torch.arange(L)[None], 'chain_index':torch.zeros(1,L,dtype=torch.long), 'fixed_sequence_mask':torch.zeros(1,L), 'fixed_structure_mask':torch.zeros(1,L,L),'fixed_group':torch.zeros(1,L),'interface_mask':torch.zeros(1,L)}
    if mode!='unconditional':
        segments=[(3,7,1)] if mode=='single_motif' else [(3,7,1),(L-8,L-4,2)]
        for a,b,g in segments:
            for i in range(a,b):
                f['fixed_sequence_mask'][0,i]=1;f['fixed_group'][0,i]=g;f['aatype'][0,i,(i*7)%20]=1
                f['atom_positions'][0,i]=torch.tensor([3.8*(i-a),2*np.sin(i),2*np.cos(i)])
        group=f['fixed_group'];mask=f['fixed_sequence_mask']
        f['fixed_structure_mask']=(group[:,:,None]==group[:,None,:]).float()*mask[:,:,None]*mask[:,None,:]
    return f

def frame(x,f): return T(compute_frenet_frames(x,f['chain_index'],f['residue_mask']),x)
def schedule(steps):
    beta=get_betas(steps,'cosine');alpha=1-beta;cum=torch.cumprod(alpha,0)
    return beta,alpha,cum
