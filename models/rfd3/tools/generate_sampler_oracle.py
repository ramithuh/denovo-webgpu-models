#!/usr/bin/env python3
import copy,json
import torch
from unittest.mock import patch
from checkpoint import ROOT,load_checkpoint
from generate_oracle import tensor_json,pipeline_example
from export_weights import atomic_json
from rfd3.model.inference_sampler import SampleDiffusionWithMotif,SampleDiffusionWithSymmetry

torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
original_rand=torch.rand
checkpoint,net,_=load_checkpoint()
for module in net.modules():
    if hasattr(module,'force_bfloat16'):module.force_bfloat16=False
    if hasattr(module,'use_deepspeed_evo'):module.use_deepspeed_evo=False
net=net.cuda();result={}
def tensor(v):return torch.tensor(v['data'],device='cuda',dtype=torch.bool if 'bool' in v['dtype'] else torch.long if 'int' in v['dtype'] else torch.float32).reshape(v['shape'])
def features_json(f):
    result={k:tensor_json(v) for k,v in f.items() if isinstance(v,torch.Tensor)}
    if 'sym_transform' in f:result['sym_transform']={k:[r.cpu().flatten().tolist(),t.cpu().flatten().tolist()] for k,(r,t) in f['sym_transform'].items()}
    return result
with torch.inference_mode():
 for name in ['aligned-motif','symmetry']:
    if name=='aligned-motif':
        case=json.loads((ROOT/'fixtures/oracle-pass2/atomic-triad.json').read_text());f={k:tensor(v) for k,v in case['features'].items()}
        sampler=SampleDiffusionWithMotif(num_timesteps=4,n_recycle=2,allow_realignment=True)
    else:
        spec={'length':'4','symmetry':{'id':'C2','is_symmetric_motif':False}}
        ex=pipeline_example(checkpoint,spec,20261123);f={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in ex['feats'].items()}
        f['sym_transform']={k:(r.cuda(),t.cuda()) for k,(r,t) in f['sym_transform'].items()}
        sampler=SampleDiffusionWithSymmetry(num_timesteps=4,n_recycle=2,allow_realignment=False,gamma_0=0.6)
    init=net.token_initializer(copy.deepcopy(f));draws=[];angles=[];generator=torch.Generator().manual_seed(1701)
    def normal(mean,std,size,device,**kwargs):
        v=torch.randn(size,generator=generator)*std+mean;draws.append(tensor_json(v));return v.to(device)
    def uniform(size,**kwargs):
        v=original_rand(size,generator=generator);angles.append(tensor_json(v));return v
    base=f['motif_pos'][None].clone()
    with patch('torch.normal',normal),patch('torch.rand',uniform):
        out=sampler.sample_diffusion_like_af3(f=copy.deepcopy(f),diffusion_module=net.diffusion_module,diffusion_batch_size=1,coord_atom_lvl_to_be_noised=base.clone(),initializer_outputs=init,ref_initializer_outputs=None,f_ref=None)
    record={'features':features_json(f),'coordinates':tensor_json(base),'schedule':tensor_json(sampler._construct_inference_noise_schedule(torch.device('cuda'))),'normal':draws,'uniform':angles,'output':{k:tensor_json(v) for k,v in out.items() if isinstance(v,torch.Tensor)}}
    result[name]=record;print(name,flush=True)
 case=json.loads((ROOT/'fixtures/oracle/monomer.json').read_text());f={k:tensor(v) for k,v in case['features'].items()};init=net.token_initializer(copy.deepcopy(f))
 result['recycles']={str(n):{k:tensor_json(v)for k,v in net.diffusion_module(tensor(case['X_noisy_L']),tensor(case['t']),copy.deepcopy(f),**init,n_recycle=n).items()}for n in [1,3]}
atomic_json(ROOT/'fixtures/sampler-oracle.json',result)
