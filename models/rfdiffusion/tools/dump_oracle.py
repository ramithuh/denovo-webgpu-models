#!/usr/bin/env python3
from reference import *
import argparse,json,types,hashlib,time
import numpy as np
from rfdiffusion.inference.model_runners import Sampler
from rfdiffusion.kinematics import xyz_to_t2d
from rfdiffusion.chemical import INIT_CRDS
from rfdiffusion.util_module import ComputeAllAtomCoords
from export_rfdiffusion import reload_bundle

class Binary:
 def __init__(self,path):self.f=path.open('wb');self.offset=0
 def add(self,t):
  t=torch.as_tensor(t).detach().cpu().float().contiguous();data=t.numpy().tobytes();entry=dict(offset=self.offset,length=t.numel(),shape=list(t.shape));self.f.write(data);self.offset+=len(data);return entry

def inputs(length,mode,t=50,selfcond=False,coords=None,previous=None):
 if coords is None:torch.manual_seed(12345+length);np.random.seed(12345+length)
 L=length;seq=torch.nn.functional.one_hot(torch.full((L,),21),22).float();mask=torch.zeros(L,dtype=torch.bool)
 if mode=='motif': mask[3:7]=True;seq[mask]=torch.nn.functional.one_hot(torch.zeros(mask.sum(),dtype=torch.long),22).float()
 if coords is None:
  x=INIT_CRDS[:14].clone()[None].repeat(L,1,1);x=x+torch.randn(L,1,3)*8
 else:x=coords.clone()
 if mode=='symmetric' and coords is None:
  from rfdiffusion.inference.symmetry import SymGen
  x,seq=SymGen('c3',False,10).apply_symmetry(x,seq)
 sampler=types.SimpleNamespace();sampler._conf=types.SimpleNamespace(diffuser=types.SimpleNamespace(T=50),preprocess=types.SimpleNamespace(sidechain_input=False,d_t1d=22),ppi=types.SimpleNamespace(hotspot_res=None))
 sampler.T=50;sampler.binderlen=L;sampler.ppi_conf=sampler._conf.ppi;sampler.preprocess_conf=sampler._conf.preprocess;sampler.mask_str=mask[None];sampler.contig_map=types.SimpleNamespace(rf=list(range(L)));sampler.device=torch.device('cpu')
 args=Sampler._preprocess(sampler,seq,x.clone(),t)
 msa,full,seq,xt,idx,t1d,t2d,xyz_t,alpha=args
 if mode=='symmetric':
  from rfdiffusion.inference.symmetry import SymGen
  idx,_=SymGen('c3',False,10).res_idx_procesing(res_idx=idx)
 if previous is not None:
  xyz_t=torch.cat([previous[None,None,:,:3],torch.zeros(1,1,L,24,3)],dim=3);t2d=xyz_to_t2d(xyz_t)
 elif selfcond:
  xyz_t=torch.cat([x[None,None,:,:3]+torch.randn(1,1,L,1,3)*.1,torch.zeros(1,1,L,24,3)],dim=3);t2d=xyz_to_t2d(xyz_t)
 else:xyz_t=torch.zeros_like(xyz_t);t2d=torch.zeros_like(t2d)
 return dict(msa_latent=msa,msa_full=full,seq=seq,xyz=xt[:,:,:3],idx=idx,t=torch.tensor(t),t1d=t1d,t2d=t2d,xyz_t=xyz_t[:,:,:,:3],alpha_t=alpha,motif_mask=mask,return_infer=True)

def capture(m,data):
 traces={};handles=[];calls={}
 def put(n,t):traces[n]=t.detach().clone()
 def emb(_,a,v):
  for n,x in zip(['msa','pair','state'],v):put('latent_emb.'+n,x)
 handles.append(m.latent_emb.register_forward_hook(emb));handles.append(m.full_emb.register_forward_hook(lambda _,a,v:put('full_emb',v)))
 def rec(_,a,v):
  for n,x in zip(['msa','pair','state'],v):put('recycle.'+n,traces['latent_emb.'+n]+x)
 handles.append(m.recycle.register_forward_hook(rec))
 for j,b in enumerate(m.templ_emb.templ_stack.block):handles.append(b.register_forward_hook(lambda _,a,v,j=j:put(f'templ_emb.templ_stack.block.{j}',v)))
 handles.append(m.templ_emb.register_forward_hook(lambda _,a,v:[put('templ_emb.'+n,x) for n,x in zip(['pair','state'],v)] and None))
 for name,module in m.named_modules():
  if name.startswith(('simulator.extra_block.','simulator.main_block.')) and name.count('.')==2:
   j=int(name.split('.')[-1]);handles.append(module.register_forward_hook(lambda _,a,v,name=name,j=j:[put(f'{name}.{j}.{n}',x) for n,x in zip(['msa','pair','R','T','state','alpha'],v)] and None))
 def refine(_,a,v):
  j=calls.get('ref',0);calls['ref']=j+1
  for n,x in zip(['R','T','state','alpha'],v):put(f'simulator.str_refiner.{j}.{n}',x)
  put(f'simulator.str_refiner.{j}.msa',a[0]);put(f'simulator.str_refiner.{j}.pair',a[1])
 handles.append(m.simulator.str_refiner.register_forward_hook(refine))
 try:
  with torch.no_grad():v=m(**data)
 finally:
  for h in handles:h.remove()
 for n,x in zip(['px0','alpha','logits','plddt'],[v[2],v[4],v[5],v[6]]):put(n,x)
 with torch.no_grad(): _,allatom=ComputeAllAtomCoords()(data['seq'].argmax(-1),v[2],v[4])
 put('backbone',allatom[:,:,:4])
 if data['idx'][0,-1]-data['idx'][0,0]>len(data['idx'][0])*2:
  from rfdiffusion.inference.symmetry import SymGen
  sym,_=SymGen('c3',False,10).apply_symmetry(allatom[0,:,:4],data['seq'][0]);put('symmetric',sym)
 return traces

def main():
 p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=24);p.add_argument('--precision',default='fp32');p.add_argument('--mode',default='unconditional');p.add_argument('--selfcond',action='store_true');a=p.parse_args()
 state=reload_bundle(ROOT/'weights/base'/a.precision);m=model(state);data=inputs(a.length,a.mode,selfcond=a.selfcond)
 name=f'{a.precision}_{a.mode}_L{a.length}'+('_selfcond' if a.selfcond else '');path=ROOT/'.cache/fixtures'/f'{name}.bin';path.parent.mkdir(parents=True,exist_ok=True);b=Binary(path)
 meta=dict(name=name,length=a.length,precision=a.precision,mode=a.mode,selfcond=a.selfcond,inputs={k:b.add(v) for k,v in data.items() if isinstance(v,torch.Tensor)},binary=f'../.cache/fixtures/{name}.bin')
 started=time.perf_counter();traces=capture(m,data);meta['oracleMs']=(time.perf_counter()-started)*1000
 meta['traces']={k:b.add(v) for k,v in traces.items()};b.f.close();meta['binary_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();meta['binary_byteLength']=b.offset
 (ROOT/'fixtures'/f'oracle_{name}.json').write_text(json.dumps(meta,indent=2)+'\n');print(name,meta['oracleMs'],'ms',len(traces),'traces',flush=True)
if __name__=='__main__':main()
