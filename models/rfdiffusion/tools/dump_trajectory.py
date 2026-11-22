from reference import *
import json,numpy as np,argparse,hashlib,logging,types
from dump_oracle import inputs,Binary,capture
from export_rfdiffusion import reload_bundle
from rfdiffusion.diffusion import IGSO3,get_beta_schedule
from rfdiffusion.inference.utils import get_next_ca,get_next_frames,Denoise
from rfdiffusion.inference.symmetry import SymGen
from rfdiffusion.util_module import ComputeAllAtomCoords
from rfdiffusion.chemical import INIT_CRDS

p=argparse.ArgumentParser();p.add_argument('--precision',default='fp32');p.add_argument('--mode',default='unconditional');p.add_argument('--length',type=int,default=24);a=p.parse_args();L=a.length
m=model(reload_bundle(ROOT/'weights/base'/a.precision));initial=inputs(L,a.mode);mask=initial['motif_mask'];seq=initial['seq'].argmax(-1)[0]
x=INIT_CRDS[:14].clone()[None].repeat(L,1,1);x[:,:3]=initial['xyz'][0]
# Initial motif sidechains must share the same sampled residue translation.
x[:,3:]+=initial['xyz'][0,:,1,None,:]-INIT_CRDS[1]
if a.mode=='symmetric':x,_=SymGen('c3',False,10).apply_symmetry(x,seq)
name=f'trajectory_{a.precision}_{a.mode}_L{L}';path=ROOT/'.cache/fixtures'/f'{name}.bin';b=Binary(path);meta=dict(name=name,length=L,mode=a.mode,precision=a.precision,noiseSeed=444,noisePolicy='independent draws from one seeded stream; input preparation does not reseed',binary=f'../.cache/fixtures/{name}.bin',initial=b.add(x),sequence=b.add(seq),mask=b.add(mask),steps=[])
s=IGSO3(T=50,min_sigma=.02,max_sigma=1.5,min_b=1.5,max_b=2.5,cache_dir=str(ROOT/'.cache/igso3'));diffuser=types.SimpleNamespace(so3_diffuser=s);beta,_,abar=get_beta_schedule(50,.01,.07,'linear');prev=None;np.random.seed(444);torch.manual_seed(444)
for step,t in enumerate([50,49,48]):
 data=inputs(L,a.mode,t=t,coords=x,previous=prev)
 with torch.no_grad():v=m(**data);_,allatom=ComputeAllAtomCoords()(data['seq'].argmax(-1),v[2],v[4]);px0=allatom[0,:,:14]
 prev=v[2][0].clone()
 if a.mode=='symmetric':
  sym,_=SymGen('c3',False,10).apply_symmetry(px0,seq);prev=sym[:,:3]
  # SelfConditioning.sample_step replaces px0 before its second allatom call.
  with torch.no_grad(): _,allatom=ComputeAllAtomCoords()(data['seq'].argmax(-1),prev[None],v[4]);px0=allatom[0,:,:14]
 if mask.any():
  align=types.SimpleNamespace(_log=logging.getLogger('oracle'));px0=Denoise.align_to_xt_motif(align,px0,x,mask)
 trans=torch.randn(L,3);rot=np.random.randn(int((~mask).sum()),3);allrot=np.zeros((L,3));allrot[~mask]=rot
 # Inject the recorded Gaussian variates into the original upstream update routines.
 oldnormal=torch.normal;oldnp=np.random.normal
 torch.normal=lambda mean,std:mean+std*trans
 np.random.normal=lambda size:rot.copy()
 transitions=[];oldreverse=s.reverse_sample_vectorized
 def reverse(*args,**kwargs):
  v=oldreverse(*args,**kwargs);transitions.append(v);return v
 s.reverse_sample_vectorized=reverse
 try:
  _,delta=get_next_ca(x,px0,t,mask,.25,beta,abar)
  frames=get_next_frames(x,px0,t,diffuser,'igso3',mask)
 finally:torch.normal=oldnormal;np.random.normal=oldnp;s.reverse_sample_vectorized=oldreverse
 xt=torch.from_numpy(frames).float()+delta[:,None,:];nextx=torch.full_like(x,float('nan'));nextx[:,:3]=xt;nextx[mask]=x[mask]
 if a.mode=='symmetric':nextx,_=SymGen('c3',False,10).apply_symmetry(nextx,seq)
 rotation=torch.eye(3)[None].repeat(L,1,1);rotation[~mask]=transitions[0].float()
 meta['steps'].append(dict(t=t,translation=b.add(trans),rotationNoise=b.add(allrot),rotation=b.add(rotation),px0=b.add(px0[:,:4]),xyz=b.add(v[2][0]),coords=b.add(nextx[:,:3]),inputs={k:b.add(v) for k,v in data.items() if isinstance(v,torch.Tensor)}))
 x=nextx
 print(name,t,flush=True)
b.f.close();meta['binary_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();meta['binary_byteLength']=b.offset;(ROOT/'fixtures'/f'{name}.json').write_text(json.dumps(meta,indent=2)+'\n')
