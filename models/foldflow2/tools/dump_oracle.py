#!/usr/bin/env python3
"""Eager pinned model forward and upstream SE(3) reverse integration, exact inputs saved."""
import argparse,json,hashlib,time
from reference import ROOT,load_model,torch,np,Rigid
from export_foldflow2 import reload_bundle
from foldflow.data.all_atom import compute_backbone

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--precision',default='fp32');ap.add_argument('--lengths',nargs='+',type=int,default=[32,64]);args=ap.parse_args()
 model=load_model();state=reload_bundle(ROOT/'weights'/args.precision)
 result=model.load_state_dict(state,strict=False)
 assert all(k.startswith(('seq_encoder.esm.lm_head.','seq_encoder.esm.contact_head.')) for k in result.missing_keys),result
 del state
 def canonical(f):
  f=f.clone();f[...,:4]*=torch.where(f[...,:1]<0,-1,1);return f
 for L in args.lengths:
  gen=torch.Generator().manual_seed(771+L);q=torch.randn(1,L,4,generator=gen);q/=q.norm(dim=-1,keepdim=True);tr=torch.randn(1,L,3,generator=gen)*10;tr-=tr.mean(1,keepdim=True)
  initial=torch.cat([q,tr],-1);records={};meta={};folder=ROOT/'.cache/fixtures'/args.precision/f'L{L}';folder.mkdir(parents=True,exist_ok=True)
  def save(name,t):
   a=t.detach().cpu().float().contiguous().numpy().astype('<f4');blob=a.tobytes();file=name+'.bin';(folder/file).write_bytes(blob);meta[name]={'shape':list(a.shape),'file':str((folder/file).relative_to(ROOT)),'sha256':hashlib.sha256(blob).hexdigest()}
  batch=dict(rigids_t=initial.clone(),res_mask=torch.ones(1,L),fixed_mask=torch.zeros(1,L),seq_idx=torch.arange(1,L+1)[None],chain_idx=torch.zeros(1,L,dtype=torch.long),aatype=torch.zeros(1,L,dtype=torch.long),sc_ca_t=torch.zeros(1,L,3),t=torch.tensor([.6]))
  save('initial',initial)
  def hook(name):
   def cb(mod,inp,out):
    if name=='seq_encoder':save('esm.single',out[0]);save('esm.pair',out[1])
    elif name=='sequence_to_trunk_network':save('sequence.single',out[0]);save('sequence.pair',out[1])
    elif name=='bb_encoder':save('encoder.single',out['single_emb']);save('encoder.pair',out['pair_emb']);save('encoder.frames',canonical(out['rigids'].to_tensor_7()))
    elif name=='trunk_network':save('trunk.single',out[0]);save('trunk.pair',out[1])
   return cb
  handles=[getattr(model,n).register_forward_hook(hook(n)) for n in ['seq_encoder','sequence_to_trunk_network','bb_encoder','trunk_network']]
  with torch.no_grad():out=model(batch)
  for h in handles:h.remove()
  for n in ['rigids','psi','rot_vectorfield','trans_vectorfield','atom37','atom14']:
   save('forward.'+n,canonical(out[n]) if n=='rigids' else out[n])
  for T in [2,3]:
   batch['rigids_t']=initial.clone();batch['sc_ca_t']=torch.zeros(1,L,3);batch['t']=torch.tensor([1.])
   with torch.no_grad():out=model(batch)
   batch['sc_ca_t']=out['rigids'][...,4:]
   trajectory=[];dt=.99/(T-1)
   for ti,t in enumerate(np.linspace(.01,1,T)[::-1]):
    batch['t']=torch.tensor([t],dtype=torch.float32)
    with torch.no_grad():out=model(batch)
    batch['sc_ca_t']=out['rigids'][...,4:]
    _,_,r=model.flow_matcher.reverse(rigid_t=Rigid.from_tensor_7(batch['rigids_t']),rot_vectorfield=out['rot_vectorfield'].numpy(),trans_vectorfield=out['trans_vectorfield'].numpy(),flow_mask=np.ones((1,L)),t=float(t),dt=dt,center=True,noise_scale=0.)
    batch['rigids_t']=r.to_tensor_7();trajectory.append(canonical(batch['rigids_t']))
    save(f'T{T}.step{ti}.pred',canonical(out['rigids']))
   save(f'T{T}.trajectory',torch.stack(trajectory));save(f'T{T}.atoms',compute_backbone(r,out['psi'])[0]);save(f'T{T}.psi',out['psi']);save(f'T{T}.rotVF',out['rot_vectorfield']);save(f'T{T}.transVF',out['trans_vectorfield'])
  if args.precision=='fp32' and L==32:
   batch.update(rigids_t=initial.clone(),sc_ca_t=torch.zeros(1,L,3),t=torch.tensor([.6]),aatype=torch.arange(L)[None]%20)
   model.conditional_generation()
   def seq_hook(mod,inp,out):
    save('conditioned.single',out[0]);save('conditioned.pair',out[1])
   h=model.seq_encoder.register_forward_hook(seq_hook)
   with torch.no_grad():out=model(batch)
   h.remove()
   save('conditioned.tokens',torch.cat([torch.tensor([[0]]),model.seq_encoder.af2_to_esm[batch['aatype']+1],torch.tensor([[2]])],1))
   for name in ['rigids','psi','atom37','atom14']:
    save('conditioned.'+name,canonical(out[name]) if name=='rigids' else out[name])
   model.eval()
  (ROOT/'fixtures'/f'oracle_{args.precision}_L{L}.json').write_text(json.dumps({'precision':args.precision,'length':L,'forward_time':.6,'step_counts':[2,3],'min_t':.01,'self_conditioning':True,'reference':'Pinned upstream eager FP32 arithmetic, identically dequantized weights (including ESM2)','tensors':meta},indent=2)+'\n')
  print(args.precision,L,'done',flush=True)
if __name__=='__main__':main()
