"""Fixed-frame motif path from pinned Interpolant.sample, using the one PDB checkpoint."""
from dump_oracle import inputs,write,coords
from oracle_common import ROOT,model,reload_bundle,torch
from data.interpolant import Interpolant
from omegaconf import OmegaConf
import json
cases=[]
for precision in ['fp32','fp16','int8']:
 net,_=model(reload_bundle(precision))
 with torch.inference_mode():
  for L in [32,64]:
   x=inputs(L);mask=torch.ones(1,L);mask[:,8:12]=0;x['diffuse_mask']=mask
   # Residue-index offsets affect the node embedding, while edge relative indices remain contiguous.
   x['res_idx'][:,16:]+=7
   out=net(x);out['coords']=coords(out)
   cases.append(write(precision+f'_motif_L{L}',x,out,{'precision':precision,'length':L,'kind':'forward','mode':'fixed-frame-motif'})['name'])
   cfg=OmegaConf.create({'min_t':.01,'rots':{'corrupt':True,'sample_schedule':'exp','exp_rate':10},'trans':{'corrupt':True},'sampling':{'num_timesteps':3},'self_condition':True,'twisting':{'use':False}})
   sampler=Interpolant(cfg);sampler.set_device('cpu');preds=[]
   def wrap(batch):
    p=net(batch);preds.append({k:v.clone() for k,v in p.items()});return p
   at,_,_=sampler.sample(1,L,wrap,trans_0=x['trans_t'],rotmats_0=x['rotmats_t'],trans_1=x['trans_t'],rotmats_1=x['rotmats_t'],diffuse_mask=mask,res_idx=x['res_idx'])
   out={**preds[-1],'coords':at[-1][...,[0,1,2,4,3],:]}
   for i,p in enumerate(preds):out.update({f'step{i}.{k}':v for k,v in p.items()})
   cases.append(write(precision+f'_motif_L{L}_T3',x,out,{'precision':precision,'length':L,'kind':'sample','steps':3,'mode':'fixed-frame-motif'})['name'])
   print(precision,L,flush=True)
(ROOT/'fixtures/motif_matrix.json').write_text(json.dumps({'cases':cases},indent=2)+'\n')
