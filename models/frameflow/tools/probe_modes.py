from oracle_common import ROOT,model,torch
from dump_oracle import inputs
from data.interpolant import Interpolant
from omegaconf import OmegaConf
import json,traceback
net,_=model();x=inputs(32);mask=torch.ones(1,32);mask[:,12:16]=0
out={}
for use in [False,True]:
 cfg=OmegaConf.create({'min_t':.01,'rots':{'corrupt':True,'sample_schedule':'exp','exp_rate':10},'trans':{'corrupt':True},'sampling':{'num_timesteps':3},'self_condition':True,'twisting':{'use':use,'t_min':.2,'num_rots':1,'align':True,'scale_rots':0.,'max_offsets':1000,'motif_loc':False,'scale':1.,'scale_w_t':'ot','obs_noise':0.,'potential_trans':True,'potential_rot':True,'update_trans':True,'update_rot':True}})
 it=Interpolant(cfg);it.set_device('cpu')
 try:
  a,b,c=it.sample(1,32,net,trans_0=x['trans_t'],rotmats_0=x['rotmats_t'],trans_1=x['trans_t'],rotmats_1=x['rotmats_t'],diffuse_mask=mask)
  out['guidance' if use else 'clamped']={'success':True}
 except Exception as e:out['guidance' if use else 'clamped']={'success':False,'exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
print(json.dumps(out,indent=2));(ROOT/'fixtures/upstream_mode_probe.json').write_text(json.dumps(out,indent=2)+'\n')
