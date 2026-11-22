"""Small scalar fixtures for all upstream time/noise schedule families."""
import json,sys,torch
from oracle_common import ROOT,UPSTREAM
sys.path.insert(0,str(UPSTREAM))
from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
fm=R3NFlowMatcher(zero_com=True)
cases=[]
for steps in [3,7,400]:
 for mode in ['uniform','power','cos_sch_v_snr','loglinear','edm','log']:
  power=7. if mode=='edm' else 2.
  ts=fm.get_schedule(mode,steps,p1=power)
  noise=[]
  for gt in ['us','tan','1/t']:
   for p in [1.,1.4]:noise.append(dict(mode=gt,power=p,clamp=10.,values=fm.get_gt(ts[:-1],gt,p,10.).tolist()))
  cases.append(dict(steps=steps,mode=mode,power=power,times=ts.tolist(),noise=noise))
(ROOT/'fixtures/sampler.json').write_text(json.dumps(cases,indent=2)+'\n')

# An actual 400-step original simulation checks the FP32 t=0.99 cutoff.
from unittest.mock import patch
draws=[]
original=torch.randn
def record(*a,**kw):
    y=original(*a,**kw);draws.append(y.flatten().tolist());return y
def predict(batch):
    return batch['x_t'],torch.zeros_like(batch['x_t'])
torch.manual_seed(1234)
with torch.inference_mode(),patch('torch.randn',record):
    out=fm.full_simulation(predict_clean_n_v=predict,dt=1/400,nsamples=1,n=4,mask=torch.ones(1,4,dtype=torch.bool),device='cpu',dtype=torch.float32,self_cond=False,schedule_mode='uniform',schedule_p=2.,sampling_mode='sc',sc_scale_noise=.45,sc_scale_score=1.,gt_mode='1/t',gt_p=1.,gt_clamp_val=10.,cath_code=None)
(ROOT/'fixtures/sampler_400.json').write_text(json.dumps(dict(noise=draws,output=out.flatten().tolist(),draw_count=len(draws)))+'\n')
