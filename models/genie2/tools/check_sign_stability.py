"""Measure upstream signed-quaternion sensitivity without modifying any upstream code."""
import json,torch
from oracle_common import ROOT,model,features,frame,stats
from genie.utils.affine_utils import rot_to_quat
m=model();m.single_feature_net.n_timestep=2
f=features(24,'unconditional');g=torch.Generator().manual_seed(17024);x=torch.randn(1,24,3,generator=g);ts=frame(x,f)
rows=[]
with torch.inference_mode():
 q=m.pair_feature_net._encode_orientations(ts.rots,f['residue_mask']);z=m(ts,torch.tensor([2]),f)['z']
 for eps in [1e-8,1e-7,1e-6]:
  xx=x+torch.randn(x.shape,generator=g)*eps;tt=frame(xx,f);qq=m.pair_feature_net._encode_orientations(tt.rots,f['residue_mask']);zz=m(tt,torch.tensor([2]),f)['z']
  rows.append({'coordinate_perturbation':eps,'quaternion_sign_flips':int(((q*qq).sum(-1)<0).sum()),'pairs':576,'input':stats(xx,x),'predicted_noise':stats(zz,z)})
result={'method':'Unmodified pinned PyTorch model, epoch 40; perturb initial coordinates only','rows':rows}
(ROOT/'fixtures/sign_stability.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
