from reference import *
import json,numpy as np
from rfdiffusion import util,chemical
from rfdiffusion.diffusion import IGSO3,get_beta_schedule
# Use the unmodified upstream table construction and schedule.
s=IGSO3(T=50,min_sigma=.02,max_sigma=1.5,min_b=1.5,max_b=2.5,cache_dir=str(ROOT/'.cache/igso3'))
beta,_,abar=get_beta_schedule(50,.01,.07,'linear')
coeff0=[];coeffT=[];std=[]
for t in range(2,51):
 i=t-1;coeff0.append((torch.sqrt(abar[i-1]+1e-6)*beta[i]/(1-abar[i])).item());coeffT.append((torch.sqrt(1-beta[i]+1e-6)*(1-abar[i-1])/(1-abar[i])).item());std.append(torch.sqrt((1-abar[i-1])/(1-abar[i])*beta[i]).item())
r=dict(coef0=[0]+coeff0,coefT=[0]+coeffT,translationStd=[0]+std,beta=beta.tolist(),alphabar=abar.tolist(),omega=s.igso3_vals['discrete_omega'].tolist(),
       score=[s.igso3_vals['score_norm'][s.t_to_idx(t)].tolist() for t in range(1,51)],
       g=[s.g(t/50).item() for t in range(1,51)],
       cdf=s.igso3_vals['cdf'][s.t_to_idx(50)].tolist())
(ROOT/'shared/diffusion-constants.js').write_text('export default '+json.dumps(r,separators=(',',':'),allow_nan=False)+';\n')
r=dict(base=util.xyzs_in_base_frame.tolist(),torsionFrames=util.RTs_by_torsion.tolist(),baseIndices=util.base_indices.tolist(),torsionIndices=util.torsion_indices.tolist(),referenceAngles=util.reference_angles.tolist(),initial=chemical.INIT_CRDS.tolist(),aminoAcids=chemical.num2aa,atoms=chemical.aa2long)
(ROOT/'shared/chemical-constants.js').write_text('export default '+json.dumps(r,separators=(',',':'))+';\n')
print('exported diffusion and chemical constants',flush=True)
