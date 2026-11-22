from reference import *
import json
from se3_transformer.model.basis import get_basis_script,update_basis_with_fused,get_all_clebsch_gordon
sh=list(torch.eye(9).split([1,3,5],dim=1))
b=get_basis_script(1,False,sh,get_all_clebsch_gordon(1,torch.device('cpu')),False)
b=update_basis_with_fused(b,1,False,False)
r={'rbfMu':torch.linspace(0,20,36).tolist(),'lddtBins':torch.linspace(.02,1,50).tolist(),'basis':{k.replace('_fused',''):dict(shape=list(v.shape),data=v.flatten().tolist()) for k,v in b.items() if k.startswith(('in','out'))}}
(ROOT/'shared/constants.json').write_text(json.dumps(r,separators=(',',':'))+'\n')
(ROOT/'shared/constants.js').write_text('export default '+json.dumps(r,separators=(',',':'))+';\n')
print({k:v['shape'] for k,v in r['basis'].items()})
