import json
import torch
from checkpoint import ROOT
from rfd3.model.layers.block_utils import create_attention_indices
case=json.loads((ROOT/'fixtures/oracle-pass2/multichain.json').read_text())
f={k:torch.tensor(v['data'],dtype=torch.bool if 'bool' in v['dtype'] else torch.long if 'int' in v['dtype'] else torch.float32,device='cuda').reshape(v['shape']) for k,v in case['features'].items()}
x=torch.tensor(case['X_noisy_L']['data'],device='cuda').reshape(case['X_noisy_L']['shape'])
result={}
for name,coords,tok,near in [('atom',x,f['atom_to_token_map'],2),('token',x[:,f['is_ca']],torch.arange(len(f['asym_id']),device='cuda'),32)]:
 idx=create_attention_indices(f,128,near,X_L=coords,tok_idx=tok)
 result[name]={'coordinates':coords.cpu().flatten().tolist(),'tokens':tok.cpu().tolist(),'indices':idx.cpu().tolist(),'near':near}
(ROOT/'fixtures/neighbour-oracle.json').write_text(json.dumps(result))
