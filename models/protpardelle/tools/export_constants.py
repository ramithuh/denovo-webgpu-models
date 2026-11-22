"""Export geometry/token tables directly from the pinned upstream constants."""
import json
import torch
from environment import ROOT
from protpardelle.common import residue_constants as rc

constants = {'restypes': rc.restypes, 'restype3to1': rc.restype_3to1, 'restype1to3': rc.restype_1to3,
             'atomTypes': rc.atom_types, 'atom37Masks': rc.restype_atom37_mask.tolist(),
             'atom73Masks': rc.restype_atom73_mask.tolist(), 'tips': rc.RFDIFFUSION_BENCHMARK_TIP_ATOMS}
constants['noiseFrequencies']={str(d):((1/10000)**(torch.arange(d//2,dtype=torch.float32)/(d//2))).tolist() for d in (128,256)}
constants['map37to73']=[]
constants['map73to37']=[]
for m37,m73 in zip(constants['atom37Masks'],constants['atom73Masks']):
    a=[i for i,v in enumerate(m37) if v];b=[i for i,v in enumerate(m73) if v]
    forward=[-1]*37;backward=[-1]*73
    for x,y in zip(a,b):forward[x]=y;backward[y]=x
    constants['map37to73'].append(forward);constants['map73to37'].append(backward)
(ROOT / 'shared/constants.js').write_text('// Generated from pinned upstream residue_constants.py (Apache-2.0).\nexport const constants = ' + json.dumps(constants, separators=(',', ':')) + ';\n')
