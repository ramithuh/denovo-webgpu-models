"""Export chemical input metadata only, never learned outputs or operation traces."""
from environment import ROOT,MOLS
import json
from boltzgen.data.mol import load_canonicals
from boltzgen.data import const
mols=load_canonicals(MOLS)
out={}
for name in ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']:
    mol=mols[name]
    ids=[c.GetId() for c in mol.GetConformers() if c.HasProp('name') and c.GetProp('name')=='Ideal']
    conf=mol.GetConformer(ids[0] if ids else 0)
    atoms={}
    for a in mol.GetAtoms():
        if a.GetAtomicNum()==1: continue
        pos=conf.GetAtomPosition(a.GetIdx())
        atoms[a.GetProp('name')]={'element':a.GetAtomicNum(),'charge':a.GetFormalCharge(),'ref':[pos.x,pos.y,pos.z]}
    out[name]={'id':const.token_ids[name],'atoms':atoms}
(ROOT/'shared/residues.js').write_text('// CCD ideal conformers from pinned BoltzGen inference-data; chemical input metadata.\nexport const residues='+json.dumps(out,separators=(',',':'))+';\n')
