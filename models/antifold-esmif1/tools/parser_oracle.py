"""Independent PDB/mmCIF parsing expectations from upstream Biotite preprocessing."""
import json
import numpy as np
import biotite.structure as bs
from make_cases import case as make_case
from reference import ROOT
from antifold.esm_util_custom import load_structure
from antifold.esm_multichain_util_custom import extract_coords_from_complex
from dump_oracle import Writer

cases=json.loads((ROOT/'fixtures/cases.json').read_text());case=next(c for c in cases if c['name']=='monomer')
atoms=[]
for i,(res,coords) in enumerate(zip(case['residues'],case['coords'])):
    name=['MSE','SEC','UNK','GLY','ALA','PRO'][i];chain='A' if i<3 else 'B';number=[-2,100,100,1,3,4][i];ins='A' if i==2 else ''
    for j,(atom,xyz) in enumerate(zip(['N','CA','C'],coords)):
        if i==4 and atom=='C':continue
        if i==0 and atom=='CA':
            atoms.append(('ATOM',atom,'B',name,chain,number,ins,xyz,.2))
            atoms.append(('ATOM',atom,'A',name,chain,number,ins,[x+5 for x in xyz],.8))
        else:atoms.append(('ATOM',atom,'',name,chain,number,ins,xyz,1.))
atoms.extend([('HETATM','O','','HOH','A',500,'',[1.,2.,3.],1.),('HETATM','ZN','','ZN','A',501,'',[2.,3.,4.],1.),('HETATM','C1','','LIG','A',502,'',[3.,4.,5.],1.),('ATOM','P','','DA','D',1,'',[4.,5.,6.],1.),('ATOM','H','','PRO','B',4,'',[4.,5.,6.],1.)])
def pdb_line(n,a):
    group,atom,alt,name,chain,num,ins,xyz,occ=a
    return f'{group:<6}{n:5d} {atom:>4}{alt:1}{name:>3} {chain:1}{num:4d}{ins:1}   {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occ:6.2f}{20.:6.2f}          {atom[0]:>2}  '
pdb='MODEL        1\n'+'\n'.join(pdb_line(i+1,a) for i,a in enumerate(atoms))+'\nENDMDL\nMODEL        2\n'+pdb_line(1,atoms[0])+'\nENDMDL\n'
(ROOT/'fixtures/parser.pdb').write_text(pdb)
cols=['group_PDB','id','type_symbol','label_atom_id','label_alt_id','label_comp_id','label_asym_id','label_entity_id','label_seq_id','pdbx_PDB_ins_code','Cartn_x','Cartn_y','Cartn_z','occupancy','B_iso_or_equiv','auth_seq_id','auth_comp_id','auth_asym_id','auth_atom_id','pdbx_PDB_model_num']
rows=[]
for i,a in enumerate(atoms):
    group,atom,alt,name,chain,num,ins,xyz,occ=a;chain={'A':'Heavy','B':'Light'}.get(chain,chain)
    row=[group,i+1,atom[0],atom,alt or '.',name,chain,1,num,ins or '?',*map(lambda v:f'{v:.3f}',xyz),occ,20,num,name,chain,atom,1]
    rows.append(' '.join(map(str,row)))
cif='data_parser\n#\nloop_\n'+'\n'.join('_atom_site.'+c for c in cols)+'\n'+'\n'.join(rows)+'\n#\n'
(ROOT/'fixtures/parser.cif').write_text(cif)
records={};w=Writer(ROOT/'fixtures/parser_oracle')
for ext in ['pdb','cif']:
    s=load_structure(str(ROOT/f'fixtures/parser.{ext}'))
    cd,sq,ps,ids=extract_coords_from_complex(s)
    records[ext]={};items=[]
    for chain in cd:
        records[ext][chain]=dict(sequence=sq[chain],positions=ps[chain].tolist(),ids=ids[chain].tolist())
        w.put(ext+'.'+chain,cd[chain])
        subset=s[s.chain_id==chain];starts=bs.get_residue_starts(subset)
        residues=[dict(chain=chain,number=int(subset.res_id[i]),insertion=str(subset.ins_code[i]),name=str(subset.res_name[i])) for i in starts]
        items.append((cd[chain],sq[chain],residues))
    c=make_case('parser_'+ext,'esmif1',items,sourceFile='fixtures/parser.'+ext)
    c['coords']=[[[float(v) if np.isfinite(v) else None for v in xyz] for xyz in r] for r in c['coords']]
    cases=[old for old in cases if old['name']!=c['name']]+[c]
w.finish(dict(cases=records))
print(records)

(ROOT/'fixtures/cases.json').write_text(json.dumps(cases,indent=2)+'\n')
