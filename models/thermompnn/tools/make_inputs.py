"""Deterministic backbone fixtures derived from the upstream 2OCJ example."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'fixtures'
source=(ROOT/'examples/2OCJ.pdb').read_text().splitlines()
res=[]
for line in source:
    if line.startswith('ATOM') and line[21]=='A' and line[12:16].strip() in ['N','CA','C','O']:
        if not res or res[-1][0][22:27] != line[22:27]: res.append([])
        res[-1].append(line)

def rewrite(lines, chain, number, ins='', mse=False, shift=0):
    result=[]
    for line in lines:
        s=list(line.ljust(80));s[21]=chain;s[22:26]=f'{number:4d}';s[26]=ins or ' '
        if mse:s[0:6]='HETATM';s[17:20]='MSE'
        if shift:s[30:38]=f'{float(line[30:38])+shift:8.3f}'
        result.append(''.join(s))
    return result

def pdb(name, lines):
    (out/name).write_text('\n'.join(lines)+'\nEND\n')

def cif(name,lines,multi=False):
    cols=['group_PDB','id','type_symbol','label_atom_id','label_alt_id','label_comp_id','label_asym_id','label_seq_id','pdbx_PDB_ins_code','Cartn_x','Cartn_y','Cartn_z','occupancy','auth_seq_id','auth_comp_id','auth_asym_id','auth_atom_id','pdbx_PDB_model_num']
    rows=['data_thermompnn','#','loop_']+['_atom_site.'+c for c in cols]
    for i,l in enumerate(lines):
        chain={'A':'Alpha','B':'Beta'}.get(l[21],l[21]) if multi else l[21]
        rows.append(' '.join([l[:6].strip(),str(i+1),l[12:16].strip()[0],l[12:16].strip(),l[16].strip() or '.',l[17:20],chain,l[22:26].strip(),l[26].strip() or '?',l[30:38].strip(),l[38:46].strip(),l[46:54].strip(),'1.0',l[22:26].strip(),l[17:20],chain,l[12:16].strip(),'1']))
    (out/name).write_text('\n'.join(rows)+'\n#\n')
small=[l for r in res[:12] for l in r]
medium=[l for r in res[:76] for l in r]
pdb('full.pdb',[l for r in res for l in r])
complex_lines=[l for i,r in enumerate(res[:8]) for l in rewrite(r,'A',i+1)]+[l for i,r in enumerate(res[12:20]) for l in rewrite(r,'B',i+1)]
stress=[]
for i,r in enumerate(res[:10]):
    # negative IDs, a real gap at 1, insertion code 3A, MSE, incomplete backbone
    num=[-2,-1,0,2,3,3,4,5,6,7][i]
    lines=rewrite(r,'A',num,'A' if i==5 else '',mse=i==2)
    if i==7: lines=[l for l in lines if l[12:16].strip()!='O']
    if i==3:
        alt=lines[1][:16]+'B'+lines[1][17:30]+f'{99.0:8.3f}'+lines[1][38:]
        lines.append(alt) # upstream keeps first occurrence, not greatest occupancy
    stress.extend(lines)
stress.append('HETATM 9998 ZN    ZN A  99      20.000  20.000  20.000  1.00 20.00          ZN  ')
stress.append('HETATM 9999  C1  LIG A 100      20.000  20.000  20.000  1.00 20.00           C  ')
for name,lines in [('small',small),('medium',medium),('complex',complex_lines),('stress',stress)]:
    pdb(name+'.pdb',lines);cif(name+'.cif',lines,multi=name=='complex')
cases=[{'id':'monomer','pdb':'small.pdb','input':'small.pdb'},
 {'id':'monomer_cif','pdb':'small.pdb','input':'small.cif','same_as':'monomer'},
 {'id':'medium','pdb':'medium.pdb','input':'medium.pdb'},
 {'id':'multichain','pdb':'complex.pdb','input':'complex.pdb'},
 {'id':'multichain_cif','pdb':'complex.pdb','input':'complex.cif','same_as':'multichain'},
 {'id':'chain_selection','pdb':'complex.pdb','input':'complex.pdb','chains':['B']},
 {'id':'chain_selection_cif','pdb':'complex.pdb','input':'complex.cif','chains':['B'],'inputChains':['Beta'],'same_as':'chain_selection'},
 {'id':'residue_edge_cases','pdb':'stress.pdb','input':'stress.pdb'},
 {'id':'residue_edge_cases_cif','pdb':'stress.pdb','input':'stress.cif','same_as':'residue_edge_cases'}]
(out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
print('fixtures:',[(c['id'],c['input']) for c in cases])
