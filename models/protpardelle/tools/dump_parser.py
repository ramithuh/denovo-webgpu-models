"""PDB/mmCIF edge-case preprocessing fixtures from the original BioPython path."""
import json
from environment import ROOT
from protpardelle.data.pdb_io import load_feats_from_pdb

def line(serial,atom,resname,chain,resid,x,alt='',icode='',occupancy=1,record='ATOM'):
    return f'{record:6}{serial:5d} {atom:>4}{alt:1}{resname:>3} {chain:1}{resid:4d}{icode:1}   {x:8.3f}{2.0:8.3f}{3.0:8.3f}{occupancy:6.2f}{20.0:6.2f}          {atom[0]:>2}  '
lines=[line(1,'CA','ALA','B',-2,1),line(2,'CA','GLY','B',3,2),line(3,'CA','SER','B',3,3,'A','A',.4),line(4,'CA','SER','B',3,4,'B','A',.6),line(5,'CA','UNK','A',100,5),line(6,'H','UNK','A',100,7)]
for i,atom in enumerate(['N','CA','C','O']):lines.append(line(7+i,atom,'MSE','A',101,10+i,record='HETATM'))
lines.append(line(11,'ZN','ZN','A',102,15,record='HETATM'))
lines.extend([
    line(12,'CA','SER','B',3,99,'C','A',.6),
    line(13,'SE','MSE','A',101,18,record='HETATM'),
    line(14,'C1','LIG','A',103,19.125,record='HETATM'),
    line(15,'O1','LIG','A',103,20.125,record='HETATM'),
    line(16,'P','DA','A',104,21.125,record='HETATM'),
    line(17,"C1'",'DA','A',104,22.125,record='HETATM'),
    line(18,'P','U','A',105,23.125,record='HETATM'),
])
(ROOT/'fixtures/parser_edge.pdb').write_text('MODEL        1\n'+'\n'.join(lines)+'\nENDMDL\nMODEL        2\n'+line(1,'CA','ALA','B',-2,999)+'\nENDMDL\nEND\n')
fields=['group_PDB','id','type_symbol','label_atom_id','label_alt_id','label_comp_id','label_asym_id','label_entity_id','label_seq_id','pdbx_PDB_ins_code','Cartn_x','Cartn_y','Cartn_z','occupancy','B_iso_or_equiv','auth_seq_id','auth_asym_id','pdbx_PDB_model_num']
rows=['ATOM 1 C CA . ALA A 1 1 ? 1.000 2.000 3.000 1.00 20.00 -9 AA 1','ATOM 2 C CA A SER A 1 2 A 2.000 3.000 4.000 0.40 20.00 100 AA 1','ATOM 3 C CA B SER A 1 2 A 3.000 4.000 5.000 0.60 20.00 100 AA 1','ATOM 4 C CA . GLY B 2 1 ? 4.000 5.000 6.000 1.00 20.00 200 ZZ 1','HETATM 5 ZN ZN . ZN C 3 . ? 5.000 6.000 7.000 1.00 20.00 1 ZN 1']
rows.extend(['HETATM 6 C C1 . LIG C 3 1 ? 6.125 7.125 8.125 1.00 20.00 1 HH 1', 'ATOM 7 C CA . ALA A 1 1 ? 999.000 2.000 3.000 1.00 20.00 -9 AA 2'])
(ROOT/'fixtures/parser_edge.cif').write_text('data_edge\n#\nloop_\n'+'\n'.join('_atom_site.'+f for f in fields)+'\n'+'\n'.join(rows)+'\n#\n')
records=[]
for name in ['parser_edge.pdb','parser_edge.cif']:
    for gap in [0,200]:
        feats,hetero=load_feats_from_pdb(ROOT/'fixtures'/name,chain_residx_gap=gap,include_pos_feats=True)
        data={k:v.tolist() if hasattr(v,'tolist') else v for k,v in feats.items()}
        context=[{'resname':name,'atoms':atoms,'coords':[pos.tolist() for pos in positions]} for name,atoms,positions in zip(hetero.hetero_aatype,hetero.hetero_atom_types,hetero.hetero_atom_positions)]
        records.append({'file':name,'gap':gap,'features':data,'hetero_count':len(hetero.hetero_aatype),'hetero':context})
(ROOT/'fixtures/parser_oracle.json').write_text(json.dumps(records,indent=2)+'\n')
