#!/usr/bin/env python3
"""Independent geometry and Biopython checks of browser-exported structures."""
import io,json
import numpy as np
from checkpoint import ROOT
from Bio.PDB import PDBParser,MMCIFParser
import biotite.structure as struc
from biotite.structure.io import pdbx

records=json.loads((ROOT/'fixtures/gpu-generation.json').read_text())['results'];output=[]
for record in records:
    cif=record['mmcif'];pdb=record['pdb']
    bp=PDBParser(QUIET=True).get_structure('p',io.StringIO(pdb));bc=MMCIFParser(QUIET=True).get_structure('c',io.StringIO(cif))
    p_atoms=list(bp.get_atoms());c_atoms=list(bc.get_atoms())
    assert len(p_atoms)==len(c_atoms)
    assert [a.element for a in p_atoms]==[a.element for a in c_atoms]
    aa=pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(cif)),model=1)
    protein=aa[struc.filter_amino_acids(aa)]
    ca=protein[protein.atom_name=='CA'];dist=[];chain_breaks=0;peptide=[]
    for i in range(1,len(ca)):
        if ca.chain_id[i]==ca.chain_id[i-1] and ca.res_id[i]==ca.res_id[i-1]+1:
            d=float(np.linalg.norm(ca.coord[i]-ca.coord[i-1]));dist.append(d);chain_breaks+=int(d>4.5)
    for i in range(1,len(ca)):
        if ca.chain_id[i]==ca.chain_id[i-1] and ca.res_id[i]==ca.res_id[i-1]+1:
            c=protein[(protein.chain_id==ca.chain_id[i-1])&(protein.res_id==ca.res_id[i-1])&(protein.atom_name=='C')]
            n=protein[(protein.chain_id==ca.chain_id[i])&(protein.res_id==ca.res_id[i])&(protein.atom_name=='N')]
            if len(c)==len(n)==1:peptide.append(float(np.linalg.norm(c.coord[0]-n.coord[0])))
    bonds=struc.connect_via_residue_names(aa).as_array();neighbors=[set() for _ in aa]
    for a,b,_ in bonds:neighbors[a].add(int(b));neighbors[b].add(int(a))
    closure=None
    if record['case']=='cyclic':
        first=np.where((aa.res_id==ca.res_id[0])&(aa.atom_name=='N'))[0][0];last=np.where((aa.res_id==ca.res_id[-1])&(aa.atom_name=='C'))[0][0]
        neighbors[first].add(int(last));neighbors[last].add(int(first))
        closure={'CN':float(np.linalg.norm(aa.coord[first]-aa.coord[last])),'CA':float(np.linalg.norm(ca.coord[0]-ca.coord[-1]))}
        chain_breaks+=int(closure['CA']>4.5)
    # Exclude covalent 1-2, 1-3 and 1-4 interactions; report two explicit thresholds.
    excluded=[]
    for i in range(len(aa)):
        seen={i};front={i}
        for _ in range(3):front={b for a in front for b in neighbors[a]}-seen;seen|=front
        excluded.append(seen)
    severe=[];overlap=[];radii={'C':1.70,'N':1.55,'O':1.52,'S':1.80,'P':1.80,'ZN':1.39}
    for i in range(len(aa)):
        for j in range(i+1,len(aa)):
            if j in excluded[i]:continue
            d=float(np.linalg.norm(aa.coord[i]-aa.coord[j]))
            if d<1.5:severe.append([i,j,d])
            # Deliberately separate severe overlaps from a stricter van der Waals check.
            threshold=radii.get(aa.element[i].upper(),1.7)+radii.get(aa.element[j].upper(),1.7)-0.4
            if d<threshold:overlap.append([i,j,d])
    try:sse=struc.annotate_sse(protein);secondary={label:int(np.count_nonzero(sse==value)) for label,value in [('helix','a'),('strand','b'),('coil','c')]}
    except (ValueError,IndexError):secondary=None
    out={k:v for k,v in record.items() if k not in ['pdb','mmcif']}
    out.update(outputAtoms=len(aa),proteinResidues=len(ca),biopythonAtoms=len(c_atoms),chainBreaks=chain_breaks,
      caMean=float(np.mean(dist)) if dist else None,caMin=min(dist) if dist else None,caMax=max(dist) if dist else None,
      peptideCNMean=float(np.mean(peptide)) if peptide else None,peptideCNMax=max(peptide) if peptide else None,
      cyclicClosure=closure,severeClashes=len(severe),vdwOverlapPairs=len(overlap),secondaryStructurePSEA=secondary,
      severeClashPairs=severe,vdwOverlapPairsFirst20=overlap[:20])
    output.append(out)
(ROOT/'fixtures/generation-geometry.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k not in ['severeClashPairs','vdwOverlapPairsFirst20']} for r in output]))
