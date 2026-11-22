"""Input fixtures from upstream structures, plus explicit edge-case derivatives."""
import json
import numpy as np
import biotite.structure as bs
from biotite.structure.io import pdb,pdbx
from reference import ROOT
from antifold.esm_util_custom import load_structure
from antifold.esm_multichain_util_custom import extract_coords_from_structure

def chain_case(path, chain, start=None, stop=None):
    structure=load_structure(str(ROOT/path),chain)
    coords,seq,pos,ids=extract_coords_from_structure(structure)
    starts=bs.get_residue_starts(structure)
    residues=[dict(chain=chain,number=int(structure.res_id[i]),insertion=str(structure.ins_code[i]),name=str(structure.res_name[i])) for i in starts]
    sl=slice(start,stop)
    return coords[sl],seq[sl],residues[sl]

def case(name,variant,items,**extra):
    coords=[];seq=[];res=[]
    for n,(c,s,r) in enumerate(items):
        if n:
            coords.extend(np.full((10,3,3),np.nan));seq.append('-'*10);res.extend([None]*10)
        coords.extend(c);seq.append(s);res.extend(dict(x) for x in r)
    out=dict(name=name,variant=variant,coords=np.array(coords).tolist(),sequence=''.join(seq),residues=res,**extra)
    return out

if __name__=='__main__':
    h=chain_case('data/pdbs/6y1l_imgt.pdb','H')
    l=chain_case('data/pdbs/6y1l_imgt.pdb','L')
    v=chain_case('data/nanobody/8oi2_imgt.pdb','B')
    m=chain_case('fixtures/1l2y.pdb','A')
    def region(x,a,b):
        indices=[i for i,r in enumerate(x[2]) if a<=r['number']<=b]
        return x[0][indices],''.join(x[1][i] for i in indices),[dict(x[2][i]) for i in indices]
    hs=region(h,105,111);ls=region(l,105,110);vs=region(v,105,111)
    v=(v[0],v[1],[dict(r,chain='H') for r in v[2]])
    vs=(vs[0],vs[1],[dict(r,chain='H') for r in vs[2]])
    tiny=(m[0][:6],m[1][:6],m[2][:6])
    other=(m[0][7:12]+np.array([1,2,3],dtype=np.float32),m[1][7:12],[dict(r,chain='BC') for r in m[2][7:12]])
    hs[2][-1].update(number=hs[2][-2]['number'],insertion='A')
    cases=[case('antibody_hl','antifold',[hs,ls],regions=['CDRH3','L:105-108'],heavyChain='H',lightChain='L',numbering='imgt'),
           case('nanobody','antifold',[vs],regions=['CDRH3'],heavyChain='H',numbering='imgt'),
           case('monomer','esmif1',[tiny]),
           case('score_single','esmif1',[(tiny[0],'A'+tiny[1][1:],tiny[2])],nativeSequence=tiny[1]),
           case('score_double','esmif1',[(tiny[0],'AG'+tiny[1][2:],tiny[2])],nativeSequence=tiny[1]),
           case('complex','esmif1',[tiny,other],targetChain='A'),
           case('antibody_antigen','antifold',[vs,other],heavyChain='H',regions=['CDRH3'],numbering='imgt'),
           case('small_full','esmif1',[m],benchmark=True),
           case('antibody_hl_full','antifold',[h,l],benchmark=True,heavyChain='H',lightChain='L',regions=['CDRH3','CDRL3'],numbering='imgt'),
           case('vhh_full','antifold',[v],benchmark=True,heavyChain='H',regions=['CDRH3'],numbering='imgt')]
    # Derivative fixture: missing atoms, negative IDs, insertion codes, confidence.
    edge=case('missing_insertion','esmif1',[tiny,other],targetChain='A')
    edge['coords'][2][2]=[float('nan')]*3
    edge['residues'][0]['number']=-2
    edge['residues'][1].update(number=100,insertion='A')
    edge['confidence']=[0.7]*len(edge['coords'])
    cases.append(edge)
    # A numbering-only derivative exercises explicitly supplied Kabat annotation.
    kabat=case('antibody_kabat','antifold',[hs,ls],heavyChain='H',lightChain='L',regions=['CDRH3','CDRL3'],numbering='kabat')
    for r in kabat['residues']:
        if r: r['number']-=10 if r['chain']=='H' else 16
    cases.append(kabat)
    # Binary input coordinates preserve NaNs, JSON records null for missing atoms.
    for c in cases:
        c['coords']=[[[float(v) if np.isfinite(v) else None for v in xyz] for xyz in r] for r in c['coords']]
    (ROOT/'fixtures/cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    print([(c['name'],len(c['sequence'])) for c in cases])
