import json
from unittest.mock import patch
from environment import ROOT
from dump_sampler import make_model
from protpardelle.data.motif import contig_to_motif_placement
from protpardelle.core.models import parse_fixed_pos_str
from protpardelle.data.pdb_io import load_feats_from_pdb
import numpy as np
import torch
np.random.seed(113)
cases=[]
for contig,length in [('1-3;A1-3;2-5',[8,10]),('B2-4;1-2;A1-2;0-3',[8,9]),('3-8',[4,7])]:
    draws=[];original=np.random.randint
    def draw(*a,**kw):
        x=original(*a,**kw);draws.append(int(x));return x
    with patch.object(np.random,'randint',draw):indices,placement,full,total=contig_to_motif_placement(contig,length,1)
    cases.append({'contig':contig,'range':length,'draws':draws,'expected':{'indices':indices[0],'placement':placement[0],'full':full[0],'length':total[0]}})
model=make_model('cc78_epoch1431');layouts=[]
for gap in [0,200]:
    mask,residue,chain=model.make_seq_mask_for_sampling(num_samples=1,prot_lens_per_chain=torch.tensor([[4,3,5]]),chain_residx_gap=gap)
    layouts.append({'lengths':[4,3,5],'gap':gap,'expected':{'seqMask':mask.flatten().tolist(),'residueIndex':residue.flatten().tolist(),'chainIndex':chain.flatten().tolist()}})
feats,_=load_feats_from_pdb(ROOT/'fixtures/parser_edge.pdb',include_pos_feats=True)
selectors=[]
for spec in ['B2-7','A1,A2','B7,B7','B1-100','']:
    expected=parse_fixed_pos_str(spec,feats['chain_id_mapping'],feats['residue_index_orig'],feats['chain_index'])
    selectors.append({'spec':spec,'expected':expected})
(ROOT/'fixtures/input_oracle.json').write_text(json.dumps({'contigs':cases,'layouts':layouts,'selectors':selectors},indent=2)+'\n')
