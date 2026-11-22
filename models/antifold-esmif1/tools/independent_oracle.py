"""Replay upstream AntiFold sampling without altering its source or model graph."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from antifold.antiscripts import amino_list, get_imgt_mask, sample_new_sequences_CDR_HL, sample_new_sequences_CDR_H

def dump_independent(case,logits,alphabet,writer):
    if case['variant']!='antifold' or case.get('numbering')!='imgt':return None
    valid=[i for i,r in enumerate(case['residues']) if r and r['chain'] in [case.get('heavyChain'),case.get('lightChain')]]
    selected_logits=logits[valid][:,[alphabet.get_idx(a) for a in amino_list]]
    df=pd.DataFrame(selected_logits,columns=amino_list)
    df['pdb_res']=[case['sequence'][i] for i in valid]
    df['pdb_pos']=[case['residues'][i]['number'] for i in valid]
    df['pdb_posins']=[str(case['residues'][i]['number'])+case['residues'][i]['insertion'] for i in valid]
    df['pdb_chain']=[case['residues'][i]['chain'] for i in valid]
    df['assumed_region']=['CDRH3' if case['residues'][i]['chain']==case['heavyChain'] else 'CDRL3' for i in valid]
    df['top_res']=[amino_list[i] for i in selected_logits.argmax(-1)]
    regions=case.get('regions',['CDR3']);mask=get_imgt_mask(df,regions)
    rng=np.random.default_rng(456);uniforms=rng.uniform(.01,.99,len(case['sequence'])).astype(np.float32)
    order=rng.permutation(len(case['sequence'])).tolist();draws=iter(uniforms[np.array(valid)[mask]])
    def multinomial(probs,num_samples):
        return torch.tensor([[min(int(torch.searchsorted(p.cumsum(-1),float(next(draws)))),len(p)-1)] for p in probs])
    def choice(indices,size,replace=False):
        return np.array(sorted(indices,key=lambda j:order.index(valid[int(j)]))[:size])
    temperature=.7
    with patch('torch.multinomial',multinomial),patch('numpy.random.choice',choice):
        if case.get('lightChain'):
            h,l=sample_new_sequences_CDR_HL(df,t=temperature,imgt_regions=regions,limit_expected_variation=True)
            sample=np.concatenate([h,l])
        else:sample=sample_new_sequences_CDR_H(df,t=temperature,imgt_regions=regions,limit_expected_variation=True)
    tokens=np.array([alphabet.get_idx(c) for c in case['sequence']],dtype=np.int32)
    tokens[valid]=[alphabet.get_idx(c) for c in sample]
    fullmask=np.zeros(len(tokens),dtype=np.int32);fullmask[valid]=mask
    writer.put('independent.mask',fullmask);writer.put('independent.uniforms',uniforms);writer.put('independent.tokens',tokens)
    return dict(temperature=temperature,regions=regions,limitExpectedVariation=True,backmutationOrder=order)
