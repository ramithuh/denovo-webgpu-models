"""Load original pinned modules, without training/evaluation datasets or weights.

The checkpoint includes C/A/T embeddings but no label-to-index vocabulary.
Only vocabulary construction is replaced, with explicitly synthetic labels for
index-based parity. Every exercised neural module and flow method is upstream.
"""
import pathlib, sys, types, json, subprocess
import torch
from typing import Dict
# Compatibility with the upstream torch 2.x typing alias, removed in 2.9.
torch.Dict = Dict
from omegaconf import OmegaConf
from download_weights import ROOT, NAME, verify
COMMIT = 'a44b407daf6a5358e43cd68907f3e3f1cbc65fdc'
UPSTREAM = ROOT / '.cache/upstream'
def checkpoint():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip() == COMMIT
    path = ROOT / 'weights' / NAME
    verify(path)
    c = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    state = {k.removeprefix('nn.'): v for k,v in c['state_dict'].items()}
    assert len(state)==490 and sum(v.numel() for v in state.values())==191102976
    cfg = OmegaConf.to_container(c['hyper_parameters']['cfg_exp'].model.nn,resolve=False)
    assert not cfg['update_pair_repr'] and not cfg['use_tri_mult']
    return state, cfg

def upstream_model(state, cfg):
    sys.path.insert(0,str(UPSTREAM))
    # torch_scatter is imported unconditionally, but only needed by multilabel
    # averaging. This equivalent CPU implementation avoids a compiled dependency.
    scatter = types.ModuleType('torch_scatter')
    def scatter_mean(src,index,dim=0,dim_size=None):
        assert dim==0
        n=dim_size or int(index.max())+1
        out=src.new_zeros((n,*src.shape[1:])).index_add_(0,index,src)
        count=src.new_zeros(n).index_add_(0,index,src.new_ones(len(index)))
        return out/count.clamp_min(1).reshape(n,*([1]*(src.ndim-1)))
    scatter.scatter_mean=scatter_mean
    sys.modules.setdefault('torch_scatter',scatter)
    from proteinfoundation.nn.feature_factory import FoldEmbeddingSeqFeat
    def mapping(self, unused):
        for level in ['C','A','T']:
            count=state[f'cond_factory.feat_creators.1.embedding_{level}.weight'].shape[0]-1
            depth={'C':1,'A':2,'T':3}[level]
            setattr(self,'class_mapping_'+level,{'.'.join([str(i)]*depth):i for i in range(count)})
            setattr(self,'num_classes_'+level,count)
    FoldEmbeddingSeqFeat.create_mapping=mapping
    from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
    model=ProteinTransformerAF3(**cfg).eval()
    model.load_state_dict(state,strict=True)
    return model

def quantize(name,t,precision):
    compress=t.ndim==2 and name.endswith('.weight') and 'embedding_' not in name
    if not compress or precision=='fp32': return t.float().contiguous(),None
    if precision=='fp16': return t.half().contiguous(),None
    scale=t.abs().amax(dim=1).clamp_min(1e-12)/127
    q=torch.round(t/scale[:,None]).clamp(-127,127).to(torch.int8)
    return q.contiguous(),scale.contiguous()
def dequant(q,s): return q.float() if s is None else q.float()*s[:,None]
def constants():
    import math
    return dict(timeFreq=torch.exp(torch.arange(128,dtype=torch.float32)*(-math.log(2000)/127)).tolist(),
      indexDivisor=(2056**(2*torch.arange(64)/128)).tolist(),
      scBins=torch.linspace(.1,3,127).tolist(),xtBins=torch.linspace(.1,3,63).tolist())
