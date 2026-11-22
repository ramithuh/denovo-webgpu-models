#!/usr/bin/env python3
"""CPU FP32 traces of unmodified upstream modules, plus bundle round-trip metrics."""
import json, math
import torch
from upstream import ROOT, SEQUENCES, load_upstream, trace, inference_state
from export_dyna1 import load_bundle

def stats(a,b):
    a=a.double().flatten(); b=b.double().flatten()
    return dict(rel_rms=float(((a-b).square().sum()/b.square().sum().clamp_min(1e-30)).sqrt()),
                max_abs=float((a-b).abs().max()),correlation=float(torch.corrcoef(torch.stack([a,b]))[0,1]))

if __name__=='__main__':
    b,h=load_upstream()
    traces={k:trace(b,h,s) for k,s in SEQUENCES.items()}
    dest=ROOT/'fixtures'; dest.mkdir(exist_ok=True)
    meta=dict(format='dyna1-oracle-v1',sequences=SEQUENCES,tensors={},torch_version=torch.__version__,device='cpu',dtype='float32')
    offset=0
    with (dest/'oracle_dyna1.bin').open('wb') as f:
        for seq,ts in traces.items():
            for k,t in ts.items():
                raw=t.numpy().astype('<i4' if k=='tokens' else '<f4').tobytes()
                meta['tensors'][f'{seq}.{k}']=dict(dtype='I32' if k=='tokens' else 'F32',shape=list(t.shape),byteOffset=offset,byteLength=len(raw))
                f.write(raw); offset+=len(raw)
    (dest/'oracle_dyna1.json').write_text(json.dumps(meta,indent=2)+'\n')
    metrics={}
    # Replace constant input tracks with their exported, precomputed lookup table.
    class FoldedEncoder(torch.nn.Module):
        def __init__(self,w): super().__init__();self.register_buffer('weight',w)
        def forward(self,t,*args): return torch.nn.functional.embedding(t,self.weight)
    for prec in ['fp32','fp16','int8']:
        sd=load_bundle(ROOT/'weights'/prec)
        b.encoder=FoldedEncoder(sd.pop('embedding.weight'))
        for i,block in enumerate(b.transformer.blocks[:23]):
            prefix=f'backbone.{i}.'
            update={k.removeprefix(prefix):v for k,v in sd.items() if k.startswith(prefix)}
            missing=block.load_state_dict(update,strict=False).missing_keys
            assert all(k.startswith('geom_attn.') for k in missing)
        h.load_state_dict({k.removeprefix('head.'):v for k,v in sd.items() if k.startswith('head.')},strict=True)
        metrics[prec]={}
        for name,s in SEQUENCES.items():
            out=trace(b,h,s)
            metrics[prec][name]={k:stats(out[k],traces[name][k]) for k in ['embedding','final_embeddings','logits','probs']}
        print(prec,json.dumps(metrics[prec]),flush=True)
        for values in metrics[prec].values():
            for key in ['logits','probs']:
                assert values[key]['rel_rms'] < ({'fp32':1e-4,'fp16':5e-4,'int8':1.0}[prec])
                assert values[key]['correlation'] > .99
    (dest/'precision_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
