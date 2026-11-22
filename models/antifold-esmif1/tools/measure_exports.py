"""Separate quantization effects on scores, perplexities, and sampling from parity."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def tensors(name,precision):
    meta=json.loads((ROOT/'fixtures'/f'oracle_{name}_{precision}.json').read_text());data=(ROOT/'fixtures'/f'oracle_{name}_{precision}.bin').read_bytes()
    return {key:np.frombuffer(data,dtype='<i4' if t['dtype']=='I32' else '<f4',count=t['length'],offset=t['offset']).reshape(t['shape']) for key,t in meta['tensors'].items()}
def relrms(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return float(np.sqrt(np.square(a-b).sum()/max(np.square(b).sum(),1e-30)))
def measure():
    path=ROOT/'fixtures/export_metrics.json';metrics=json.loads(path.read_text()) if path.exists() else {}
    cases=json.loads((ROOT/'fixtures/cases.json').read_text())
    for case in cases:
        name=case['name'];base=tensors(name,'fp32')
        selected=np.array([bool(r) and (not case.get('targetChain') or r['chain']==case['targetChain']) for r in case['residues']])
        for precision in ['fp32','fp16','int8']:
            t=tensors(name,precision);rec=metrics.setdefault(f'{name}/{precision}',{})
            for key in ['logits','logProbabilities','canonicalProbabilities','entropyPerplexity']:rec[key+'_relRMS']=relrms(t[key],base[key])
            ll=float(t['logProbabilities'][selected].mean());baseline=float(base['logProbabilities'][selected].mean())
            rec.update(mean_log_likelihood=ll,mean_log_likelihood_abs_error=abs(ll-baseline),perplexity=float(np.exp(-ll)),perplexity_relative_error=float(abs(np.exp(-ll)-np.exp(-baseline))/np.exp(-baseline)),autoregressive_token_agreement=float(np.mean(t['replay.tokens']==base['replay.tokens'])))
            if 'independent.tokens' in t:rec['independent_token_agreement']=float(np.mean(t['independent.tokens']==base['independent.tokens']))
            coord_mask=np.isfinite(np.array(case['coords'],dtype=np.float32)).all(axis=(-1,-2)) & selected
            coord_ll=float(t['logProbabilities'][coord_mask].mean())
            meta_path=ROOT/'fixtures'/f'oracle_{name}_{precision}.json';meta=json.loads(meta_path.read_text())
            meta['scores']=[ll,float(np.exp(np.float32(-ll))),coord_ll,float(np.exp(np.float32(-coord_ll)))]
            meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    path.write_text(json.dumps(metrics,indent=2)+'\n')
if __name__=='__main__':measure()
