"""Summarize final masked-fill outputs across the verified precision oracles."""
import json
import numpy as np
from upstream import ROOT, SEQUENCES

def tensor(precision,name,key):
 stem=ROOT/'fixtures'/f'oracle_{precision}_{name}';meta=json.loads(stem.with_suffix('.json').read_text());r=meta['tensors'][key]
 return np.fromfile(stem.with_suffix('.bin'),dtype='<i4' if r['dtype']=='I32' else '<f4',count=r['length'],offset=r['offset']).astype(np.float64)
def main():
 result={}
 for p in ['fp32','fp16','int8']:
  result[p]={}
  for name in SEQUENCES:
   pred=tensor(p,name,'fill/predictions');base=tensor('fp32',name,'fill/predictions')
   prob=tensor(p,name,'fill/probabilities');bp=tensor('fp32',name,'fill/probabilities')
   result[p][name]=dict(fill_top1_agreement=float(np.mean(pred==base)),fill_probability_relRms=float(np.sqrt(np.sum((prob-bp)**2)/np.sum(bp**2))))
 (ROOT/'fixtures/quality_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
