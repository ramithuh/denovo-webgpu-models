"""Adversarial exact PyTorch neighbor tie-order fixtures."""
import json
from reference import ROOT
import torch
cases=[]
for n in [1,3,16,31,48,49,76,128,2048]:
    for kind in ['equal','ties','increasing','descending']:
        d=(torch.zeros(n) if kind=='equal' else torch.arange(n).float()%7 if kind=='ties' else torch.arange(n).float() if kind=='increasing' else torch.arange(n-1,-1,-1).float())
        k=min(n,48)
        cases.append({'name':f'{n}-{kind}','values':d.tolist(),'indices':torch.topk(d,k,largest=False).indices.tolist()})
(ROOT/'fixtures/topk_cases.json').write_text(json.dumps(cases,separators=(',',':'))+'\n')
