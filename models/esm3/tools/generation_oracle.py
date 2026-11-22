#!/usr/bin/env python3
"""PyTorch replay of the public confidence-ranked cosine mask schedule, temperature zero."""
import json, math, hashlib
import torch
from upstream import ROOT, load_models, restore
from export_esm3 import load_bundle
from dump_oracle import inputs, Writer
@torch.inference_mode()
def run(models,precision):
    m,e,d=[models[k] for k in ['esm','encoder','decoder']];w=Writer('generation-'+precision);cases={}
    for name,source,track in [('fill','masked','sequence'),('inverse_fill','inverse','sequence'),('fold','sequence','structure')]:
        raw=inputs()[source];kw={k:v.clone() for k,v in raw.items()};case_input={k:v.tolist()[0] for k,v in raw.items()}
        if 'structure_coords' in kw:_,kw['structure_tokens']=e.encode(kw['structure_coords'])
        key=track+'_tokens';mask=32 if track=='sequence' else 4096
        if key not in kw:
            kw[key]=torch.full_like(kw['sequence_tokens'],4096);kw[key][:,0]=4098;kw[key][:,-1]=4097
        pending=[i for i,v in enumerate(kw[key][0].tolist()) if v==mask];initial=len(pending);history=[]
        for step in range(3):
            if not pending:break
            out=m(**kw);logits=getattr(out,track+'_logits')[0];lo,hi=(4,24) if track=='sequence' else (0,4096)
            w.tensor(f'{name}/{step}/logits',logits)
            candidates=[]
            for i in pending:
                row=logits[i,lo:hi].double();p=row.softmax(-1);j=int(row.argmax());candidates.append((float(p[j]),i,j+lo))
            candidates.sort(key=lambda c:(-c[0],c[1]));remaining=math.floor(initial*math.cos((step+1)/3*math.pi/2));count=max(1,len(pending)-remaining)
            chosen=candidates[:count]
            for _,i,t in chosen:kw[key][0,i]=t;pending.remove(i)
            history.append(dict(step=step,tokens=kw[key][0].tolist(),filled=[i for _,i,_ in chosen]))
        if track=='structure':
            out=d.decode(kw[key]);w.tensor(name+'/coordinates',out['bb_pred'])
        cases[name]=dict(input=case_input,track=track,steps=3,history=history)
        print(precision,name,'generated',flush=True)
    w.close()
    def clean(v):
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    (ROOT/'fixtures'/f'generation-{precision}.json').write_text(json.dumps(clean(dict(precision=precision,temperature=0,schedule='confidence-ranked cosine',cases=cases,tensors=w.records,binary='../weights/'+w.file.name,sha256=hashlib.sha256(w.file.read_bytes()).hexdigest())),indent=2)+'\n')
if __name__=='__main__':
    models=load_models()
    for precision in ['fp32','fp16','int8']:
        if precision!='fp32':restore(models,load_bundle(ROOT/'weights'/precision))
        run(models,precision)
