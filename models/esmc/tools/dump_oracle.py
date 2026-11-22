"""Full upstream traces and strict export round trips, including every mask position."""
import json, hashlib
import numpy as np
import torch
from upstream import ROOT, COMMIT, REVISION, SEQUENCES, AA, load_upstream, trace, metrics
from export_esmc import load_bundle

class Writer:
 def __init__(self,name,meta):
  self.name=name;self.meta=meta;self.entries={};self.f=(ROOT/'fixtures'/f'{name}.bin').open('wb')
 def put(self,key,value):
  a=value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
  a=np.asarray(a,dtype='<i4' if a.dtype.kind in 'ib' else '<f4')
  self.entries[key]=dict(offset=self.f.tell(),shape=list(a.shape),dtype='I32' if a.dtype.kind=='i' else 'F32',length=a.size)
  self.f.write(a.tobytes())
 def close(self):
  self.f.close();path=ROOT/'fixtures'/f'{self.name}.bin'
  self.meta.update(tensors=self.entries,byteLength=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  (ROOT/'fixtures'/f'{self.name}.json').write_text(json.dumps(self.meta,indent=2)+'\n')

@torch.inference_mode()
def run(model,precision):
 aa=torch.tensor([model.tokenizer.convert_tokens_to_ids(c) for c in AA])
 outputs={}
 for name,seq in SEQUENCES.items():
  w=Writer(f'oracle_{precision}_{name}',dict(precision=precision,sequence=seq,upstream_commit=COMMIT,weight_revision=REVISION,alphabet=AA,mask_positions=[1,len(seq)-2]))
  tokens=model._tokenize([seq]);wt=tokens[0,1:-1]
  print(precision,name,'unmasked',flush=True)
  normal=trace(model,seq,attentions=True)
  for k,v in normal.items(): w.put('wt/'+k,v)
  logits=normal['logits'][0,1:-1];llr=logits[:,aa]-logits.gather(1,wt[:,None])
  w.put('wt/llr',llr)
  outputs[name+'/wt_logits']=logits.clone();outputs[name+'/wt_llr']=llr.clone();outputs[name+'/embeddings']=normal['final_ln'].clone();outputs[name+'/pooled']=normal['pooled'].clone()
  masked=tokens.clone();masked[0,torch.tensor([1,len(seq)-2])+1]=32
  filled=trace(model,tokens=masked.flatten().tolist())
  for k,v in filled.items():w.put('fill/'+k,v)
  fill_logits=filled['logits'][0,torch.tensor([1,len(seq)-2])+1][:,aa]
  w.put('fill/probabilities',fill_logits.softmax(-1));w.put('fill/predictions',aa[fill_logits.argmax(-1)].int())
  outputs[name+'/fill_logits']=fill_logits
  rows=[];stages={}
  for i in range(len(seq)):
   masked=tokens.clone();masked[0,i+1]=32
   t=trace(model,tokens=masked.flatten().tolist(),row=i+1)
   for k,v in t.items():stages.setdefault(k,[]).append(v)
   rows.append(t['logits'].reshape(64))
   if (i+1)%10==0:print(precision,name,'mask',i+1,'/',len(seq),flush=True)
  for k,vs in stages.items():w.put('masked/'+k,torch.stack(vs))
  diagonal=torch.stack(rows);logp=diagonal.log_softmax(-1);llr=diagonal[:,aa]-diagonal.gather(1,wt[:,None])
  wt_logp=logp.gather(1,wt[:,None]).flatten();naturalness=wt_logp.mean();pppl=(-naturalness).exp()
  w.put('masked/llr',llr);w.put('masked/wt_logp',wt_logp);w.put('masked/naturalness',naturalness.reshape(1));w.put('masked/pseudo_perplexity',pppl.reshape(1))
  outputs[name+'/masked_logits']=diagonal;outputs[name+'/masked_llr']=llr;outputs[name+'/wt_logp']=wt_logp;outputs[name+'/naturalness']=naturalness;outputs[name+'/pseudo_perplexity']=pppl
  w.close()
 # Token masks / packed sequence IDs use the real pretrained model too.
 w=Writer(f'oracle_{precision}_tokens',dict(precision=precision,upstream_commit=COMMIT))
 tokens=[0,5,32,2,1,0,23,24,2];sid=[0,0,0,0,-1,1,1,1,1]
 for label,ids in [('packed',sid),('padding',None)]:
  for k,v in trace(model,tokens=tokens,sequence_id=ids,attentions=True).items():w.put(label+'/'+k,v)
 w.close()
 return outputs

if __name__=='__main__':
 model=load_upstream();results={};base=None
 for precision in ['fp32','fp16','int8']:
  original=model.state_dict() if precision=='fp32' else None
  state=load_bundle(ROOT/'weights/esmc-300m'/precision)
  if original is not None:
   assert set(state)==set(original)
   assert all(torch.equal(state[k],original[k]) for k in state), 'FP32 round trip is not exact'
  model.load_state_dict(state,strict=True);del state,original
  out=run(model,precision)
  if base is None:base=out
  results[precision]={k:metrics(v,base[k]) for k,v in out.items()}
 (ROOT/'fixtures/export_metrics.json').write_text(json.dumps(results,indent=2)+'\n')
 tokenizer=model.tokenizer
 samples=['NLYIQWLKDGGPSSGRPPPS','ACXBUZO.-|_','A<mask>C','a?']
 (ROOT/'fixtures/tokenizer.json').write_text(json.dumps({s:model._tokenize([s]).flatten().tolist() for s in samples},indent=2)+'\n')
