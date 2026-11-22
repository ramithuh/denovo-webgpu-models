#!/usr/bin/env python3
"""Sharded, row-splittable safetensors; packed F16/I8, exact round-trip checks."""
import argparse,json,hashlib,torch,pathlib
from safetensors.torch import save,load_file
from download_weights import ROOT,digest,ASSETS
COMMIT='9d2c260813da3c9a2bc944973953f63ea6d71203'
LIMIT=25_000_000

def state_dict():
 for name,(_,sha) in ASSETS.items():
  assert digest(ROOT/'weights'/name)==sha, f'Checkpoint integrity failure: {name}'
 c=torch.load(ROOT/'weights/ff2_reft.pth',map_location='cpu',weights_only=False)
 s={k.removeprefix('vectorfield_network.'):v.float() for k,v in c['state_dict'].items()}
 e=torch.load(ROOT/'weights/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False)['model']
 for k,v in e.items():
  k=k.removeprefix('encoder.sentence_encoder.').removeprefix('encoder.')
  # Frozen sequence inference consumes representations/attention, not contact/LM logits.
  if k.startswith(('lm_head.','contact_head.')):continue
  s['seq_encoder.esm.'+k]=v.float()
 return s

def encode(t,precision):
 t=t.detach().cpu().float().contiguous()
 if precision=='fp32' or t.ndim!=2:return t,None,'F32'
 if precision=='fp16':return t.half(),None,'F16'
 scale=t.abs().amax(1).div(127).clamp_min(1e-30)
 return (t/scale[:,None]).round().clamp(-127,127).to(torch.int8),scale,'I8'

def reload_bundle(directory):
 m=json.loads((directory/'manifest.json').read_text());out={};cache={}
 for name,e in m['tensors'].items():
  parts=[]
  for p in e['parts']:
   f=p['shard']
   if f not in cache:
    cache={f:load_file(directory/f)}
   v=cache[f][p['key']].float()
   if e['dtype']=='I8':v=v*cache[f][p['scale_tensor']][:,None]
   parts.append(v)
  out[name]=torch.cat(parts,dim=0) if len(parts)>1 else parts[0]
 return out

def export(state,precision):
 dst=ROOT/'weights'/precision;dst.mkdir(exist_ok=True)
 entries={};shards=[];pending={};refs=[];cost=0
 def flush():
  nonlocal pending,refs,cost
  if not pending:return
  blob=save(pending);assert len(blob)<=LIMIT
  file=f'model-{len(shards)+1:05d}.safetensors';(dst/file).write_bytes(blob)
  shards.append(dict(file=file,byteLength=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
  for ref in refs:ref['shard']=file
  pending={};refs=[];cost=0
 for name,t in state.items():
  v,scale,dtype=encode(t,precision);parts=[]
  rows=v.shape[0] if v.ndim else 1
  rowbytes=v.numel()*v.element_size()//rows+(4 if scale is not None else 0)
  maxrows=max(1,(LIMIT-4096)//rowbytes)
  for start in range(0,rows,maxrows):
   end=min(start+maxrows,rows);part=v[start:end].contiguous() if v.ndim else v
   key=name if rows<=maxrows else name+f'.__part{start}'
   group={key:part};ref=dict(key=key,row_start=start,shape=list(part.shape))
   if scale is not None:
    ref['scale_tensor']=key+'.__scale';group[key+'.__scale']=scale[start:end].contiguous()
   added=sum(x.numel()*x.element_size()+len(k)+256 for k,x in group.items())
   if cost+added>LIMIT:flush()
   pending.update(group);refs.append(ref);parts.append(ref);cost+=added
  entries[name]=dict(dtype=dtype,shape=list(v.shape),parts=parts)
 flush()
 m=dict(format='foldflow2-safetensors-v1',precision=precision,upstream_commit=COMMIT,checkpoints={n:digest(ROOT/'weights'/n) for n in ASSETS},tensors=entries,shards=shards,total_bytes=sum(x['byteLength'] for x in shards),config={'esm_layers':33,'esm_dim':1280,'esm_heads':20,'coordinate_scaling':0.1,'bb_blocks':2,'trunk_blocks':2})
 (dst/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 # Validate every physical shard and decoded tensor, one tensor at a time.
 cache={};numel=0;error=0.
 for sh in shards:assert digest(dst/sh['file'])==sh['sha256']
 for name,t in state.items():
  v,sc,_=encode(t,precision);expected=v.float() if sc is None else v.float()*sc[:,None]
  for p in entries[name]['parts']:
   f=p['shard']
   if f not in cache:cache={f:load_file(dst/f)}
   got=cache[f][p['key']].float()
   if sc is not None:got=got*cache[f][p['scale_tensor']][:,None]
   target=expected[p['row_start']:p['row_start']+got.shape[0]] if got.ndim else expected
   assert torch.equal(got,target),name
  error+=float((expected-t).double().square().sum());numel+=t.numel()
 metric=dict(precision=precision,tensors=len(state),parameters=numel,bytes=m['total_bytes'],shards=len(shards),roundtrip_exact=True,weight_rmse=(error/numel)**.5)
 print(metric,flush=True);return metric
if __name__=='__main__':
 torch.set_num_threads(8)
 s=state_dict()
 (ROOT/'fixtures/architecture.json').write_text(json.dumps({k:list(v.shape) for k,v in s.items()},indent=2)+'\n')
 metrics=[export(s,p) for p in ('fp32','fp16','int8')]
 (ROOT/'fixtures/export_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
 # Tiny bundle exercises the same browser Cache/OPFS loader without recaching gigabytes.
 dst=ROOT/'.cache/fixtures/loader';dst.mkdir(parents=True,exist_ok=True)
 blob=save({'probe':torch.arange(8).float()});(dst/'probe.safetensors').write_bytes(blob)
 meta=dict(format='foldflow2-safetensors-v1',precision='fp32',total_bytes=len(blob),shards=[dict(file='probe.safetensors',byteLength=len(blob),sha256=hashlib.sha256(blob).hexdigest())],tensors={'probe':dict(dtype='F32',shape=[8],parts=[dict(shard='probe.safetensors',key='probe',shape=[8],row_start=0)])})
 (dst/'manifest.json').write_text(json.dumps(meta)+'\n')
