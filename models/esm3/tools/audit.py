#!/usr/bin/env python3
import hashlib,json,torch
from upstream import ROOT,CACHE,FILES,COMMIT
from safetensors.torch import load_file
report={'upstream_commit':COMMIT,'checkpoints':{},'bundles':{}}
for component,filename in {**FILES,'function_decoder':'esm3_function_decoder_v0.pth'}.items():
    path=CACHE/filename;h=hashlib.sha256()
    with path.open('rb')as f:
        while block:=f.read(16*1024*1024):h.update(block)
    sd=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
    report['checkpoints'][component]={'file':filename,'bytes':path.stat().st_size,'sha256':h.hexdigest(),'state_elements':sum(t.numel()for t in sd.values()),'tensors':len(sd),'dtypes':sorted(set(str(t.dtype)for t in sd.values()))}
for precision in ['fp32','fp16','int8']:
    base=ROOT/'weights'/precision;m=json.loads((base/'manifest.json').read_text());seen=set()
    for shard in m['shards']:
        p=base/shard['file'];raw=p.read_bytes();assert len(raw)==shard['bytes']<=25_000_000;assert hashlib.sha256(raw).hexdigest()==shard['sha256'];del raw
        sd=load_file(p);seen.update(sd)
        for name,rec in m['tensors'].items():
            for part in rec.get('parts',[rec]):
                if part['shard']!=shard['file']:continue
                assert list(sd[part['key']].shape)==part['shape']
                if 'quant'in part:assert part['quant']['scale_tensor'] in sd
    report['bundles'][precision]={k:m[k]for k in ['total_params','num_tensors','total_bytes','weight_rel_rms']}
    report['bundles'][precision].update(shards=len(m['shards']),max_shard_bytes=max(s['bytes']for s in m['shards']))
(ROOT/'fixtures/audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
