#!/usr/bin/env python3
"""Export the primary ESM-3 Dyna-1 sequence predictor to capped safetensors shards."""
import argparse, hashlib, json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from upstream import ROOT, CONFIG, load_upstream, inference_state

FORMAT = 'dyna1-safetensors-v1'

def export(sd, precision, outdir, cap=25_000_000):
    dest = outdir / precision
    dest.mkdir(parents=True, exist_ok=True)
    tensors, shards, pending, records = {}, [], {}, []
    size = 0
    err = ref = 0.
    def flush():
        nonlocal pending, records, size
        if not pending: return
        filename = f'shard-{len(shards):03d}.safetensors'
        p = dest / filename
        save_file(pending, str(p), metadata={'format':FORMAT,'precision':precision})
        raw = p.read_bytes()
        assert len(raw) <= cap, (p,len(raw))
        shards.append(dict(file=filename,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),num_tensors=len(records),num_raw_tensors=len(pending)))
        for r in records: r['shard'] = filename
        pending, records, size = {}, [], 0
    for name,w in sd.items():
        w=w.float()
        if w.ndim == 2 and precision == 'int8':
            scale=w.abs().amax(1).clamp_min(1e-10)/127
            q=(w/scale[:,None]).round().clamp(-127,127).to(torch.int8)
            restored=q.float()*scale[:,None]
            dtype='I8'
        else:
            scale=None
            # Preserve the first two classifier blocks: all-half misses the 5e-4 output target.
            sensitive = name.startswith(('head.encoder.layers.0.', 'head.encoder.layers.1.'))
            q=w.half() if w.ndim==2 and precision=='fp16' and not sensitive else w
            restored=q.float()
            dtype='F16' if q.dtype==torch.float16 else 'F32'
        err+=float(((restored-w).double()**2).sum())
        ref+=float((w.double()**2).sum())
        row_bytes=q[0].numel()*q.element_size()+(4 if scale is not None else 0)
        rows=max(1,(cap-65536)//row_bytes)
        parts=[]
        for start in range(0,q.shape[0],rows):
            end=min(start+rows,q.shape[0])
            split=q.shape[0]>rows
            key=f'{name}.__part{len(parts):03d}' if split else name
            data={key:q[start:end].contiguous()}
            rec=dict(key=key,dtype=dtype,shape=list(q[start:end].shape),row_start=start)
            if scale is not None:
                sk=key+'.__scale'
                data[sk]=scale[start:end].contiguous()
                rec['quant']=dict(scheme='symmetric_per_row',axis=0,scale_tensor=sk,scale_dtype='F32',scale_shape=[end-start])
            nbytes=sum(t.numel()*t.element_size() for t in data.values())+len(key)*4+1024
            if size+nbytes>cap-8192: flush()
            pending.update(data); records.append(rec); size+=nbytes; parts.append(rec)
        tensors[name]=dict(dtype=dtype,shape=list(w.shape),parts=parts) if len(parts)>1 else parts[0]
    flush()
    manifest=dict(format=FORMAT,model='dyna1-esm3-sequence',precision=precision,
        source={'repo':'gelnesr/Dyna-1','checkpoint':'dyna1.pt','backbone':'EvolutionaryScale/esm3-sm-open-v1',
                'upstream_commit':'814a453fd85305536b919901e41062b91d5c9d86'},
        total_params=sum(w.numel() for w in sd.values()),num_tensors=len(sd),
        total_bytes=sum(s['bytes'] for s in shards),config=CONFIG,shards=shards,tensors=tensors,
        weight_rel_rms=(err/ref)**.5,
        fp32_exceptions=['head.encoder.layers.0.*','head.encoder.layers.1.*'] if precision=='fp16' else [])
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(precision,manifest['total_params'],manifest['total_bytes'],len(shards),manifest['weight_rel_rms'],flush=True)
    return manifest

def load_bundle(path):
    m=json.loads((path/'manifest.json').read_text())
    cache={}; sd={}
    for name,rec in m['tensors'].items():
        values=[]
        for part in rec.get('parts',[rec]):
            file=part['shard']
            if file not in cache: cache[file]=load_file(str(path/file))
            raw=cache[file]; w=raw[part['key']].float()
            if 'quant' in part: w=w*raw[part['quant']['scale_tensor']][:,None]
            values.append(w)
        sd[name]=torch.cat(values,dim=0)
    return sd

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,default=ROOT/'weights');p.add_argument('--precisions',nargs='+',default=['fp32','fp16','int8']);args=p.parse_args()
    b,h=load_upstream();sd=inference_state(b,h)
    for precision in args.precisions: export(sd,precision,args.outdir)
