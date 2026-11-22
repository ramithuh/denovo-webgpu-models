#!/usr/bin/env python3
"""Export all checkpoint tensors without silently dropping training heads."""
import argparse, hashlib, json, pathlib, torch
from safetensors.torch import save, load_file
ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMIT = '86507b6538f51fce57b5a72477165f03999ed7ae'
LIMIT = 25_000_000

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def quantizable(name, t):
    return t.ndim == 2 and not any(s in name for s in ('emb_q.', 'emb_left.', 'emb_right.', 'emb_state.', '.pos.emb.', '.se3.'))

def encode(name, t, precision):
    t = t.detach().cpu().float().contiguous()
    if precision == 'fp32' or not quantizable(name, t): return {name: t}, 'F32'
    if precision == 'fp16': return {name: t.half()}, 'F16'
    scale = t.abs().amax(dim=1).div(127).clamp_min(1e-30)
    return {name: (t / scale[:, None]).round().clamp(-127,127).to(torch.int8), name+'.__scale': scale}, 'I8'

def reload_bundle(directory):
    meta = json.loads((directory/'manifest.json').read_text()); shards = {}; result = {}
    for s in meta['shards']:
        path = directory/s['file']
        assert path.stat().st_size == s['byteLength'] and digest(path) == s['sha256']
        shards[s['file']] = load_file(path)
    for name, entry in meta['tensors'].items():
        data = shards[entry['shard']]; value = data[name].float()
        if entry['dtype'] == 'I8': value = value * data[name+'.__scale'][:,None]
        result[name] = value
    return result

def export(checkpoint, precision):
    dst = ROOT/'weights/base'/precision; dst.mkdir(parents=True, exist_ok=True)
    state = checkpoint['model_state_dict']; shards=[]; entries={}; pending={}; names=[]; size=0
    def flush():
        nonlocal pending,names,size
        if not pending: return
        blob=save(pending); assert len(blob)<=LIMIT
        file=f'model-{len(shards)+1:05d}.safetensors'; (dst/file).write_bytes(blob)
        shards.append(dict(file=file,byteLength=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
        for n in names: entries[n]['shard']=file
        pending={}; names=[]; size=0
    for name,t in state.items():
        group,dtype=encode(name,t,precision)
        cost=sum(x.numel()*x.element_size() for x in group.values())+sum(len(n)+256 for n in group)
        if size+cost>LIMIT: flush()
        assert cost<LIMIT, f'Tensor must be row-split: {name}'
        pending.update(group); names.append(name); size+=cost
        entries[name]=dict(key=name,dtype=dtype,shape=list(t.shape),row_start=0)
        if dtype=='I8': entries[name]['scale_tensor']=name+'.__scale'
    flush()
    manifest=dict(format='rfdiffusion-safetensors-v1',precision=precision,upstream_commit=COMMIT,
                  checkpoint_sha256=digest(ROOT/'weights/Base_ckpt.pt'),config=checkpoint['config_dict'],
                  tensors=entries,shards=shards,total_bytes=sum(s['byteLength'] for s in shards))
    (dst/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    loaded=reload_bundle(dst)
    expected={n:(g[n].float()*g[n+'.__scale'][:,None] if dt=='I8' else g[n].float()) for n,t in state.items() for g,dt in [encode(n,t,precision)]}
    assert all(torch.equal(loaded[n],expected[n]) for n in loaded)
    print(precision,manifest['total_bytes'],len(shards),'shards',len(loaded),'tensors',flush=True)
    return dict(precision=precision,total_bytes=manifest['total_bytes'],shards=len(shards),tensor_reload_exact=True)

def main():
    from download_weights import verify
    path=ROOT/'weights/Base_ckpt.pt'; verify(path)
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    state=ckpt['model_state_dict']; groups={}
    for n,t in state.items(): groups[n.split('.')[0]]=groups.get(n.split('.')[0],0)+t.numel()
    inventory=dict(upstream_commit=COMMIT,checkpoint_sha256=digest(path),config=ckpt['config_dict'],
                   parameter_count=sum(t.numel() for t in state.values()),groups=groups,
                   tensors={n:dict(shape=list(t.shape),dtype=str(t.dtype),numel=t.numel()) for n,t in state.items()})
    (ROOT/'fixtures/architecture.json').write_text(json.dumps(inventory,indent=2)+'\n')
    metrics=[export(ckpt,p) for p in ('fp32','fp16','int8')]
    (ROOT/'fixtures/export_storage.json').write_text(json.dumps(metrics,indent=2)+'\n')
if __name__=='__main__': main()
