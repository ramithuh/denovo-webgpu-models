"""Reproducible packed, row-major sharded safetensors export. No graph capture."""
import argparse, json, hashlib, math
import torch
from safetensors.torch import save, load_file
from oracle_common import *
MAX_BYTES=25_000_000

def export(state,cfg,precision,max_bytes=MAX_BYTES):
    out=ROOT/'weights'/precision; out.mkdir(exist_ok=True)
    manifest=dict(format='proteina-safetensors-v1',model='proteina_v1.2_DFS_200M_notri',commit=COMMIT,
      checkpoint_sha256=verify(ROOT/'weights'/NAME),precision=precision,config=cfg,constants=constants(),
      num_tensors=len(state),total_params=sum(t.numel() for t in state.values()),shards=[],tensors={})
    pending={}; pending_names=[]; max_error=0.; error2=0.; ref2=0.
    def flush():
        if not pending:return
        blob=save(pending); assert len(blob)<=max_bytes
        filename=f'model-{len(manifest["shards"])+1:05}.safetensors'
        (out/filename).write_bytes(blob)
        manifest['shards'].append(dict(file=filename,byteLength=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
        reloaded=load_file(out/filename)
        for n,t in pending.items(): assert torch.equal(t,reloaded[n]), n
        for name,part in pending_names: part['shard']=filename
        pending.clear(); pending_names.clear()
    for name,t in state.items():
        q,s=quantize(name,t,precision)
        r=dequant(q,s); d=r-t
        max_error=max(max_error,d.abs().max().item()); error2+=d.double().square().sum().item(); ref2+=t.double().square().sum().item()
        row_bytes=q[0].numel()*q.element_size()+(4 if s is not None else 0)
        # Reserve ample room for safetensors JSON metadata. Full rows are kept.
        rows=max(1,(max_bytes-65536)//row_bytes)
        parts=[]
        for start in range(0,q.shape[0],rows):
            stop=min(start+rows,q.shape[0]); key=name if q.shape[0]<=rows else f'{name}.__part{len(parts)}'
            data={key:q[start:stop].contiguous()}
            if s is not None:data[key+'.__scale']=s[start:stop].contiguous()
            if pending and len(save({**pending,**data}))>max_bytes:flush()
            pending.update(data)
            part=dict(key=key,rowStart=start,rowEnd=stop)
            if s is not None:part['scale']=key+'.__scale'
            pending_names.append((name,part)); parts.append(part)
        manifest['tensors'][name]=dict(dtype={torch.float32:'F32',torch.float16:'F16',torch.int8:'I8'}[q.dtype],shape=list(q.shape),parts=parts)
    flush()
    manifest['total_bytes']=sum(s['byteLength'] for s in manifest['shards'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Independently reload all shards and concatenate multipart rows, compare
    # every original tensor with the explicitly quantized reference, strictly.
    reloads={}
    for shard in manifest['shards']: reloads.update(load_file(out/shard['file']))
    loaded_state={}
    for name,t in state.items():
        q,s=quantize(name,t,precision); parts=manifest['tensors'][name]['parts']
        loaded_q=torch.cat([reloads[p['key']] for p in parts])
        assert torch.equal(q,loaded_q)
        loaded_s=torch.cat([reloads[p['scale']] for p in parts]) if s is not None else None
        loaded_state[name]=dequant(loaded_q,loaded_s)
        assert torch.equal(loaded_state[name],dequant(q,s))
        if s is not None: assert torch.equal(s,torch.cat([reloads[p['scale']] for p in parts]))
    reloaded_model=upstream_model(loaded_state,cfg)
    assert all(torch.equal(v,reloaded_model.state_dict()[n]) for n,v in loaded_state.items())
    del reloaded_model,loaded_state,reloads
    return dict(strict_model_reload=True,precision=precision,tensors=len(state),parameters=manifest['total_params'],shards=len(manifest['shards']),bytes=manifest['total_bytes'],
      max_shard_bytes=max(s['byteLength'] for s in manifest['shards']),roundtrip_max_abs=0,weight_drift_relRMS=math.sqrt(error2/ref2),weight_drift_max_abs=max_error)
if __name__=='__main__':
    torch.set_num_threads(8)
    state,cfg=checkpoint()
    result={p:export(state,cfg,p) for p in ['fp32','fp16','int8']}
    (ROOT/'fixtures/export_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
