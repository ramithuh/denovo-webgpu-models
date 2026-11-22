"""Row-split sharded safetensors with atomic, colocated int8 scale companions."""
import hashlib, json
import torch
from safetensors.torch import save
from oracle_common import ROOT, COMMIT, model, quantize, reload_bundle, stats
LIMIT = 25_000_000

def export(state, cfg, precision):
    folder=ROOT/'weights'/precision; folder.mkdir(parents=True,exist_ok=True)
    expected, packed=quantize(state,precision)
    manifest={'format':'frameflow-safetensors-v1','precision':precision,'upstream':COMMIT,'config':cfg,
      'checkpoint':{'file':'pdb/published.ckpt','sha256':hashlib.sha256((ROOT/'weights/published.ckpt').read_bytes()).hexdigest()},'shards':[],'tensors':{}}
    pending={}; refs=[]
    def flush():
        if not pending:return
        blob=save(pending); assert len(blob)<=LIMIT
        file=f'model-{len(manifest["shards"])+1:03}.safetensors'; (folder/file).write_bytes(blob)
        manifest['shards'].append({'file':file,'byteLength':len(blob),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
        for p in refs:p['shard']=file
        pending.clear();refs.clear()
    for name,(t,scale) in sorted(packed.items()):
        dtype={torch.float32:'F32',torch.float16:'F16',torch.int8:'I8'}[t.dtype]
        rec={'shape':list(t.shape),'dtype':dtype,'parts':[]};manifest['tensors'][name]=rec
        rowbytes=t[0].numel()*t.element_size()+(4 if scale is not None else 0)
        rows=max(1,(LIMIT-4096)//rowbytes)
        for start in range(0,t.shape[0],rows):
            part=t[start:start+rows].contiguous();key=name if rows>=t.shape[0] else f'{name}.__part{start}'
            p={'key':key,'shape':list(part.shape),'dtype':dtype,'row_start':start}
            group={key:part}
            if scale is not None:
                group[key+'__scale']=scale[start:start+rows].contiguous()
                p['quant']={'scheme':'symmetric_per_row','axis':0,'scale_tensor':key+'__scale'}
            if pending and len(save({**pending,**group}))>LIMIT:flush()
            pending.update(group);refs.append(p);rec['parts'].append(p)
    flush()
    manifest.update(total_bytes=sum(s['bytes'] for s in manifest['shards']), total_params=sum(t.numel() for t in state.values()),num_tensors=len(state))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    restored=reload_bundle(precision)
    assert set(restored)==set(expected)
    for n in expected:assert torch.equal(expected[n],restored[n]),n
    return {k:manifest[k] for k in ['total_bytes','total_params','num_tensors','shards']}|{'reload_exact':True,'max_weight_drift':max(stats(restored[n],state[n])['maxAbs'] for n in state)}
if __name__=='__main__':
    net,cfg=model();results={p:export(net.state_dict(),cfg,p) for p in ['fp32','fp16','int8']}
    (ROOT/'fixtures/export_metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({p:{k:v for k,v in r.items() if k!='shards'} for p,r in results.items()},indent=2))
