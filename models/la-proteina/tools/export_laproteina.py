"""Deterministic packed safetensors export, <=25 MB including headers.

Only inference tensors: LD1's nn, and AE1's decoder. No optimizer or encoder.
INT8 is symmetric per output row, round-to-nearest-even; scales stay FP32.
"""
import hashlib
import json
import argparse
import torch
from safetensors.torch import save
from oracle_common import ROOT, COMMIT, model_state, make_model, quantize, dequantize, reload_bundle, stats
from download_weights import CHECKPOINTS, digest

LIMIT=25_000_000

def export(model, precision, cfg, state):
    folder=ROOT/'weights'/model.lower()/precision
    folder.mkdir(parents=True,exist_ok=True)
    manifest={'format':'la-proteina-safetensors-v1','model':model,'precision':precision,'upstream_commit':COMMIT,
      'checkpoint':{'name':CHECKPOINTS[model][0],'sha256':CHECKPOINTS[model][1]},'config':cfg,
      'scope':'LD1 generator' if model=='LD1' else 'AE1 decoder (encoder omitted)', 'shards':[],'tensors':{}}
    pending={}; records=[]
    def flush():
        if not pending: return
        blob=save(pending)
        assert len(blob)<=LIMIT
        file=f'model-{len(manifest["shards"])+1:03}.safetensors'
        (folder/file).write_bytes(blob)
        manifest['shards'].append({'file':file,'byteLength':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
        for rec in records: rec['shard']=file
        pending.clear(); records.clear()
    for name,v in sorted(state.items()):
        packed,scale=quantize(v,precision)
        dtype={torch.float32:'F32',torch.float16:'F16',torch.int8:'I8'}[packed.dtype]
        rec={'shape':list(v.shape),'dtype':dtype,'parts':[]}
        manifest['tensors'][name]=rec
        rows=v.shape[0]; width=v.numel()//rows
        per_row=width*packed.element_size()+(4 if scale is not None else 0)
        rows_per=max(1,(LIMIT-16384)//per_row)
        for start in range(0,rows,rows_per):
            end=min(rows,start+rows_per)
            key=name if end==rows and start==0 else f'{name}.__part{start}'
            group={key:packed[start:end].contiguous()}
            part={'key':key,'row_start':start,'shape':list(group[key].shape)}
            if scale is not None:
                skey=key+'.__scale'
                group[skey]=scale[start:end].contiguous()
                part['scale_tensor']=skey
            if pending and len(save({**pending,**group}))>LIMIT: flush()
            assert len(save(group))<=LIMIT
            pending.update(group); records.append(part); rec['parts'].append(part)
    flush()
    manifest['total_bytes']=sum(x['byteLength'] for x in manifest['shards'])
    manifest['total_params']=sum(v.numel() for v in state.values())
    manifest['num_tensors']=len(state)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    restored=reload_bundle(model,precision)
    max_error=0.0
    for name,v in state.items():
        expected=dequantize(*quantize(v,precision))
        assert torch.equal(restored[name],expected), name
        max_error=max(max_error,stats(restored[name],v)['maxAbs'])
    # Strict reload checks every expected module parameter and shape.
    net,_=make_model(model,state=restored)
    return {'total_bytes':manifest['total_bytes'],'total_params':manifest['total_params'],
      'num_tensors':len(state),'shards':len(manifest['shards']),'max_shard_bytes':max(s['byteLength'] for s in manifest['shards']),
      'roundtrip_maxAbs':0.0,'quantization_maxAbs':max_error,'strict_load':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['LD1','AE1']); args=p.parse_args()
    metrics={'upstream_commit':COMMIT,'models':{}}
    for model in args.models:
        assert digest(ROOT/'weights'/CHECKPOINTS[model][0])==CHECKPOINTS[model][1]
        cfg,state=model_state(model)
        metrics['models'][model]={}
        for precision in ['fp32','fp16','int8']:
            result=export(model,precision,cfg,state)
            metrics['models'][model][precision]=result
            print(model,precision,result,flush=True)
    (ROOT/'fixtures/export_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
