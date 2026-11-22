"""Lossless reload of each identically quantized state. Shards include header overhead."""
import hashlib,json,torch
from safetensors.torch import save
from oracle_common import ROOT,COMMIT,config,original_state,quantize,reload_bundle,stats
from download_weights import URL,SHA256
LIMIT=25_000_000

def export(state,precision):
    folder=ROOT/'weights'/precision;folder.mkdir(parents=True,exist_ok=True)
    raw,scales,decoded=quantize(state,precision)
    m={'format':'genie2-safetensors-v1','precision':precision,'upstream':{'repo':'https://github.com/aqlaboratory/genie2','commit':COMMIT},'checkpoint':{'url':URL,'sha256':SHA256,'epoch':40,'license':'Apache-2.0'},'config':config(),'shards':[],'tensors':{},'num_tensors':len(state),'total_params':sum(x.numel() for x in state.values())}
    pending={};records=[]
    def flush():
        if not pending:return
        blob=save(pending);assert len(blob)<=LIMIT
        name=f'model-{len(m["shards"])+1:03}.safetensors';(folder/name).write_bytes(blob)
        m['shards'].append({'file':name,'byteLength':len(blob),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
        for p in records:p['shard']=name
        pending.clear();records.clear()
    for k,t in sorted(raw.items()):
        dtype={torch.float32:'F32',torch.float16:'F16',torch.int8:'I8'}[t.dtype]
        r={'shape':list(t.shape),'dtype':dtype,'parts':[]};m['tensors'][k]=r
        row_bytes=t[0].numel()*t.element_size()+(4 if k in scales else 0)
        rows=max(1,(LIMIT-10000)//row_bytes)
        for start in range(0,t.shape[0],rows):
            end=min(start+rows,t.shape[0]);key=k if t.shape[0]<=rows else k+f'.__part_{start}'
            p={'key':key,'shape':[end-start,*t.shape[1:]],'dtype':dtype,'row_start':start}
            group={key:t[start:end].contiguous()}
            if k in scales:
                sk=key+'.__scale';group[sk]=scales[k][start:end].contiguous();p['quant']={'scheme':'symmetric_per_row','axis':0,'scale_tensor':sk}
            if pending and len(save({**pending,**group}))>LIMIT:flush()
            assert len(save(group))<=LIMIT
            pending.update(group);records.append(p);r['parts'].append(p)
    flush();m['total_bytes']=sum(s['byteLength'] for s in m['shards'])
    (folder/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    restored=reload_bundle(precision)
    assert restored.keys()==decoded.keys()
    assert all(torch.equal(restored[k],decoded[k]) for k in decoded),'non-exact roundtrip'
    return {'total_params':m['total_params'],'num_tensors':m['num_tensors'],'total_bytes':m['total_bytes'],'shards':m['shards'],'roundtrip':{'identically_quantized':True,'maxAbs':0},'weight_drift':stats(torch.cat([restored[k].flatten() for k in state]),torch.cat([state[k].flatten() for k in state]))}
if __name__=='__main__':
    state=original_state();out={p:export(state,p) for p in ['fp32','fp16','int8']}
    (ROOT/'fixtures/export_metrics.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({p:{k:v for k,v in x.items() if k!='shards'} for p,x in out.items()},indent=2))
