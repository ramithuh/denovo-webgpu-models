"""Lossless FP32 / packed FP16 / symmetric per-row INT8 safetensors export."""
import hashlib,json,torch,math
from safetensors.torch import save
from oracle_common import ROOT,COMMIT,state,quantize,reload_bundle,stats
from download_weights import SHA256
LIMIT=25_000_000
CONFIG={'tokenDim':768,'condDim':512,'pairDim':512,'layers':15,'heads':12,'registers':10,'target':'v','epsilon':1e-5,'defaultSteps':16,'defaultNoiseScale':.45,'cathCounts':[6,44,1473],'cathNull':[5,43,1472],'selfConditioningInSampler':False,'pairUpdates':False}

def export(s,p):
    folder=ROOT/'weights'/p; folder.mkdir(parents=True,exist_ok=True)
    m={'format':'sid-protein-safetensors-v1','precision':p,'upstream':COMMIT,'checkpoint_sha256':SHA256,'config':CONFIG,'tensors':{},'shards':[],'total_params':sum(t.numel() for t in s.values()),'num_tensors':len(s)}
    pending={}; refs=[]; payload=0
    def flush():
        nonlocal payload
        if not pending: return
        data=save(pending); assert len(data)<=LIMIT
        name=f'model-{len(m["shards"])+1:03}.safetensors'
        (folder/name).write_bytes(data)
        m['shards'].append({'file':name,'byteLength':len(data),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        for r in refs: r['shard']=name
        pending.clear(); refs.clear(); payload=0
    for name,original in sorted(s.items()):
        q,sc=quantize(original,p); dtype={torch.float32:'F32',torch.float16:'F16',torch.int8:'I8'}[q.dtype]
        m['tensors'][name]={'shape':list(q.shape),'dtype':dtype,'parts':[]}
        rowbytes=q[0].numel()*q.element_size()+(4 if sc is not None else 0)
        rows=max(1,(LIMIT-65536)//rowbytes)
        for start in range(0,q.shape[0],rows):
            part=q[start:start+rows].contiguous(); key=name+f'.part{start}'
            group={key:part}; r={'key':key,'shape':list(part.shape),'dtype':dtype,'row_start':start}
            if sc is not None:
                sn=key+'__scale'; group[sn]=sc[start:start+rows].contiguous()
                r['quant']={'scheme':'symmetric_per_row','axis':0,'scale_tensor':sn}
            size=sum(t.numel()*t.element_size() for t in group.values())
            if pending and payload+size+65536>LIMIT: flush()
            pending.update(group); refs.append(r);payload+=size
            m['tensors'][name]['parts'].append(r)
    flush();m['total_bytes']=sum(x['byteLength'] for x in m['shards'])
    (folder/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    loaded=reload_bundle(p); worst=0
    for name,t in s.items():
        q,sc=quantize(t,p); expected=q.float() if sc is None else q.float()*sc[:,None]
        assert torch.equal(loaded[name],expected),name
        worst=max(worst,stats(loaded[name],t)['relRms'])
    return {k:m[k] for k in ['total_bytes','total_params','num_tensors','shards']}|{'strict_roundtrip':True,'max_tensor_quantization_relRms':worst}
if __name__=='__main__':
    constants={'time':torch.exp(torch.arange(128,dtype=torch.float32)*(-math.log(2000)/127)).tolist(),'position':(2056**(2*torch.arange(64)/128)).tolist(),'distance':torch.linspace(.1,3,63).tolist()}
    (ROOT/'shared/constants.js').write_text('// Analytic feature constants, evaluated by pinned PyTorch (independent of L and T).\nexport const constants='+json.dumps(constants)+';\n')
    s=state(); results={}
    for p in ['fp32','fp16','int8']:
        results[p]=export(s,p);print(p,results[p]['total_bytes'],flush=True)
    (ROOT/'fixtures/export_metrics.json').write_text(json.dumps(results,indent=2)+'\n')
