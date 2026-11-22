"""Reproducible capped safetensors bundles, including multipart row splitting."""
import json, hashlib
import numpy as np
import torch
from safetensors.numpy import save, load_file
from reference import ROOT, COMMIT, load_model, specialized, restore, run, metric
CAP=25_000_000

def export(state, precision, destination, cap=CAP):
    destination.mkdir(parents=True,exist_ok=True)
    shards=[]; tensors={}; pending={}; entries=[]
    def flush():
        if not pending:return
        data=save(pending);assert len(data)<=cap
        file=f'model-{len(shards)+1:03d}.safetensors'
        (destination/file).write_bytes(data)
        shards.append({'file':file,'byteLength':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        for entry in entries:entry['shard']=file
        pending.clear();entries.clear()
    for name,tensor in state.items():
        value=tensor.numpy()
        # Embeddings and geometry positional parameters remain F32.
        quantize=value.ndim==2 and name not in ['prot_mpnn.W_s.weight','prot_mpnn.features.embeddings.linear.weight']
        dtype='F32';scale=None
        if quantize and precision=='fp16':value=value.astype('<f2');dtype='F16'
        if quantize and precision=='int8':
            scale=np.maximum(np.abs(value).max(axis=1)/127,1e-30).astype('<f4')
            value=np.clip(np.round(value/scale[:,None]),-127,127).astype('i1');dtype='I8'
        row_bytes=value[0:1].nbytes+(4 if scale is not None else 0)
        rows=max(1,(cap-4096)//row_bytes)
        parts=[]
        for start in range(0,value.shape[0],rows):
            chunk=np.ascontiguousarray(value[start:start+rows]);key=name if rows>=value.shape[0] else f'{name}.__part{len(parts)}'
            raw={key:chunk};entry={'key':key,'dtype':dtype,'shape':list(chunk.shape),'row_start':start}
            if scale is not None:
                sk=key+'.__scale';raw[sk]=np.ascontiguousarray(scale[start:start+rows])
                entry['quant']={'scheme':'symmetric_per_row','axis':0,'scale_tensor':sk}
            if len(save({**pending,**raw}))>cap:flush()
            if len(save(raw))>cap:raise ValueError('One tensor row exceeds shard cap')
            pending.update(raw);entries.append(entry);parts.append(entry)
        tensors[name]=parts[0] if len(parts)==1 else {'shape':list(value.shape),'dtype':dtype,'parts':parts}
    flush()
    manifest={'format':'thermompnn-safetensors-v1','model':'ThermoMPNN','variant':'default','precision':precision,
      'upstream_commit':COMMIT,'total_params':sum(v.numel() for v in state.values()),'num_tensors':len(state),
      'total_bytes':sum(s['byteLength'] for s in shards),'config':{'hidden_dim':128,'neighbors':48,
      'encoder_layers':3,'decoder_layers':3,'head_dims':[384,64,32,21],'alphabet':'ACDEFGHIKLMNPQRSTVWYX',
      'max_length':2048,'layer_norm_epsilon':1e-5,'augment_eps':0,'message_scale':30,
      'decoder_visibility':'all','readout_layers':[2,1],'light_attention_kernel':'center_only'},
      'shards':shards,'tensors':tensors}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def reload_bundle(path):
    m=json.loads((path/'manifest.json').read_text());raw={}
    for shard in m['shards']:
        data=(path/shard['file']).read_bytes()
        assert len(data)==shard['byteLength'] and hashlib.sha256(data).hexdigest()==shard['sha256']
        raw[shard['file']]=load_file(path/shard['file'])
    tensors={}
    for name,entry in m['tensors'].items():
        parts=[]
        for part in entry.get('parts',[entry]):
            s=raw[part['shard']];x=s[part['key']].astype('float32')
            if 'quant' in part:x*=s[part['quant']['scale_tensor']][:,None]
            parts.append(x)
        tensors[name]=torch.from_numpy(np.concatenate(parts,axis=0))
    return tensors

if __name__=='__main__':
    model=load_model();state=specialized(model.state_dict())
    cases=json.loads((ROOT/'fixtures/cases.json').read_text())
    cases.append({'id':'full_length','pdb':'full.pdb','input':'full.pdb'})
    reference={c['id']:run(model,c,full_ssm=c['id']=='monomer') for c in cases if 'same_as' not in c}
    metrics={}
    for precision in ['fp32','fp16','int8']:
        dest=ROOT/'weights/default'/precision
        manifest=export(state,precision,dest);reloaded=reload_bundle(dest)
        weight_error=metric(np.concatenate([x.numpy().ravel() for x in reloaded.values()]),np.concatenate([x.numpy().ravel() for x in state.values()]))
        restore(model,reloaded)
        results={}
        for c in cases:
            if 'same_as' in c:results[c['id']]=results[c['same_as']];continue
            prediction=run(model,c);ref=reference[c['id']]
            results[c['id']]={'ddg_relRMS':metric(prediction['ddg'],ref['ddg']),
               'ddg_max_abs':float(np.max(np.abs(prediction['ddg']-ref['ddg']))),
               'correlation':float(np.corrcoef(prediction['ddg'].ravel(),ref['ddg'].ravel())[0,1]),
               'targeted_relRMS':metric(prediction['targeted'],ref['targeted'])}
            if precision=='fp32':assert results[c['id']]['ddg_relRMS']==0
        metrics[precision]={'shards':len(manifest['shards']),'bytes':manifest['total_bytes'],
          'total_params':manifest['total_params'],'num_tensors':manifest['num_tensors'],'weight_relRMS':weight_error,'cases':results}
        print(precision,json.dumps(metrics[precision]),flush=True)
    (ROOT/'fixtures/export_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
