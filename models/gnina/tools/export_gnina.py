"""Reproducible capped safetensors bundles, including multipart row splitting."""
import json, hashlib
import numpy as np
import torch
from safetensors.numpy import save, load_file
from download_weights import ROOT, COMMIT, MODELS, checkpoint, fetch
CAP=25_000_000

def export(state, precision, destination, config, cap=CAP):
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
        quantize=value.ndim >= 2
        dtype='F32';scale=None
        if quantize and precision=='fp16':value=value.astype('<f2');dtype='F16'
        if quantize and precision=='int8':
            scale=np.maximum(np.abs(value.reshape(value.shape[0], -1)).max(axis=1)/127,1e-30).astype('<f4')
            value=np.clip(np.round(value/scale.reshape((-1,) + (1,)*(value.ndim-1))),-127,127).astype('i1');dtype='I8'
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
    manifest={'format':'gnina-safetensors-v1','model':'GNINA','variant':'default','precision':precision,
      'upstream_commit':COMMIT,'total_params':sum(v.numel() for v in state.values()),'num_tensors':len(state),
      'total_bytes':sum(s['byteLength'] for s in shards),'config':config,
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
            if 'quant' in part:x*=s[part['quant']['scale_tensor']].reshape((-1,) + (1,)*(x.ndim-1))
            parts.append(x)
        tensors[name]=torch.from_numpy(np.concatenate(parts,axis=0))
    return tensors

def load(name):
    extra = {"metadata": ""}
    model = torch.jit.load(str(checkpoint(name)), map_location="cpu", _extra_files=extra).eval()
    return model, json.loads(extra["metadata"])

def restore(model, name, all_weights):
    state = {key: value if not value.is_floating_point() else all_weights[name + "/" + key]
             for key, value in model.state_dict().items()}
    model.load_state_dict(state, strict=True)

@torch.inference_mode()
def main():
    fetch()
    torch.set_num_threads(3)
    torch.manual_seed(13)
    probe = torch.rand(1, 28, 48, 48, 48)
    state, configs, references = {}, {}, {}
    for name in MODELS:
        model, configs[name] = load(name)
        references[name] = torch.cat([x.flatten() for x in model(probe)])
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                state[name + "/" + key] = value.contiguous()
            elif not key.endswith("num_batches_tracked"):
                raise ValueError(key)
    metrics = {}
    for precision in ["fp32", "fp16", "int8"]:
        path = ROOT / "weights/default" / precision
        manifest = export(state, precision, path, {"ensemble":list(MODELS), "metadata":configs})
        decoded = reload_bundle(path)
        assert set(decoded) == set(state)
        errors = {}
        for name in MODELS:
            model, _ = load(name)
            restore(model, name, decoded)
            actual = torch.cat([x.flatten() for x in model(probe)])
            errors[name] = float((actual - references[name]).norm() / references[name].norm())
        for key, value in state.items():
            assert value.shape == decoded[key].shape
            if precision == "fp32" or value.ndim < 2:
                assert torch.equal(value, decoded[key])
        if precision == "fp32":
            assert max(errors.values()) == 0
        metrics[precision] = {k:manifest[k] for k in ("total_bytes","total_params","num_tensors")}
        metrics[precision]["probe_relRMS"] = errors
        print(precision, metrics[precision])
    (ROOT/"fixtures/export_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")

if __name__ == "__main__":
    main()
