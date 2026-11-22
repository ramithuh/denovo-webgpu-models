"""FP32/F16/per-row-I8 safetensors, with bounded shards and strict reload."""
import argparse, hashlib, json
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from reference import ROOT, COMMIT, model_for, relrms
CAP = 25_000_000
# Keep this geometric projection F32: I8 rounding puts a vector within 2.4e-9
# of the hard 2*epsilon gate, making equivalent GEMM reductions flip the gate.
INT8_F32_EXCEPTIONS = {'esmif1': {'encoder.gvp_encoder.encoder_layers.0.conv.message_func.2.wv.weight'}}

def export(variant, precision):
    model, alphabet = model_for(variant)
    state = model.state_dict()
    # Positional embedding's uninitialized one-element dtype marker is not a parameter.
    # It never contributes numerically; record it as zero rather than exporting heap data.
    for name in state:
        if name.endswith('._float_tensor'): state[name] = torch.zeros_like(state[name])
    dest = ROOT / 'weights' / variant / precision
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dict(format='gvp-transformer-safetensors-v1', model=variant, precision=precision,
        upstream_commit=COMMIT,total_params=sum(p.numel() for p in model.parameters()),
        num_tensors=len(state),total_bytes=0,config={k:v for k,v in vars(model.args).items()
        if k.startswith(('gvp_', 'encoder_', 'decoder_')) and isinstance(v,(int,float,bool,str,type(None)))},
        vocabulary=alphabet.all_toks, f32_exceptions=sorted(INT8_F32_EXCEPTIONS.get(variant,set())) if precision=='int8' else [], shards=[],tensors={})
    shard, records, rawbytes = {}, [], 0
    def flush():
        nonlocal shard, records, rawbytes
        if not shard: return
        filename=f'model-{len(manifest["shards"])+1:04d}.safetensors'
        path=dest/filename
        save_file(shard, str(path))
        data=path.read_bytes()
        assert len(data)<=CAP
        manifest['shards'].append(dict(file=filename,byteLength=len(data),sha256=hashlib.sha256(data).hexdigest()))
        manifest['total_bytes']+=len(data)
        for rec in records: rec['shard']=filename
        shard,records,rawbytes={}, [], 0
    # replaced below; explicit grouping keeps I8 rows and scales colocated.
    error_num=error_den=0.
    for name, value in state.items():
        value=value.detach().contiguous().float()
        quantize=value.ndim==2 and 'embed_tokens' not in name and not (precision=='int8' and name in INT8_F32_EXCEPTIONS.get(variant,set()))
        dtype='F32' if precision=='fp32' or not quantize else 'F16' if precision=='fp16' else 'I8'
        scale=None
        if dtype=='I8':
            scale=value.abs().amax(dim=1).clamp(min=1e-12)/127
            stored=(value/scale[:,None]).round().clamp(-127,127).to(torch.int8)
            decoded=stored.float()*scale[:,None]
        elif dtype=='F16':
            stored=value.half(); decoded=stored.float()
        else: stored=value; decoded=value
        error_num+=float((decoded.double()-value.double()).square().sum())
        error_den+=float(value.double().square().sum())
        rec=dict(dtype=dtype,shape=list(value.shape))
        manifest['tensors'][name]=rec
        row_bytes=stored[0].numel()*stored.element_size()+(4 if scale is not None else 0)
        rows=max(1,(CAP-65536)//row_bytes)
        parts=[]
        for start in range(0,len(value),rows):
            chunk=stored[start:start+rows].contiguous()
            key=name if len(value)<=rows else f'{name}.__part_{start}'
            nbytes=chunk.numel()*chunk.element_size()
            if scale is not None: nbytes+=len(chunk)*4
            if rawbytes+nbytes+65536>CAP: flush()
            part=dict(key=key,dtype=dtype,shape=list(chunk.shape),row_start=start)
            if scale is not None:
                skey=f'{key}.__scale'; shard[skey]=scale[start:start+rows].contiguous()
                part['quant']=dict(scheme='symmetric_per_row',axis=0,scale_tensor=skey)
            shard[key]=chunk
            rawbytes+=nbytes
            records.append(part);parts.append(part)
        # Keep references until shard assignments have been written by flush().
        rec['parts']=parts
    flush()
    manifest['weight_relRMS']=(error_num/max(error_den,1e-30))**.5
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    loaded=reload_bundle(dest)
    model.load_state_dict(loaded, strict=True)
    for name,v in loaded.items():
        original=state[name]
        if precision=='fp32': assert torch.equal(v,original), name
    print(variant,precision,manifest['total_params'],len(state),len(manifest['shards']),manifest['total_bytes'],manifest['weight_relRMS'],flush=True)
    return manifest

def reload_bundle(dest):
    manifest=json.loads((dest/'manifest.json').read_text())
    shards={}
    for s in manifest['shards']:
        data=(dest/s['file']).read_bytes()
        assert len(data)==s['byteLength'] and hashlib.sha256(data).hexdigest()==s['sha256']
        shards[s['file']]=load_file(str(dest/s['file']))
    result={}
    for name,rec in manifest['tensors'].items():
        parts=[]
        for p in rec['parts']:
            vals=shards[p['shard']][p['key']].float()
            if p['dtype']=='I8': vals=vals*shards[p['shard']][p['quant']['scale_tensor']][:,None]
            parts.append(vals)
        result[name]=torch.cat(parts,dim=0)
        assert list(result[name].shape)==rec['shape']
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--variants',nargs='+',default=['antifold','esmif1'])
    parser.add_argument('--precisions',nargs='+',default=['fp32','fp16','int8'])
    args=parser.parse_args()
    for v in args.variants:
        for p in args.precisions: export(v,p)
