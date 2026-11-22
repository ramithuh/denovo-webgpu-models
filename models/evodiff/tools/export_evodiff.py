"""Reproducible whole-tensor sharding. Scale companions are colocated atomically."""
import hashlib
import json
import torch
from safetensors.torch import save
from oracle_common import ROOT, COMMITS, model, reload_bundle, tokens, stats, A_CASES, input_mask
from download_weights import SHA256, MD5

LIMIT = 25_000_000

def export(state, precision):
    folder = ROOT / 'weights/oa_dm_38m' / precision
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 'evodiff-safetensors-v1', 'model': 'OA_DM_38M', 'precision': precision,
        'upstream': COMMITS, 'checkpoint': {'record': '8045076', 'file': 'oaar-38M.tar', 'sha256': SHA256, 'md5': MD5, 'license': 'CC-BY-4.0'},
        'config': {'vocab': 31, 'embedding': 8, 'width': 1024, 'hidden': 512, 'layers': 16, 'kernel': 5,
            'dilations': [1,2,4,8,16,32,64,128]*2, 'activation': 'gelu_exact', 'eps': 1e-5, 'timesteps': None, 'sampling_vocab': 21},
        'shards': [], 'tensors': {}}
    pending, pending_names = {}, []
    def flush():
        if not pending: return
        blob = save(pending)
        assert len(blob) <= LIMIT
        name = f'model-{len(manifest["shards"])+1:03}.safetensors'
        (folder / name).write_bytes(blob)
        manifest['shards'].append({'file': name, 'byteLength': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()})
        for n in pending_names: manifest['tensors'][n]['shard'] = name
        pending.clear()
        pending_names.clear()
    for name, original in sorted(state.items()):
        t = original.detach().cpu().contiguous()
        rec = {'shape': list(t.shape), 'dtype': 'F32'}
        group = {}
        if t.ndim == 3 and name.endswith('.weight'):
            if precision == 'fp16':
                t, rec['dtype'] = t.half(), 'F16'
            elif precision == 'int8':
                scale = t.abs().flatten(1).amax(1).clamp_min(1e-30) / 127
                t = (t / scale[:, None, None]).round().clamp(-127, 127).to(torch.int8)
                group[name + '.__scale'] = scale
                rec.update(dtype='I8', quant={'scheme': 'symmetric_per_output_channel', 'axis': 0, 'scale_tensor': name + '.__scale'})
        group[name] = t
        # Include exact safetensors header cost in every packing decision.
        if pending and len(save({**pending, **group})) > LIMIT: flush()
        pending.update(group)
        pending_names.append(name)
        manifest['tensors'][name] = rec
    flush()
    manifest['total_bytes'] = sum(s['byteLength'] for s in manifest['shards'])
    manifest['total_params'] = sum(t.numel() for t in state.values())
    manifest['num_tensors'] = len(state)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return {k: manifest[k] for k in ['total_bytes', 'total_params', 'num_tensors', 'shards']}

if __name__ == '__main__':
    original = model()
    results = {}
    with torch.inference_mode():
        refs = {name: original(tokens(seq), torch.tensor([0]), input_mask=input_mask(name, seq)) for name, seq in A_CASES}
        for precision in ['fp32', 'fp16', 'int8']:
            results[precision] = export(original.state_dict(), precision)
            restored = model(reload_bundle(precision))
            results[precision]['roundtrip'] = {}
            for name, seq in A_CASES:
                pred = restored(tokens(seq), torch.tensor([0]), input_mask=input_mask(name, seq))
                results[precision]['roundtrip'][name] = {'logits': stats(pred, refs[name]), 'probabilities': stats(pred.softmax(-1), refs[name].softmax(-1))}
            print(precision, results[precision], flush=True)
    (ROOT / 'fixtures/export_metrics.json').write_text(json.dumps(results, indent=2)+'\n')
