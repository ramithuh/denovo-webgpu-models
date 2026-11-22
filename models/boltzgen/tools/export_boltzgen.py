"""Export verified checkpoints as bounded, SHA-256-addressed safetensors shards."""
from environment import ROOT, COMMIT, REVISION, checkpoint
import argparse
import hashlib
import json
import math
import torch
from safetensors.torch import save_file, load_file
from audit import plain

CAP = 24_000_000  # leaves ample space for the safetensors JSON header

def quantize(name, value, precision):
    value = value.detach().cpu().contiguous().clone()
    # Embeddings/Fourier projections and vectors retain their original precision.
    matrix = value.ndim == 2 and name.endswith('.weight') and not any(
        s in name for s in ('conditioning_init', 'fourier', 'token_bonds_type'))
    if precision == 'fp16' and matrix:
        return {name: value.half()}
    if precision == 'int8' and matrix:
        scale = value.abs().amax(dim=1).clamp_min(1e-30) / 127
        return {name: (value / scale[:, None]).round().clamp(-127, 127).to(torch.int8),
                name + '.__scale': scale}
    return {name: value}

def export(variant, precision):
    source = checkpoint(variant)
    state = source['state_dict']
    dest = ROOT / 'weights' / variant / precision
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 'boltzgen-safetensors-v1', 'model': 'boltzgen-1', 'variant': variant,
        'precision': precision, 'upstream_commit': COMMIT, 'checkpoint_revision': REVISION,
        'total_params': sum(t.numel() for t in state.values()), 'num_tensors': len(state),
        'config': plain(source['hyper_parameters']), 'shards': [], 'tensors': {}}
    pending, size = {}, 0
    roundtrip_sse, original_ss, max_abs = 0., 0., 0.
    def flush():
        nonlocal pending, size, roundtrip_sse, original_ss, max_abs
        if not pending:
            return
        filename = f'model-{len(manifest["shards"]):04d}.safetensors'
        path = dest / filename
        save_file(pending, path)
        length = path.stat().st_size
        assert length < 25_000_000, (filename, length)
        with path.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        manifest['shards'].append({'file': filename, 'byteLength': length, 'sha256': sha})
        restored = load_file(path)
        assert set(restored) == set(pending)
        for name, tensor in restored.items():
            assert torch.equal(tensor, pending[name]), name
            manifest['tensors'][name] = {'shard': filename, 'shape': list(tensor.shape),
                'dtype': str(tensor.dtype).removeprefix('torch.')}
            if name.endswith('.__scale'):
                continue
            value = tensor.float()
            scale_name = name + '.__scale'
            if scale_name in restored:
                value = value * restored[scale_name][:, None]
                manifest['tensors'][name]['scale'] = scale_name
            original = state[name].float()
            error = value - original
            roundtrip_sse += error.double().square().sum().item()
            original_ss += original.double().square().sum().item()
            max_abs = max(max_abs, error.abs().max().item() if error.numel() else 0)
        pending, size = {}, 0
    for name, tensor in state.items():
        group = quantize(name, tensor, precision)
        bytes_ = sum(t.numel() * t.element_size() for t in group.values())
        # The audited maximum tensor is 9 MiB: no multipart export is necessary.
        if bytes_ >= CAP:
            raise ValueError(f'Tensor group needs row splitting: {name}')
        if size + bytes_ > CAP:
            flush()
        pending.update(group)
        size += bytes_
    flush()
    manifest['total_bytes'] = sum(s['byteLength'] for s in manifest['shards'])
    manifest['roundtrip'] = {'relRMS': math.sqrt(roundtrip_sse / original_ss), 'maxAbs': max_abs,
        'all_tensor_bytes_verified': True, 'scope': 'weights; output drift measured by dump_oracle.py'}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(variant, precision, manifest['total_bytes'], manifest['roundtrip'], flush=True)
    return {'variant': variant, 'precision': precision, 'shards': len(manifest['shards']),
        'total_bytes': manifest['total_bytes'], **manifest['roundtrip']}

def reload_bundle(model, variant, precision):
    dest = ROOT / 'weights' / variant / precision
    manifest = json.loads((dest / 'manifest.json').read_text())
    state = {}
    for shard in manifest['shards']:
        path = dest / shard['file']
        assert path.stat().st_size == shard['byteLength']
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == shard['sha256']
        tensors = load_file(path)
        for name, tensor in tensors.items():
            if name.endswith('.__scale'):
                continue
            state[name] = tensor.float() * tensors[name + '.__scale'][:, None] if tensor.dtype == torch.int8 else tensor.float()
    migrated = {'state_dict': state}
    model.on_load_checkpoint(migrated)
    model.load_state_dict(migrated['state_dict'], strict=True, assign=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['diverse', 'adherence'])
    parser.add_argument('--precisions', nargs='+', default=['fp32', 'fp16', 'int8'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    metrics = [export(v, p) for v in args.variants for p in args.precisions]
    (ROOT / 'fixtures/export_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
