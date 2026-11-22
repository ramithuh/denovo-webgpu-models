"""Strict, size-capped, row-split safetensors export of original checkpoints."""
import argparse
import hashlib
import json
import pickle
import types
from pathlib import Path
from environment import ROOT, COMMIT
import numpy as np
import torch
import yaml
from safetensors.numpy import save, load_file

CAP = 25_000_000


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('protein_diffusion.'):
            module = module.replace('protein_diffusion.', 'protpardelle.core.', 1)
            if name == 'Noise_Embedding':
                name = 'NoiseEmbedding'
        return super().find_class(module, name)


def checkpoint_state(path):
    compat = types.SimpleNamespace(Unpickler=LegacyUnpickler, __name__='pickle', load=pickle.load)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, pickle_module=compat)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    if 'model' in checkpoint and isinstance(checkpoint['model'], torch.nn.Module):
        return checkpoint['model'].state_dict()
    raise ValueError(f'Unrecognized checkpoint: {path}')


def config_for(name):
    family = name.split('_epoch')[0]
    if 'minimpnn' in name:
        family = 'cc58-minimpnn'
    path = ROOT / '.cache/model_params/configs' / f'{family}.yaml'
    if not path.exists():
        path = ROOT / 'examples/training' / f'{family}.yaml'
    config = yaml.safe_load(path.read_text())
    config['model'].setdefault('conditioning_style', 'concat')
    return config


def parameter_inventory(config,state):
    from protpardelle.core.models import CoordinateDenoiser,MiniMPNN
    from protpardelle.utils import dict_to_namespace
    model=torch.nn.Module()
    if any(k.startswith('struct_model.') for k in state):model.add_module('struct_model',CoordinateDenoiser(dict_to_namespace(config)))
    if any(k.startswith('mpnn_model.') for k in state):model.add_module('mpnn_model',MiniMPNN(dict_to_namespace(config)))
    parameters=dict(model.named_parameters())
    return {'total_params':sum(p.numel() for k,p in parameters.items() if k in state),
            'learned_params':sum(p.numel() for k,p in parameters.items() if k in state and p.requires_grad),
            'state_values':sum(p.numel() for p in state.values()),
            'state_only_tensors':[k for k in state if k not in parameters],
            'unserialized_optional_parameters':[k for k in parameters if k not in state]}


def quantize(name, data, precision):
    # Positional encoding projections and geometry remain F32.
    matrix = data.ndim == 2 and not any(x in name for x in ('relpos', 'relchain', 'embeddings', 'embedding.weight'))
    if precision == 'fp16' and matrix:
        return data.astype('<f2'), None
    if precision == 'int8' and matrix:
        scale = np.maximum(np.max(np.abs(data), axis=1), np.finfo(np.float32).tiny) / np.float32(127)
        return np.clip(np.rint(data / scale[:, None]), -127, 127).astype('i1'), scale.astype('<f4')
    return data.astype('<f4'), None


def reload_bundle(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    shards = {s['file']: load_file(path / s['file']) for s in manifest['shards']}
    result = {}
    for name, entry in manifest['tensors'].items():
        parts = []
        for part in entry.get('parts', [entry]):
            raw = shards[part['shard']][part['key']].astype(np.float32)
            if 'scale_tensor' in part:
                raw = raw * shards[part['shard']][part['scale_tensor']][:, None]
            parts.append(raw)
        result[name] = torch.from_numpy(np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0])
    return result


def export(name, state, config, precision):
    destination = ROOT / 'weights' / name / precision
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 'protpardelle-safetensors-v1', 'model': 'ProtPardelle-1c', 'variant': name,
                'precision': precision, 'upstream_commit': COMMIT, 'config': config,
                **parameter_inventory(config,state), 'num_tensors': len(state),
                'total_bytes': 0, 'shards': [], 'tensors': {}}
    pending, references = {}, []
    def flush():
        if not pending:
            return
        binary = save(pending)
        assert len(binary) <= CAP
        file = f'model-{len(manifest["shards"]):04}.safetensors'
        (destination / file).write_bytes(binary)
        manifest['shards'].append({'file': file, 'byteLength': len(binary), 'sha256': hashlib.sha256(binary).hexdigest()})
        manifest['total_bytes'] += len(binary)
        for ref in references:
            ref['shard'] = file
        pending.clear()
        references.clear()
    for key, tensor in state.items():
        data = tensor.detach().cpu().float().contiguous().numpy()
        raw, scale = quantize(key, data, precision)
        # Leave room for JSON headers, keys and colocated scale companions.
        row_bytes = int(np.prod(raw.shape[1:])) * raw.itemsize + (4 if scale is not None else 0) if raw.ndim else 4
        rows = max(1, (CAP - 65536) // row_bytes)
        count = raw.shape[0] if raw.ndim else 1
        parts = []
        for start in range(0, count, rows):
            piece = np.ascontiguousarray(raw[start:start + rows]) if raw.ndim else raw
            piece_key = key if count <= rows else f'{key}.__part_{start}'
            values = {piece_key: piece}
            part = {'key': piece_key, 'shape': list(piece.shape), 'dtype': {np.dtype('float32'): 'F32', np.dtype('float16'): 'F16', np.dtype('int8'): 'I8'}[piece.dtype], 'row_start': start}
            if scale is not None:
                scale_key = piece_key + '.__scale'
                values[scale_key] = np.ascontiguousarray(scale[start:start + rows])
                part['scale_tensor'] = scale_key
            if len(save(pending | values)) > CAP:
                flush()
            if len(save(values)) > CAP:
                raise ValueError(f'Tensor row exceeds shard cap: {key}')
            pending.update(values)
            references.append(part)
            parts.append(part)
        manifest['tensors'][key] = parts[0] if len(parts) == 1 else {'shape': list(raw.shape), 'dtype': parts[0]['dtype'], 'parts': parts}
    flush()
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    reloaded = reload_bundle(destination)
    error, norm = 0., 0.
    for key, original in state.items():
        expected = original.float().cpu()
        actual = reloaded[key]
        assert expected.shape == actual.shape, key
        # Check serialization against independently quantized values, bit for bit.
        q, scale = quantize(key, expected.numpy(), precision)
        target = q.astype(np.float32)
        if scale is not None:
            target *= scale[:, None]
        np.testing.assert_array_equal(actual.numpy(), target)
        error += float((expected.double() - actual.double()).square().sum())
        norm += float(expected.double().square().sum())
    return {'variant': name, 'precision': precision, 'tensors': len(state), 'state_values': manifest['state_values'],'total_params':manifest['total_params'],'learned_params':manifest['learned_params'],
            'shards': len(manifest['shards']), 'bytes': manifest['total_bytes'], 'weight_relRMS': (error / max(norm, 1e-30)) ** .5,
            'serialization_roundtrip': 'exact', 'upstream_output_roundtrip': 'pending'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', action='append')
    args = parser.parse_args()
    metrics = []
    for path in sorted((ROOT / '.cache/model_params/weights').glob('*.pth')):
        if args.variant and path.stem not in args.variant:
            continue
        state = checkpoint_state(path)
        config = config_for(path.stem)
        for precision in ('fp32', 'fp16', 'int8'):
            metric = export(path.stem, state, config, precision)
            print(json.dumps(metric), flush=True)
            metrics.append(metric)
    if not metrics:
        raise FileNotFoundError('No checkpoint weights; run tools/download_weights.py')
    (ROOT / 'fixtures/export_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    main()
