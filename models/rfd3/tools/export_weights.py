#!/usr/bin/env python3
"""Export EMA inference weights, with exact schema/provenance and verified narrow shards.

No graph capture. Dtype selection uses actual upstream module classes. Shards are
content-addressed; the manifest is published last with an atomic file replacement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import torch
import struct
from checkpoint import ROOT, UPSTREAM, COMMIT, DEFAULT_CHECKPOINT, load_checkpoint

LIMIT = 25_000_000


def save(tensors, metadata):
    header = {'__metadata__': metadata}
    payloads, offset = [], 0
    for name, tensor in tensors.items():
        data = tensor.numpy().tobytes(order='C')
        header[name] = dict(dtype={torch.float32: 'F32', torch.float16: 'F16', torch.int8: 'I8'}[tensor.dtype],
                            shape=list(tensor.shape), data_offsets=[offset, offset + len(data)])
        payloads.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode()
    encoded += b' ' * (-len(encoded) % 8)
    return struct.pack('<Q', len(encoded)) + encoded + b''.join(payloads)



def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, sort_keys=True) + '\n').encode())


def inventory(net):
    from rfd3.model.layers.layer_utils import EmbeddingLayer
    modules = dict(net.named_modules())
    parameters = dict(net.named_parameters())
    result = {}
    for name, tensor in net.state_dict().items():
        owner, _, attr = name.rpartition('.')
        module = modules[owner]
        linear = (isinstance(module, torch.nn.Linear) and not isinstance(module, EmbeddingLayer)
                  and attr == 'weight' and tensor.ndim == 2)
        result[name] = dict(shape=list(tensor.shape), source_dtype=str(tensor.dtype),
                            numel=tensor.numel(), parameter=name in parameters,
                            module=type(module).__name__, role='linear' if linear else 'fp32')
    return result


def export(net, schema, directory, precision, checkpoint_hash):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(format='rfd3-packed-v1', precision=precision, checkpoint_sha256=checkpoint_hash,
                    upstream_commit=COMMIT, state_branch='shadow', tensors={}, constants={}, shards=[])
    pending, names, estimated = {}, [], 0

    def flush():
        nonlocal pending, names, estimated
        if not pending:
            return
        data = save(pending, metadata={'format': 'rfd3-packed-v1', 'precision': precision})
        if len(data) >= LIMIT:
            raise ValueError(f'Shard is too large: {len(data)}')
        digest = hashlib.sha256(data).hexdigest()
        filename = digest + '.safetensors'
        path = directory / filename
        if not path.exists() or digest_file(path) != digest:
            atomic_bytes(path, data)
        if path.stat().st_size != len(data) or digest_file(path) != digest:
            raise IOError(f'Write verification failed: {path}')
        manifest['shards'].append(dict(file=filename, byteLength=len(data), sha256=digest))
        for name in names:
            manifest['tensors'][name]['shard'] = filename
        pending, names, estimated = {}, [], 0

    for name, source in net.state_dict().items():
        info = schema[name]
        if not source.is_floating_point():
            manifest['constants'][name] = dict(shape=info['shape'], dtype=str(source.dtype), value=source.tolist())
            continue
        source = source.detach().float().contiguous()
        if not bool(torch.isfinite(source).all()):
            raise ValueError('Nonfinite checkpoint tensor: ' + name)
        tensors = {}
        entry = dict(shape=info['shape'], role=info['role'], key=name)
        if info['role'] == 'linear' and precision == 'fp16':
            tensors[name] = source.half()
            if not bool(torch.isfinite(tensors[name]).all()):
                raise ValueError('FP16 overflow: ' + name)
            entry.update(dtype='F16', packing='2xf16/u32')
        elif info['role'] == 'linear' and precision == 'int8':
            scales = source.abs().amax(dim=1) / 127
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            tensors[name] = torch.round(source / scales[:, None]).clamp(-127, 127).to(torch.int8)
            scale_name = name + '.__row_scale'
            tensors[scale_name] = scales.contiguous()
            entry.update(dtype='I8', packing='4xi8/u32', scale_key=scale_name, scale_axis=0)
        else:
            tensors[name] = source
            entry.update(dtype='F32', packing='f32')
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        # Conservative header reserve; exact encoded size is checked in flush().
        cost = size + sum(len(k.encode()) + 512 for k in tensors)
        if estimated + cost >= LIMIT:
            flush()
        if cost >= LIMIT:
            raise ValueError('Tensor requires splitting: ' + name)
        pending.update(tensors)
        names.append(name)
        estimated += cost
        manifest['tensors'][name] = entry
    flush()
    manifest['total_bytes'] = sum(s['byteLength'] for s in manifest['shards'])
    atomic_json(directory / 'manifest.json', manifest)
    print(json.dumps(dict(precision=precision, shards=len(manifest['shards']), bytes=manifest['total_bytes'],
                          max_shard_bytes=max(s['byteLength'] for s in manifest['shards']))), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, default=ROOT / 'weights')
    parser.add_argument('--precisions', nargs='+', choices=['fp32', 'fp16', 'int8'], default=['fp32', 'fp16', 'int8'])
    args = parser.parse_args()
    actual_commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_commit != COMMIT:
        raise ValueError('Upstream checkout does not match pinned commit')
    torch.set_num_threads(8)
    checkpoint_hash = digest_file(args.checkpoint)
    checkpoint, net, config = load_checkpoint(args.checkpoint)
    schema = inventory(net)
    provenance = dict(format='rfd3-architecture-v1', upstream_commit=COMMIT,
        checkpoint_sha256=checkpoint_hash, checkpoint_bytes=Path(args.checkpoint).stat().st_size,
        state_branch='shadow', parameter_count=sum(p.numel() for p in net.parameters()),
        state_numel=sum(t.numel() for t in net.state_dict().values()),
        tensor_count=len(schema), config=config, tensors=schema,
        source_files={str(p.relative_to(UPSTREAM)): digest_file(p) for p in sorted((UPSTREAM / 'src/rfd3/model').rglob('*.py'))},
        global_step=checkpoint['global_step'], current_epoch=checkpoint['current_epoch'])
    atomic_json(ROOT / 'fixtures/architecture.json', provenance)
    print(json.dumps({k: provenance[k] for k in ['checkpoint_sha256', 'parameter_count', 'state_numel', 'tensor_count']}), flush=True)
    for precision in args.precisions:
        export(net, schema, args.output / precision, precision, checkpoint_hash)


if __name__ == '__main__':
    main()
