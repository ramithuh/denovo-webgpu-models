"""Deterministic packed safetensors export for La-Proteina motif scaffolding (AE3 + LD4..LD7).

- Strict shard size cap: <= 25,000,000 bytes per .safetensors file (including header).
- AE3 exports BOTH encoder.* and decoder.* inference tensors.
- LD4..LD7 export nn.* inference tensors with authoritative motif conditioning metadata.
- Verifies SHA-256 of source NGC checkpoints and strict round-trip PyTorch reload.
"""
import argparse
import hashlib
import json
import torch
from safetensors.torch import save

from download_motif_weights import MOTIF_CHECKPOINTS, digest
from motif_common import (
    COMMIT,
    MOTIF_MODEL_META,
    ROOT,
    dequantize,
    make_motif_model,
    model_state,
    quantize,
    reload_motif_bundle,
    stats,
)

LIMIT = 25_000_000


def export_model(model: str, precision: str, cfg: dict, state: dict) -> dict:
    weights_root = (ROOT / 'weights').resolve()
    if not str(weights_root).startswith('/home/ruh/flame/'):
        raise RuntimeError(f'Refusing to write weights outside /home/ruh/flame/: {weights_root}')
    folder = ROOT / 'weights' / model.lower() / precision
    folder.mkdir(parents=True, exist_ok=True)
    # Remove any stale shards before writing deterministic shards
    for stale in folder.glob('*.safetensors'):
        stale.unlink()

    manifest = {
        'format': 'la-proteina-safetensors-v1',
        'model': model,
        'precision': precision,
        'upstream_commit': COMMIT,
        'checkpoint': {
            'name': MOTIF_CHECKPOINTS[model][0],
            'sha256': MOTIF_CHECKPOINTS[model][1],
        },
        'config': cfg,
        'scope': MOTIF_MODEL_META[model]['scope'],
        'shards': [],
        'tensors': {},
    }
    pending = {}
    records = []

    def flush():
        if not pending:
            return
        blob = save(pending)
        assert len(blob) <= LIMIT, f'Shard exceeded 25 MB limit: {len(blob)}'
        file = f'model-{len(manifest["shards"]) + 1:03}.safetensors'
        (folder / file).write_bytes(blob)
        manifest['shards'].append({
            'file': file,
            'byteLength': len(blob),
            'sha256': hashlib.sha256(blob).hexdigest(),
        })
        for rec in records:
            rec['shard'] = file
        pending.clear()
        records.clear()

    for name, v in sorted(state.items()):
        packed, scale = quantize(v, precision)
        dtype = {torch.float32: 'F32', torch.float16: 'F16', torch.int8: 'I8'}[packed.dtype]
        rec = {'shape': list(v.shape), 'dtype': dtype, 'parts': []}
        manifest['tensors'][name] = rec
        rows = v.shape[0]
        width = v.numel() // rows
        per_row = width * packed.element_size() + (4 if scale is not None else 0)
        rows_per = max(1, (LIMIT - 16384) // per_row)
        for start in range(0, rows, rows_per):
            end = min(rows, start + rows_per)
            key = name if end == rows and start == 0 else f'{name}.__part{start}'
            group = {key: packed[start:end].contiguous()}
            part = {'key': key, 'row_start': start, 'shape': list(group[key].shape)}
            if scale is not None:
                skey = key + '.__scale'
                group[skey] = scale[start:end].contiguous()
                part['scale_tensor'] = skey
            if pending and len(save({**pending, **group})) > LIMIT:
                flush()
            assert len(save(group)) <= LIMIT
            pending.update(group)
            records.append(part)
            rec['parts'].append(part)
    flush()

    manifest['total_bytes'] = sum(x['byteLength'] for x in manifest['shards'])
    manifest['total_params'] = sum(v.numel() for v in state.values())
    manifest['num_tensors'] = len(state)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    restored = reload_motif_bundle(model, precision)
    max_error = 0.0
    for name, v in state.items():
        expected = dequantize(*quantize(v, precision))
        assert torch.equal(restored[name], expected), f'Round-trip mismatch on {model} {precision} {name}'
        max_error = max(max_error, stats(restored[name], v)['maxAbs'])

    # Strict reload into upstream PyTorch module(s) verifies every parameter name and shape.
    make_motif_model(model, state=restored, cfg=cfg)
    return {
        'total_bytes': manifest['total_bytes'],
        'total_params': manifest['total_params'],
        'num_tensors': len(state),
        'shards': len(manifest['shards']),
        'max_shard_bytes': max(s['byteLength'] for s in manifest['shards']),
        'roundtrip_maxAbs': 0.0,
        'quantization_maxAbs': max_error,
        'strict_load': True,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['AE3', 'LD4', 'LD5', 'LD6', 'LD7'])
    p.add_argument('--precisions', nargs='+', default=['fp32', 'fp16', 'int8'])
    p.add_argument('--output', default='.cache/fixtures/motif_export_metrics.json')
    args = p.parse_args()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {'upstream_commit': COMMIT, 'models': {}}

    for model in args.models:
        ckpt_path = ROOT / 'weights' / MOTIF_CHECKPOINTS[model][0]
        assert ckpt_path.exists(), f'Missing checkpoint {ckpt_path}'
        assert digest(ckpt_path) == MOTIF_CHECKPOINTS[model][1], f'SHA-256 mismatch on {ckpt_path}'
        cfg, state = model_state(model)
        metrics['models'][model] = {}
        for precision in args.precisions:
            result = export_model(model, precision, cfg, state)
            metrics['models'][model][precision] = result
            print(f'{model} {precision}: {result}', flush=True)

    out_path.write_text(json.dumps(metrics, indent=2) + '\n')
    print(f'Wrote {out_path}', flush=True)
