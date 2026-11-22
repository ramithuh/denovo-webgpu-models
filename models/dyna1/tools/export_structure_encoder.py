#!/usr/bin/env python3
"""Export additive ESM-3 VQ-VAE StructureTokenEncoder + structure_tokens_embed delta bundle."""
import argparse, hashlib, json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'dyna1-safetensors-v1'
STRUCTURE_ENCODER_CONFIG = dict(
    d_model=1024,
    layers=2,
    geom_heads=128,
    ffn_size=4096,
    knn=16,
    codebook_size=4096,
    code_dim=128,
    relative_position_bins=32,
    structure_vocab_size=4101,
    mask_token_id=4096,
    eos_token_id=4097,
    bos_token_id=4098,
    pad_token_id=4099,
    chainbreak_token_id=4100,
    sequence_mask_token_id=32,
)
SKIP_KEYS = {'codebook.N', 'codebook.z_avg'}
F32_KEEP_2D = {'encoder.relative_positional_embedding.embedding.weight', 'encoder.codebook.embeddings', 'structure_tokens_embed_delta'}


def build_state(cache: Path):
    enc = torch.load(cache / 'esm3_structure_encoder_v0.pth', map_location='cpu', weights_only=True)
    sd = {}
    for k, v in enc.items():
        if k in SKIP_KEYS:
            continue
        sd[f'encoder.{k}'] = v.detach().float().contiguous()
    base = torch.load(cache / 'esm3_sm_open_v1.pth', map_location='cpu', weights_only=True, mmap=True)
    struct_embed = base['encoder.structure_tokens_embed.weight'].detach().float()
    assert list(struct_embed.shape) == [4101, 1536], struct_embed.shape
    delta = (struct_embed - struct_embed[4096:4097]).contiguous()
    sd['structure_tokens_embed_delta'] = delta
    return sd


def export_structure_encoder(sd, outdir: Path, cap: int = 25_000_000):
    dest = outdir / 'structure_encoder'
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob('shard-*.safetensors'):
        old.unlink()
    tensors, shards, pending, records = {}, [], {}, []
    size = 0
    err = ref = 0.0

    def flush():
        nonlocal pending, records, size
        if not pending:
            return
        filename = f'shard-{len(shards):03d}.safetensors'
        p = dest / filename
        save_file(pending, str(p), metadata={'format': FORMAT, 'precision': 'fp16-mixed'})
        raw = p.read_bytes()
        assert len(raw) <= cap, (p, len(raw))
        shards.append(dict(
            file=filename,
            bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            num_tensors=len(records),
            num_raw_tensors=len(pending),
        ))
        for r in records:
            r['shard'] = filename
        pending, records, size = {}, [], 0

    for name, w in sd.items():
        w = w.float()
        use_f16 = (w.ndim == 2 and name not in F32_KEEP_2D)
        q = w.half() if use_f16 else w
        dtype = 'F16' if use_f16 else 'F32'
        restored = q.float()
        err += float(((restored - w).double() ** 2).sum())
        ref += float((w.double() ** 2).sum())
        row_bytes = q[0].numel() * q.element_size()
        rows = max(1, (cap - 65536) // row_bytes)
        parts = []
        for start in range(0, q.shape[0], rows):
            end = min(start + rows, q.shape[0])
            split = q.shape[0] > rows
            key = f'{name}.__part{len(parts):03d}' if split else name
            data = {key: q[start:end].contiguous()}
            rec = dict(key=key, dtype=dtype, shape=list(q[start:end].shape), row_start=start)
            nbytes = sum(t.numel() * t.element_size() for t in data.values()) + len(key) * 4 + 1024
            if size + nbytes > cap - 8192:
                flush()
            pending.update(data)
            records.append(rec)
            size += nbytes
            parts.append(rec)
        tensors[name] = dict(dtype=dtype, shape=list(w.shape), parts=parts) if len(parts) > 1 else parts[0]

    flush()
    manifest = dict(
        format=FORMAT,
        model='dyna1-esm3-structure-encoder',
        precision='fp16-mixed',
        source={
            'repo': 'EvolutionaryScale/esm3-sm-open-v1',
            'structure_encoder': 'esm3_structure_encoder_v0.pth',
            'backbone': 'esm3_sm_open_v1.pth',
            'upstream_commit': '814a453fd85305536b919901e41062b91d5c9d86',
        },
        total_params=sum(w.numel() for w in sd.values()),
        num_tensors=len(sd),
        total_bytes=sum(s['bytes'] for s in shards),
        config=STRUCTURE_ENCODER_CONFIG,
        shards=shards,
        tensors=tensors,
        weight_rel_rms=(err / ref) ** 0.5,
        fp32_exceptions=sorted(F32_KEEP_2D),
    )
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('structure_encoder', manifest['total_params'], manifest['total_bytes'], len(shards), manifest['weight_rel_rms'], flush=True)
    return manifest


def load_structure_bundle(path: Path):
    m = json.loads((path / 'manifest.json').read_text())
    cache, sd = {}, {}
    for name, rec in m['tensors'].items():
        vals = []
        for part in rec.get('parts', [rec]):
            f = part['shard']
            if f not in cache:
                cache[f] = load_file(str(path / f))
            vals.append(cache[f][part['key']].float())
        sd[name] = torch.cat(vals, dim=0)
    return sd


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=ROOT / '.cache')
    p.add_argument('--outdir', type=Path, default=ROOT / 'weights')
    args = p.parse_args()
    sd = build_state(args.cache)
    export_structure_encoder(sd, args.outdir)
