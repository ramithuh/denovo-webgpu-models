#!/usr/bin/env python3
"""Validate the release artifacts and write the Hugging Face model card locally."""
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundles', nargs='+', default=['fp16', 'int8', 'structure_encoder'])
    args = ap.parse_args()
    dest = ROOT / 'weights'
    index = {}
    for name in args.bundles:
        p = dest / name / 'manifest.json'
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        for shard in m['shards']:
            raw = (p.parent / shard['file']).read_bytes()
            assert len(raw) == shard['bytes'] and len(raw) <= 25_000_000, (name, shard['file'], len(raw))
            assert hashlib.sha256(raw).hexdigest() == shard['sha256'], (name, shard['file'])
        index[name] = {
            'manifest': f'{name}/manifest.json',
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'bytes': m['total_bytes'],
            'shards': len(m['shards']),
        }
    (dest / 'bundle-index.json').write_text(json.dumps(index, indent=2) + '\n')
    (dest / 'README.md').write_text('''---
license: mit
base_model:
- gelnesr/Dyna-1
- EvolutionaryScale/esm3-sm-open-v1
tags:
- protein
- webgpu
- dyna1
- safetensors
---
# Dyna-1 browser inference bundles

Primary ESM-3 Dyna-1 model (840,417,793 main inference parameters) plus an optional additive ESM-3 VQ-VAE structure encoder bundle (`structure_encoder/`, 36,939,648 parameters) enabling full **sequence-only**, **sequence + structure** (`dyna1.py --pdb ... --use_pdb_seq`), and **structure-only** (`dyna1.py --pdb ...`) inference in `dyna1-webgpu`.

These are custom `dyna1-safetensors-v1` bundles for the `dyna1-webgpu` JavaScript runtime, not Transformers checkpoints. Load the manifest and all of its referenced shards.

- `fp16/`: F16 matrices with F32 vectors and the first two classifier blocks retained in F32 for output accuracy (1,744,615,172 bytes, 97 shards).
- `int8/`: symmetric per-row INT8 matrices with colocated F32 scales and F32 vectors (843,626,504 bytes, 43 shards).
- `structure_encoder/` *(additive, optional)*: ESM-3 VQ-VAE `StructureTokenEncoder` (2-layer geometric transformer, F16 linear matrices with F32 vectors/embeddings/codebook, 100% exact discrete token match vs upstream BF16/FP32) + `structure_tokens_embed_delta` `[4101, 1536]` in F32 (`struct_embed[t] - struct_embed[4096]`), enabling zero-breaking-change structure conditioning on top of the existing `fp16/` and `int8/` shards.

Every safetensors file is at most 25,000,000 bytes; large matrices have row-split manifest entries.

The output is a per-residue exchange probability, `sigmoid(logit - log(0.05/0.95))`, plus classifier logits. It is not a calibrated physical exchange rate.

Sources: [Dyna-1](https://github.com/WaymentSteeleLab/Dyna-1), [official classifier](https://huggingface.co/gelnesr/Dyna-1), [ESM-3 small](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1). Cite Wayment-Steele et al., *Learning millisecond protein dynamics from what is missing in NMR spectra*, Nature (2026), doi:10.1038/s41586-026-10989-4.

The runtime uses a base URL ending at this repository's pinned `resolve/<revision>/` root, selects `int8`, `fp16`, or `fp32` for the main backbone+classifier, and lazily fetches `structure_encoder/` only when 3D structure coordinates or structure tokens are supplied.
''')
    print(json.dumps(index, indent=2))
    print('Validated release prepared locally. Nothing uploaded.')
