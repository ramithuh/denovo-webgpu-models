#!/usr/bin/env python3
"""Validate the release artifacts and write the Hugging Face model card locally."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 dest=ROOT/'weights';index={}
 for precision in ['fp32','fp16','int8']:
  p=dest/precision/'manifest.json';m=json.loads(p.read_text())
  for shard in m['shards']:
   raw=(p.parent/shard['file']).read_bytes()
   assert len(raw)==shard['bytes'] and len(raw)<=25_000_000
   assert hashlib.sha256(raw).hexdigest()==shard['sha256']
  index[precision]={'manifest':f'{precision}/manifest.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':m['total_bytes'],'shards':len(m['shards'])}
 (dest/'bundle-index.json').write_text(json.dumps(index,indent=2)+'\n')
 (dest/'README.md').write_text('''---
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

Sequence-only specialization of the primary ESM-3 Dyna-1 model: 840,417,793 inference parameters. These are custom `dyna1-safetensors-v1` bundles for the dyna1-webgpu JavaScript runtime, not Transformers checkpoints. Load the manifest and all of its referenced shards.

`fp32/`: original FP32 inference weights. `fp16/`: F16 matrices with F32 vectors and the first two classifier blocks retained in F32 for output accuracy. `int8/`: symmetric per-row INT8 matrices with colocated F32 scales and F32 vectors. Every safetensors file is at most 25,000,000 bytes; large matrices have row-split manifest entries.

The input is a canonical amino acid sequence without BOS/EOS tokens. The output is a per-residue exchange probability, `sigmoid(logit - log(0.05/0.95))`, plus classifier logits. It is not a calibrated physical exchange rate. Structure-conditioned inference is not included.

The port has deterministic upstream CPU FP32 traces on Trp-cage and ubiquitin, JavaScript CPU intermediate parity, and real Chromium/Dawn WebGPU parity. See the runtime repository's README_WEBGPU.md and committed fixture metrics for exact measurements and scope. Quantized parity on these two sequences is not a comprehensive biological validation.

Sources: [Dyna-1](https://github.com/WaymentSteeleLab/Dyna-1), [official classifier](https://huggingface.co/gelnesr/Dyna-1), [ESM-3 small](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1). Cite Wayment-Steele et al., *Learning millisecond protein dynamics from what is missing in NMR spectra*, Nature (2026), doi:10.1038/s41586-026-10989-4.

The runtime uses a base URL ending at this repository's pinned `resolve/<revision>/` root, then selects `int8`, `fp16`, or `fp32`. `bundle-index.json` identifies each manifest's checksum; manifests identify every shard checksum.
''')
 print(json.dumps(index,indent=2))
 print('Validated release prepared locally. Nothing uploaded.')
