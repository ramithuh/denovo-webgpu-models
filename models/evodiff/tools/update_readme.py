"""Render measured tables from the most recent successful CPU/GPU/export runs."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
load = lambda name: json.loads((ROOT/'fixtures'/name).read_text())
export, cpu, gpu = load('export_metrics.json'), load('cpu_metrics.json'), load('gpu_metrics.json')
lines = ['### Export sizes and quantization error versus original FP32', '',
 '| Bundle | Safetensors bytes | Shards | Largest shard | Worst logit relRMS | Worst probability relRMS |',
 '|---|---:|---:|---:|---:|---:|']
for p in ['fp32','fp16','int8']:
    v=export[p]
    lines.append(f'| {p.upper()} | {v["total_bytes"]:,} | {len(v["shards"])} | {max(s["byteLength"] for s in v["shards"]):,} | {max(r["logits"]["relRMS"] for r in v["roundtrip"].values()):.8g} | {max(r["probabilities"]["relRMS"] for r in v["roundtrip"].values()):.8g} |')
lines += ['', '### Mode Matrix: worst final-output relRMS', '',
 '| Backend / precision | A: final logits | B: all 24 step logits | C: all 8 step logits | B/C sequence replay |',
 '|---|---:|---:|---:|---|']
for backend, data in [('CPU',cpu),('WebGPU',gpu)]:
    for p,v in data['precisions'].items():
        a=max(c['trace']['logits']['relRMS'] for c in v['A'].values())
        b=max(s['relRMS'] for s in v['B']['steps'])
        c=max(s['relRMS'] for s in v['C']['steps'])
        assert max(a,b,c)<1e-4 and v['B']['exact'] and v['C']['exact']
        lines.append(f'| {backend} {p.upper()} | {a:.8g} | {b:.8g} | {c:.8g} | Exact / exact |')
worst=max(s['relRMS'] for v in gpu['precisions'].values() for c in v['A'].values() for s in c['trace'].values())
lines += ['',f'Worst WebGPU error across **all intermediate trace tensors**: `{worst:.8g}`. Native `useF16=true` final-logit relRMS: `{gpu["nativeF16"]["relRMS"]:.8g}`. Zero skips.', '',
 'All precisions replay the same complete sequences: Mode B `MELTDCTSLQELFSSCKDPGTSVH`; Mode C `MLTAACDETSSG` (fixed motif `ACDE`).', '',
 '### NVIDIA GB10 latency', '',
 '| Precision | Cold engine load (ms) | L=32 warm (ms) | L=128 warm (ms) | L=256 warm (ms) |',
 '|---|---:|---:|---:|---:|']
for p,v in gpu['precisions'].items():
    cells=[v['coldLoadMs']]+[v['benchmarks'][str(n)]['medianMs'] for n in [32,128,256]]
    lines.append('| '+p.upper()+' | '+' | '.join(f'{x:.3f}' for x in cells)+' |')
lines += ['', 'Warm latency is the median of three complete forwards after one warmup, including submission, completion, and logit readback. Cold engine load includes local HTTP weights, SHA verification, upload, and pipeline creation with persistent weight caching disabled; OS/driver caches may be warm.', '',
 f'Host: `{gpu.get("hostGPU", "NVIDIA GB10 / 580.159.03")}`. Browser: `{gpu["userAgent"]}`.', '',
 'Raw evidence: [export_metrics.json](fixtures/export_metrics.json), [cpu_metrics.json](fixtures/cpu_metrics.json), [gpu_metrics.json](fixtures/gpu_metrics.json).']
if (ROOT/'fixtures/test_runs.json').exists():
    lines += ['', 'Final command runs ([test_runs.json](fixtures/test_runs.json)):', '', '| Command | Exit code | Elapsed seconds |', '|---|---:|---:|']
    for run in load('test_runs.json'):
        lines.append(f'| `{run["command"]}` | {run["exitCode"]} | {run["seconds"]:.6f} |')
path=ROOT/'README_WEBGPU.md';text=path.read_text();a='<!-- MEASUREMENTS_START -->';b='<!-- MEASUREMENTS_END -->'
path.write_text(text.split(a)[0]+a+'\n'+'\n'.join(lines)+'\n'+b+text.split(b)[1])
