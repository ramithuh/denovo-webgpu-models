#!/usr/bin/env python3
"""Render measured, successful validation artifacts; missing metrics are fatal."""
import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/'fixtures'/n).read_text())
cpu={r['name']:r for r in read('cpu_metrics.json')};gpu={r['name']:r for r in read('gpu_metrics.json')['records']};matrix=read('mode_matrix.json');exports=read('export_metrics.json');bench=read('benchmarks.json')
def f(x):return f'{x:.3e}'
def maximum(r,key):return max(v['relRMS'] for v in r['metrics'] if v['key']==key)
lines=['## Measured validation','', 'These measurements compare actual learned outputs against the pinned upstream PyTorch/DGL runner. All coordinate errors use the original coordinate frame, without fitting a rigid transform. The mode matrix uses L=24, with additional unconditional L=64 checks for all precisions; each trajectory contains t=50, 49, 48. Quantized oracles use the identical exported weights dequantized to F32.','', '| Precision / mode | CPU final xyz relRMS | GPU final xyz relRMS | GPU N/CA/C/O relRMS | Worst GPU block trace |','| --- | ---: | ---: | ---: | ---: |']
for c in matrix['cases']:
 name=c['name'];a=cpu[name];b=gpu[name];mode=c['mode']+(' (self-conditioned)' if c['selfcond'] else '')
 lines.append(f"| {c['precision']} / {mode} | {f(a['px0']['relRMS'])} | {f(b['px0']['relRMS'])} | {f(b['backbone']['relRMS'])} | {f(b['worst']['relRMS'])} |")
lines+=['','| Three-step trajectory | CPU px0 relRMS | GPU px0 relRMS | GPU raw xyz relRMS | GPU reverse N/CA/C relRMS | GPU rotation relRMS |','| --- | ---: | ---: | ---: | ---: | ---: |']
for name in matrix['trajectories']:
 a=cpu[name];b=gpu[name];lines.append(f"| {name.removeprefix('trajectory_').removesuffix('_L24')} | {f(maximum(a,'px0'))} | {f(maximum(b,'px0'))} | {f(maximum(b,'xyz'))} | {f(maximum(b,'coords'))} | {f(maximum(b,'rotation'))} |")
lines+=['','| L=64 precision | CPU final xyz relRMS | GPU final xyz relRMS | GPU N/CA/C/O relRMS | Worst block trace |','| --- | ---: | ---: | ---: | ---: |']
for r in read('length64_metrics.json'):
 lines.append(f"| {r['name'].split('_')[0]} | {f(cpu[r['name']]['px0']['relRMS']) if r['name'] in cpu else '—'} | {f(r['px0']['relRMS'])} | {f(r['backbone']['relRMS'])} | {f(r['worst']['relRMS'])} |")
lines+=['','| Export precision | Worst unaligned Cα RMSD vs original FP32 (Å) |','| --- | ---: |']
for p in ['fp32','fp16','int8']:lines.append(f"| {p} | {max(r['caRmsdAngstrom'] for r in exports['coordinates'] if r['precision']==p):.9f} |")
lines+=['','| Third-step estimate / mode | FP16 Cα RMSD vs FP32 (Å) | INT8 Cα RMSD vs FP32 (Å) |','| --- | ---: | ---: |']
quant=read('trajectory_quantization_metrics.json')['coordinates']
for mode in ['unconditional','motif','symmetric']:
 values=[next(r['caRmsdAngstrom'] for r in quant if r['mode']==mode and r['precision']==p and r['step']==3) for p in ['fp16','int8']]
 lines.append(f"| {mode} | {values[0]:.9f} | {values[1]:.9f} |")
lines+=['','These three-step quantization comparisons use the same initial coordinates and recorded Gaussian variates, verified independently. They measure model quantization drift, separately from browser-versus-PyTorch implementation parity.']
lines+=['','| NVIDIA GB10 precision / length | Cold bundle/device load (ms) | Median warm denoiser step (ms), 3 runs |','| --- | ---: | ---: |']
for r in bench:lines.append(f"| {r['precision']} / L={r['length']} | {r['coldLoadMs']:.1f} | {r['medianMs']:.1f} |")
lines+=['','Cold loading uses local HTTP with application caching disabled; it includes hashing, parsing, upload and device creation. Warm timings follow one untimed forward pass and include final-coordinate readback. They exclude optional intermediate trace readback. GPU and CPU timings depend on other activity on this shared host.','', 'Browser contract tests confirm caller-owned GPU device survival after engine destruction, module-worker inference, verified offline Cache API and OPFS reads, and a storage-denied fallback matching Safari Private Mode behavior. This fallback is tested by injecting the browser storage error; Safari itself was not run. Buffer checks require zero transient allocations after normal inference, an injected exception and an AbortSignal, and zero tracked bytes after idempotent destruction.','']
validation=read('validation.json')
lines+=['| Final validation command | Passed | Failed / skipped | Wall time (s) |','| --- | ---: | ---: | ---: |']
for r in validation['commands']:lines.append(f"| `{r['command']}` | {r['pass']} | {r['fail']} / {r['skipped']} | {r['durationMs']/1000:.3f} |")
lines+=['',f"Recorded {validation['generatedAt']}. Runtime source SHA-256: `{validation['runtimeSourceSha256']}`. The CPU suite uses independent Node workers for fixture batches; runtime modules themselves import no Node APIs. Full logs are saved locally under `.cache/final-*.log` and their hashes are in `fixtures/validation.json`.",'']
path=ROOT/'README_WEBGPU.md' ;text=path.read_text().split('\n<!-- measured-metrics -->')[0];path.write_text(text+'\n<!-- measured-metrics -->\n'+'\n'.join(lines)+'\n');print('Rendered measured mode matrix, quantization errors and timings')
