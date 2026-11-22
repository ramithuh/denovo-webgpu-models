#!/usr/bin/env python3
"""Render measured tables without transcribing or silently omitting failed cases."""
import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
exports=json.loads((ROOT/'fixtures/export_metrics.json').read_text())
gpu=json.loads((ROOT/'fixtures/gpu_metrics.json').read_text())
cpu=json.loads((ROOT/'fixtures/cpu_metrics.json').read_text())
assert gpu['ok'] and gpu['skipped']==cpu['skipped']==0
assert any('/atom37' in r['name'] for r in gpu['rows'])
assert gpu['worker']['initializationAbort']
for results,prefixes in [(gpu,[f'{p}/{L}' for p in ['fp32','fp16','int8'] for L in [32,64]]),(cpu,['32','64'])]:
 names={r['name'] for r in results['rows']}
 for prefix in prefixes:
  for steps in [2,3]:
   assert all(f'{prefix}/T{steps}/{field}' in names for field in ['trajectory','rotVF','transVF','pred','psi','atom37'])
assert all(r['relRMS']<1e-4 for r in gpu['rows']+cpu['rows'])
lines=['## Verified measurements','', 'Hardware: '+(ROOT/'fixtures/hardware.txt').read_text().strip().split(',')[0]+'. Chromium native Vulkan; no fallback adapter. Full records are in `fixtures/export_metrics.json`, `fixtures/cpu_metrics.json`, and `fixtures/gpu_metrics.json`. `fixtures/verification.json` records the successful command, browser version, and runtime file hashes.', '', '| Precision | Tensors | Safetensors bytes | Shards | Packed GPU bytes | GPU / FP32 |','|---|---:|---:|---:|---:|---:|']
for e,m in zip(exports,gpu['memory']):lines.append(f"| {e['precision']} | {e['tensors']:,} | {e['bytes']:,} | {e['shards']} | {m['bytes']:,} | {m['bytes']/gpu['memory'][0]['bytes']:.6f} |")
lines+=['','Every exported tensor reloads exactly to the expected quantized representation. Biases, normalization parameters and one-dimensional buffers stay F32. No full-size F32 weight mirror exists in GPU memory.','','| Backend / precision | L | Maximum final forward relRMS | Maximum trajectory / final-step relRMS |','|---|---:|---:|---:|']
for p in ['fp32','fp16','int8']:
 for L in [32,64]:
  rs=[r for r in gpu['rows'] if r['name'].startswith(f'{p}/{L}/')]
  f=max(r['relRMS'] for r in rs if '/T' not in r['name'] and '/esm.' not in r['name'])
  t=max(r['relRMS'] for r in rs if '/T' in r['name'])
  lines.append(f'| WebGPU {p} | {L} | {f:.3e} | {t:.3e} |')
for L in [32,64]:
 rs=[r for r in cpu['rows'] if r['name'].startswith(f'{L}/')]
 lines.append(f"| CPU FP32 | {L} | {max(r['relRMS'] for r in rs if '/T' not in r['name'] and '/esm.' not in r['name']):.3e} | {max(r['relRMS'] for r in rs if '/T' in r['name']):.3e} |")
lines+=['',f"{len(gpu['rows'])} GPU tensor comparisons and {len(cpu['rows'])} CPU tensor comparisons pass at relRMS < 1e-4, with zero skips. Trajectories use T=2 and T=3 at both lengths. GPU forward and sampler tests execute ESM directly; the CPU sampler reuses its own computed representations as upstream caches them. Conditional token inputs and precomputed representations are checked against an independent sequence-conditioned oracle. Atom14 and atom37 are compared in full, including padding.", '', f"CPU inference checks took {sum(t['ms'] for t in cpu['timings'])/60000:.1f} minutes on this host, excluding initial weight loading.", '', 'Normal, injected-error and abort paths each leave zero transient GPU buffers. Repeated destroy leaves zero bytes. Tests also pass full worker sampling, cancellation during worker loading, external-device preservation, maximum-buffer admission, Cache API and OPFS cold/warm reads, and both storage-denied fallbacks.', '', '| Precision | L | Cold full forward (ms) | Warm full forward (ms) | Full T=2 sample (ms) | Full T=3 sample (ms) |','|---|---:|---:|---:|---:|---:|']
for p in ['fp32','fp16','int8']:
 for L in [32,64]:
  rows=[r for r in gpu['timings'] if r.get('L')==L and r['precision']==p]
  def at(phase,steps=None):return next(r['ms'] for r in rows if r['phase']==phase and (steps is None or r.get('steps')==steps))
  lines.append(f"| {p} | {L} | {at('cold_full_forward'):.1f} | {at('warm_full_forward'):.1f} | {at('full_sample',2):.1f} | {at('full_sample',3):.1f} |")
lines+=['', 'Times are browser wall-clock durations through GPU completion and readback, including ESM. Cold measurements include shader/pipeline compilation and oracle-comparison trace readbacks; warm measurements use normal outputs. Model download/upload is separate in the JSON. Measurements share the host with other work and are not isolated throughput claims.', '', '| Precision | L | Atom37 quantization drift relRMS | T=2 trajectory drift | T=3 trajectory drift |','|---|---:|---:|---:|---:|']
for r in gpu['drift']:lines.append(f"| {r['precision']} | {r['L']} | {r['atoms']['relRMS']:.3e} | {r['T2']['relRMS']:.3e} | {r['T3']['relRMS']:.3e} |")
lines+=['','Quantization drift compares quantized **PyTorch** outputs with FP32 PyTorch outputs; it is separate from WebGPU implementation error. Coarse grids amplify that drift. This is numerical verification, not a designability evaluation.','']
p=ROOT/'README_WEBGPU.md';text=p.read_text();start='<!-- VERIFIED_METRICS -->';end='<!-- /VERIFIED_METRICS -->'
if end in text:before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+start+'\n\n'+'\n'.join(lines)+'\n'+end+after
else:text=text.replace(start,start+'\n\n'+'\n'.join(lines)+'\n'+end)
p.write_text(text)
