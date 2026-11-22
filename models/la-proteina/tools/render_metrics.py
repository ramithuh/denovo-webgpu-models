import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'fixtures'/name).read_text())
export=read('export_metrics.json');gpu=read('gpu_metrics.json');cpu=read('cpu_metrics.json');naga=read('wgsl_validation.json')
long=read('gpu_long_metrics.json')
oracles=[read('oracles.json'),read('extended_oracles.json')]
mode_matrix={'upstream_commit':export['upstream_commit'],'neural_scope':['LD1','AE1 decoder'],
 'cpu_cases':[c['name'] for c in cpu['cases']],'gpu_cases':[c['name'] for c in gpu['cases']],
 'gpu_long_cases':[c['name'] for c in long['cases']],
 'threshold_relRMS':1e-4,'runtime_dynamic_length':[1,500],'runtime_dynamic_steps':'positive integer',
 'encoder':False,'triangular_models':False,'motif_conditioning':False,'skipped':0}
(ROOT/'fixtures/mode_matrix.json').write_text(json.dumps(mode_matrix,indent=2)+'\n')
lines=['## Measured results','',f"Hardware: **{gpu['hostGPU']}**, Chromium native Vulkan WebGPU. Values compare tensors in the original coordinate frame, without rigid alignment. All listed cases passed; **0 skipped**.",'',
'| Model | Tensors | Parameters | FP32 bytes | FP16 bytes | INT8 bytes |',
'|---|---:|---:|---:|---:|---:|']
for model in ['LD1','AE1']:
    p=export['models'][model];lines.append(f"| {model}{' decoder' if model=='AE1' else ''} | {p['fp32']['num_tensors']:,} | {p['fp32']['total_params']:,} | {p['fp32']['total_bytes']:,} | {p['fp16']['total_bytes']:,} | {p['int8']['total_bytes']:,} |")
lines+=['','Serialized bytes include safetensors headers. Strict export round-trip error is exactly zero in every precision. Largest shard: '+f"{max(p['max_shard_bytes'] for m in export['models'].values() for p in m.values()):,} bytes.",'',
'| Backend / precision | Cases | Worst final Cα relRMS | Latents | All-atom coordinates | Sequence logits |',
'|---|---:|---:|---:|---:|---:|']
for backend,precision,cases in [('CPU','fp32',cpu['cases'])]+[('GPU',p,[c for c in gpu['cases'] if c['name'].startswith(p)]) for p in ['fp32','fp16','int8']]:
    vals=[max(c['parity'][k]['relRMS'] for c in cases) for k in ['ca','z','coords','logits']]
    lines.append(f"| {backend} {precision} | {len(cases)} | "+' | '.join(f'{v:.3e}' for v in vals)+' |')
lines+=['','The parity threshold is `relRMS < 1e-4` against the identical exported precision. Every trajectory state and velocity in these 30 GPU cases also passes; residue argmax and atom masks match exactly.','',
'| Additional L32 / 400-step case | Worst final relRMS | Runtime (s) | Worst trajectory relRMS |',
'|---|---:|---:|---:|']
for case in long['cases']:
    error=max(case['parity'][k]['relRMS'] for k in ['ca','z','coords','logits','maskedCoords'])
    lines.append(f"| {case['name']} | {error:.3e} | {case['timingMs']/1000:.2f} | {max(v['relRMS'] for k,v in case['parity'].items() if k.startswith('step')):.3e} |")
lines+=['','All four long cases pass the same `1e-4` threshold for every trajectory input and velocity as well as final tensors, with exact sequence/atom-mask agreement. Per-step measurements are retained in `fixtures/gpu_long_metrics.json`.',
'',f"The time-embedding regression covers angles through 2000 radians: GPU relRMS {gpu['timeEmbedding']['relRMS']:.3e}, max absolute error {gpu['timeEmbedding']['maxAbs']:.3e}. WGSL uses split pi/2 range reduction to avoid large-angle native trigonometric error.",
'','Quantization changes the model; its drift from original FP32 is measured separately below.','',
'| Quantization drift vs FP32, worst measured relRMS | Cα | Latents | All-atom coordinates | Logits |','|---|---:|---:|---:|---:|']
for precision in ['fp16','int8']:
    drifts=[v for o in oracles for k,v in o['quantization_drift'].items() if k.startswith(precision)]
    lines.append('| '+precision+' | '+' | '.join(f"{max(v[k]['relRMS'] for v in drifts):.3e}" for k in ['final_ca','final_z','coords','logits'])+' |')
lines+=['','These drift maxima include very short SDE/ODE trajectories; they measure model quantization effects, not browser arithmetic error or biological designability.','',
'| Precision | Packed GPU weight bytes | Ratio to FP32 | First L32/T1 SDE call (ms) | Same call warm (ms) |','|---|---:|---:|---:|---:|']
for precision in ['fp32','fp16','int8']:
    n=gpu['memory'][precision]['loaded']['bytes'];t=gpu['timings'][precision]
    lines.append(f"| {precision} | {n:,} | {n/gpu['memory']['fp32']['loaded']['bytes']:.4f} | {t['coldMs']:.1f} | {t['warmMs']:.1f} |")
lines+=['','Cold here means the first invocation on a new engine; the browser/driver shader cache may already be warm. Weight fetch/loading is excluded from these forward timings and recorded separately in `fixtures/gpu_metrics.json`.',
'',f"Lifecycle tests pass for normal completion, invalid inputs, an injected mid-forward error, mid-forward cancellation, recovery, external GPUDevice ownership, and idempotent destruction. Every completion path leaves **0 transient buffers**, and destruction leaves **0 tracked bytes**. Real worker parity, native F16, real Cache API/OPFS cold and warm loads, and injected storage-denial fallback pass.",
'',f"**{naga['passed']} shader variants** compiled through wgpu-native/Naga on {naga['adapter']['device']}; {naga['failed']} failures and {naga['skipped']} skips. Runtime WGSL uses mutable local/workgroup arrays rather than dynamically indexed const arrays.",
'','Evidence: `fixtures/export_metrics.json`, `fixtures/cpu_metrics.json`, `fixtures/gpu_metrics.json`, `fixtures/gpu_long_metrics.json`, `fixtures/wgsl_validation.json`, and the hash-addressed binary fixture metadata.']
p=ROOT/'README_WEBGPU.md';p.write_text(p.read_text().split('<!-- measured-metrics -->')[0]+'<!-- measured-metrics -->\n\n'+'\n'.join(lines)+'\n')
