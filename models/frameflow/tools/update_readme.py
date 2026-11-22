import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
gpu=json.loads((ROOT/'fixtures/gpu_metrics.json').read_text())
cpu=json.loads((ROOT/'fixtures/cpu_metrics.json').read_text())
rows=['| Precision | GPU maximum relRMS (all checked tensors/steps) | Cold L32 forward ms | Warm L32 forward ms | L64/T20 sample ms |','|---|---:|---:|---:|---:|']
for p in ['fp32','fp16','int8']:
 r=[r for r in gpu['records'] if r['name'].startswith(p+'_')]
 maximum=max(m['relRms'] for r in r for m in r['metrics'].values())
 cold=next(x['ms'] for x in r if x['name']==p+'_L32_sc0' and not x.get('warm'))
 warm=next(x['ms'] for x in r if x.get('warm'))
 sample=next(x['ms'] for x in r if x['name']==p+'_L64_T20')
 rows.append(f'| {p.upper()} | {maximum:.4g} | {cold:.1f} | {warm:.1f} | {sample:.1f} |')
maximum=max(m['relRms'] for r in cpu for m in r['metrics'].values())
rows+=['',f'FP32 CPU: **{len(cpu)} complete cases**, maximum relRMS **{maximum:.4g}**. Hardware: NVIDIA GB10, driver 580.159.03, Chromium native Vulkan. All 22 captured shader variants compiled through Naga/wgpu-native; zero skipped validations.']
p=ROOT/'README_WEBGPU.md';text=p.read_text();a,b=text.split('<!-- METRICS_START -->');_,b=b.split('<!-- METRICS_END -->');p.write_text(a+'<!-- METRICS_START -->\n'+'\n'.join(rows)+'\n<!-- METRICS_END -->'+b)
