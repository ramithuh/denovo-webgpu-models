"""Render measured tables; never turn missing verification into a pass."""
import json,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
load=lambda p:json.loads((ROOT/p).read_text())
gpu=load('fixtures/gpu_metrics.json')['metrics'];cpu=load('fixtures/cpu_metrics.json');quality=load('fixtures/export_metrics.json');cases=load('fixtures/cases.json')
readme=(ROOT/'README_WEBGPU.md').read_text()
def update(tag,text):
 global readme
 pattern=r'<!-- '+tag+r' -->(?:\n.*?<!-- /'+tag+r' -->)?'
 readme=re.sub(pattern,'<!-- '+tag+' -->\n'+text+'\n<!-- /'+tag+' -->',readme,flags=re.S)
rows=['| Variant | Precision | Shards | Bytes | Weight relRMS |','| --- | --- | ---: | ---: | ---: |']
for v in ['antifold','esmif1']:
 for p in ['fp32','fp16','int8']:
  m=load(f'weights/{v}/{p}/manifest.json');rows.append(f"| {v} | {p} | {len(m['shards'])} | {m['total_bytes']:,} | {m['weight_relRMS']:.6g} |")
update('EXPORT_TABLE','\n'.join(rows))
rows=['| Variant | Precision | Worst logit relRMS vs original FP32 (all modes) |','| --- | --- | ---: |']
for v in ['antifold','esmif1']:
 for p in ['fp32','fp16','int8']:
  q=max(quality[f"{c['name']}/{p}"]['logits_relRMS'] for c in cases if c['variant']==v);rows.append(f'| {v} | {p} | {q:.6g} |')
update('QUALITY_TABLE','\n'.join(rows))
rows=['| Case | Model | Residues incl. linkers | CPU FP32 worst trace relRMS | GPU worst trace relRMS (all precisions) | Replay |','| --- | --- | ---: | ---: | ---: | --- |']
for c in cases:
 n=c['name'];assert n in cpu,n
 cr=max(v.get('relRMS',0) for v in cpu[n]['traces'].values());gr=max(v.get('relRMS',0) for p in ['fp32','fp16','int8'] for v in gpu[f'{n}/{p}']['traces'].values());assert cr<1e-4 and gr<1e-4
 replay='Exact AR + CDR' if cpu[n].get('independentReplay') else 'Exact AR'
 rows.append(f"| {n} | {c['variant']} | {len(c['sequence'])} | {cr:.3g} | {gr:.3g} | {replay} |")
update('MODE_TABLE','\n'.join(rows))
rows=['| Model / input | Precision | Cold weight load (ms) | Warm median (ms) |','| --- | --- | ---: | ---: |']
for c in cases:
 if not c.get('benchmark'):continue
 for p in ['fp32','fp16','int8']:
  m=gpu[f"{c['name']}/{p}"];rows.append(f"| {c['variant']} / {c['name']} ({len(c['sequence'])}) | {p} | {m['coldLoadMs']:.1f} | {statistics.median(m['warmMs']):.1f} |")
update('TIMING_TABLE','\n'.join(rows))
(ROOT/'README_WEBGPU.md').write_text(readme)
