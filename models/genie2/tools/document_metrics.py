"""Render measured tables. Does not change thresholds or relabel failures."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def tables():
    export=json.loads((ROOT/'fixtures/export_metrics.json').read_text());gpu=json.loads((ROOT/'fixtures/gpu_metrics.json').read_text())
    out=['| Precision | Tensors | Parameters | Shards | Bundle bytes | GPU weight bytes | GPU / FP32 |','|---|---:|---:|---:|---:|---:|---:|']
    for p,e in export.items():
        b=gpu['memory'][p]['bytes'];out.append(f'| {p} | {e["num_tensors"]} | {e["total_params"]:,} | {len(e["shards"])} | {e["total_bytes"]:,} | {b:,} | {b/gpu["memory"]["fp32"]["bytes"]:.6f} |')
    out+=['','| Case | Final coordinate relRMS | GPU trajectory ms | Strict result |','|---|---:|---:|---|']
    for r in gpu['records']:out.append(f'| {r["case"]} | {r["final"]["relRms"]:.6g} | {r["elapsedMs"]:.1f} | {"PASS" if r["final"]["relRms"]<1e-4 else "FAIL"} |')
    out+=['','| Precision | Load ms | Cold L24 forward ms | Warm L24 forward ms |','|---|---:|---:|---:|']
    for p,t in gpu.get('timings',{}).items():out.append(f'| {p} | {t["loadMs"]:.1f} | {t["coldForwardMs"]:.1f} | {t["warmForwardMs"]:.1f} |')
    return '\n'.join(out)
if __name__=='__main__':
    p=ROOT/'README_WEBGPU.md';s=p.read_text();a='<!-- METRICS START -->';b='<!-- METRICS END -->'
    s=s.split(a)[0]+a+'\n'+tables()+'\n'+b+s.split(b)[1];p.write_text(s)
