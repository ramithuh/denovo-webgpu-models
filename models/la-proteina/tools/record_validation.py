"""Record reproducible source/evidence hashes after running every validation command."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(name): return json.loads((ROOT/'fixtures'/name).read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

cpu=read('cpu_metrics.json');gpu=read('gpu_metrics.json');long=read('gpu_long_metrics.json')
naga=read('wgsl_validation.json');export=read('export_metrics.json')
assert long['fullTrajectory']
assert len(cpu['cases'])==8 and len(gpu['cases'])==30 and len(long['cases'])==4
for suite in [cpu,gpu,long]:
    assert suite['skipped']==0
    for case in suite['cases']:
        for value in case['parity'].values(): assert value['relRMS']<1e-4
assert naga['failed']==0 and naga['skipped']==0
assert gpu['timeEmbedding']['relRMS']<1e-6
assert gpu['linear']['relRMS']<1e-7
for model in export['models'].values():
    for precision in model.values():
        assert precision['strict_load'] and precision['roundtrip_maxAbs']==0
        assert precision['max_shard_bytes']<=25_000_000
for precision in ['fp32','fp16','int8']:
    assert gpu['lifecycle'][precision]['transientBuffers']==0

sources={}
for directory in ['shared','cpu','webgpu','test','tools','demo']:
    for path in sorted((ROOT/directory).iterdir()):
        if path.is_file(): sources[str(path.relative_to(ROOT))]=digest(path)
for name in ['package.json','sync-upstream.mjs','README_WEBGPU.md','LICENSE','NOTICE','WEIGHTS_LICENSE','THIRD_PARTY_LICENSES']:
    sources[name]=digest(ROOT/name)
evidence={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'fixtures').glob('*.json'))
          if p.name not in ['validation.json','launchpad_report.json']}
runtime={k:v for k,v in sources.items() if k.split('/')[0] in ['shared','cpu','webgpu']}
result={'recorded_at':datetime.now(timezone.utc).isoformat(),'upstream_commit':export['upstream_commit'],
 'host_gpu':gpu['hostGPU'],'threshold_relRMS':1e-4,'skipped':0,
 'cpu_cases_passed':len(cpu['cases']),'gpu_cases_passed':len(gpu['cases']),
 'gpu_400_step_cases_passed':len(long['cases']),'naga_variants_passed':naga['passed'],
 'runtime_sha256':hashlib.sha256(json.dumps(runtime,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'commands':['npm test','npm run test:gpu:long','XDG_CACHE_HOME="$PWD/.cache/wgpu" .venv/bin/python tools/validate_wgsl.py'],
 'log_sha256':{name:digest(ROOT/name) for name in ['.cache/final-npm-test.log','.cache/final-gpu-long.log','.cache/final-naga.log']},
 'source_sha256':sources,'evidence_sha256':evidence}
(ROOT/'fixtures/validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','evidence_sha256']},indent=2))
