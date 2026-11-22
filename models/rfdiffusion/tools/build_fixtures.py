#!/usr/bin/env python3
"""Regenerate every release fixture. Subprocess failures are fatal."""
import pathlib,subprocess,sys,json,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
cases=[];trajectories=[]
for precision in ['fp32','fp16','int8']:
 for mode in ['unconditional','motif','symmetric']:
  for selfcond in ([False,True] if mode=='unconditional' else [False]):
   args=[sys.executable,str(ROOT/'tools/dump_oracle.py'),'--precision',precision,'--mode',mode,'--length','24']+(['--selfcond'] if selfcond else [])
   subprocess.run(args,check=True)
   name=f'{precision}_{mode}_L24'+('_selfcond' if selfcond else '')
   cases.append(dict(name=name,precision=precision,mode=mode,length=24,selfcond=selfcond))
  subprocess.run([sys.executable,str(ROOT/'tools/dump_trajectory.py'),'--precision',precision,'--mode',mode],check=True)
  trajectories.append(f'trajectory_{precision}_{mode}_L24')
(ROOT/'fixtures/mode_matrix.json').write_text(json.dumps(dict(cases=cases,trajectories=trajectories),indent=2)+'\n')
rows=[]
def px0(name):
 meta=json.loads((ROOT/'fixtures'/f'oracle_{name}.json').read_text());v=meta['traces']['px0'];return np.fromfile(ROOT/'fixtures'/meta['binary'],dtype='<f4',offset=v['offset'],count=v['length']).reshape(-1,3,3)
for case in cases:
 actual=px0(case['name']);reference=px0(case['name'].replace(case['precision'],'fp32',1));delta=actual.astype(np.float64)-reference
 rows.append({**case,'relRMS':float(np.sqrt(np.sum(delta**2)/np.sum(reference.astype(np.float64)**2))),'caRmsdAngstrom':float(np.sqrt(np.mean(np.sum(delta[:,1]**2,axis=-1))))})
(ROOT/'fixtures/export_metrics.json').write_text(json.dumps(dict(reference='Original unquantized Base_ckpt.pt model_state_dict; FP32 export tensor reload is bit-exact',coordinates=rows),indent=2)+'\n')

subprocess.run([sys.executable,str(ROOT/'tools/verify_roundtrip.py')],check=True)

for precision in ['fp32','fp16','int8']:
 subprocess.run([sys.executable,str(ROOT/'tools/dump_oracle.py'),'--precision',precision,'--length','64'],check=True)

subprocess.run([sys.executable,str(ROOT/'tools/quantization_trajectories.py')],check=True)
