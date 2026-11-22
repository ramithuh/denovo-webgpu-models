#!/usr/bin/env python3
from reference import *
from dump_oracle import inputs
import json,time,numpy as np
matrix=json.loads((ROOT/'fixtures/mode_matrix.json').read_text());original=model();reference={};timings={}
for case in matrix['cases']:
 key=(case['mode'],case['selfcond'])
 if key not in reference:
  args=inputs(case['length'],case['mode'],selfcond=case['selfcond']);start=time.perf_counter()
  with torch.no_grad():value=original(**args)[2][0].numpy()
  reference[key]=value;timings[str(key)]=(time.perf_counter()-start)*1000
rows=[]
for case in matrix['cases']:
 meta=json.loads((ROOT/'fixtures'/f"oracle_{case['name']}.json").read_text());v=meta['traces']['px0'];actual=np.fromfile(ROOT/'fixtures'/meta['binary'],dtype='<f4',offset=v['offset'],count=v['length']).reshape(-1,3,3);expected=reference[(case['mode'],case['selfcond'])];delta=actual.astype('f8')-expected.astype('f8')
 rows.append({**case,'relRMS':float(np.sqrt(np.sum(delta**2)/np.sum(expected.astype('f8')**2))),'caRmsdAngstrom':float(np.sqrt(np.mean(np.sum(delta[:,1]**2,axis=-1))))})
 if case['precision']=='fp32':assert np.array_equal(actual,expected), 'FP32 checkpoint round trip changed final coordinates'
(ROOT/'fixtures/export_metrics.json').write_text(json.dumps(dict(reference='Direct original Base_ckpt.pt model_state_dict forward, separately executed',originalForwardMs=timings,coordinates=rows),indent=2)+'\n');print(rows)
