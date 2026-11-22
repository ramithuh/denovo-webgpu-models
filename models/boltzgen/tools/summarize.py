"""Collect measured dynamic parity, memory and browser contracts without changing documentation."""
from environment import ROOT
import json
cases=['unconditional_l8_s3','unconditional_l24_s5','binder_l8_t12_s3']
records=[]
for variant,precision in [('diverse','fp32'),('diverse','fp16'),('diverse','int8'),('adherence','fp32')]:
    for case in cases+(['motif_l16_s5_r1'] if (variant,precision)==('diverse','fp16') else []):
        cpu=json.loads((ROOT/f'fixtures/cpu_{variant}_{precision}_{case}.json').read_text())
        gpu=json.loads((ROOT/f'fixtures/gpu_{variant}_{precision}_{case}.json').read_text())
        oracle=json.loads((ROOT/f'fixtures/dynamic_oracle_{variant}_{precision}_{case}.json').read_text())
        assert all(v['relRMS']<1e-4 for result in [cpu,gpu] for v in result['stats'].values())
        assert gpu['memory']['transientBytes']==0
        records.append({'case':case,'variant':variant,'precision':precision,'oracle':oracle,'cpu':cpu,'gpu':gpu})
validation={'status':'dynamic coordinate parity validated','threshold':1e-4,'alignment_fitting':False,'cases':records,'browser_contracts':json.loads((ROOT/'fixtures/browser_contracts.json').read_text())}
(ROOT/'fixtures/validation.json').write_text(json.dumps(validation,indent=2)+'\n')
for r in records:print(r['variant'],r['precision'],r['case'],r['cpu']['stats']['atom_coords']['relRMS'],r['gpu']['stats']['atom_coords']['relRMS'],r['gpu']['memory']['persistentBytes'])
