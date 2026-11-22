from environment import ROOT, load_model
from export_boltzgen import reload_bundle
import json
import torch

torch.set_num_threads(4)
model,_=load_model()
records=[]
for variant in ['diverse','adherence']:
    for precision in ['fp32','fp16','int8']:
        reload_bundle(model,variant,precision)
        assert all(torch.isfinite(t).all() for t in model.state_dict().values())
        records.append({'variant':variant,'precision':precision,'strict_load_state_dict':True,'all_finite':True})
        print(variant,precision,'strict reload passed',flush=True)
(ROOT/'fixtures/roundtrip.json').write_text(json.dumps(records,indent=2)+'\n')
