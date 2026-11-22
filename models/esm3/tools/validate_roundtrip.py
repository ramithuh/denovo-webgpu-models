#!/usr/bin/env python3
"""Strict model state reload for every bundle and independent final-output check."""
import json,torch,numpy as np
from upstream import ROOT,load_models,restore
from export_esm3 import load_bundle
from dump_oracle import inputs,stats
models=load_models();results={}
with torch.inference_mode():
 for precision in ['fp32','fp16','int8']:
    state=load_bundle(ROOT/'weights'/precision);restore(models,state);del state
    out=models['esm'](**inputs()['sequence']);m=json.loads((ROOT/'fixtures'/f'oracle-{precision}.json').read_text());metrics={}
    with (ROOT/'weights'/f'oracle-{precision}.bin').open('rb') as f:
        for field in ['sequence_logits','structure_logits','secondary_structure_logits','sasa_logits','function_logits','residue_logits','embeddings']:
            key='ss8_logits'if field=='secondary_structure_logits'else field;rec=m['tensors']['sequence/'+key];f.seek(rec['offset']);ref=torch.from_numpy(np.frombuffer(f.read(rec['bytes']),dtype='<f4').copy()).reshape(rec['shape']);metrics[key]=stats(getattr(out,field),ref);assert metrics[key]['relRms']<1e-4
    results[precision]=dict(strict_load=True,final_outputs=metrics);print(precision,'strict reload passed',flush=True)
(ROOT/'fixtures/roundtrip.json').write_text(json.dumps(results,indent=2)+'\n')
