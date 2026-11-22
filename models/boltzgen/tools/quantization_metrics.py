from environment import ROOT
import json
import torch
from safetensors.torch import load_file

def metrics(a,b):
    a,b=a.double(),b.double()
    return {'relRMS':float(torch.sqrt((a-b).square().sum()/b.square().sum())), 'maxAbs':float((a-b).abs().max())}

records=[]
for case in ['unconditional_l8_s3','unconditional_l24_s5','binder_l8_t12_s3']:
    prefix=ROOT / f'weights/dynamic/diverse_fp32_{case}'
    original=load_file(str(prefix)+'.outputs.safetensors')
    mask=load_file(str(prefix)+'.inputs.safetensors')['f:atom_pad_mask'].bool()[0]
    for precision in ['fp16','int8']:
        quantized=load_file(ROOT / f'weights/dynamic/diverse_{precision}_{case}.outputs.safetensors')
        record={'precision':precision,'case':case,'comparison':'dequantized weights with FP32 arithmetic versus original checkpoint FP32','tensors':{}}
        for key in original:
            if key=='atom_coords' or key.startswith('denoised.'):
                record['tensors'][key]=metrics(quantized[key][0,mask],original[key][0,mask])
        records.append(record)
(ROOT / 'fixtures/quantization_drift.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
