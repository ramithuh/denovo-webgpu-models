import json
from environment import ROOT
import export_protpardelle as exporter
import torch
origin=ROOT/'.cache/multipart';origin.mkdir(parents=True,exist_ok=True)
config=exporter.config_for('cc58_epoch416')
exporter.ROOT=origin;exporter.CAP=131072
state={'example.weight':torch.arange(512*256,dtype=torch.float32).reshape(512,256)/10000}
for precision in ['fp32','fp16','int8']:
    exporter.export('fixture',state,config,precision)
    m=json.loads((origin/'weights/fixture'/precision/'manifest.json').read_text())
    assert len(m['tensors']['example.weight']['parts'])>1
    for part in m['tensors']['example.weight']['parts']:
        if precision=='int8':assert part['scale_tensor']
