#!/usr/bin/env python3
"""Compare full-model weight quantization, including a targeted INT8 head ablation."""
import copy,json
import torch
from checkpoint import ROOT,load_checkpoint
from generate_oracle import tensor_json
from export_weights import atomic_json
from rfd3.model.layers.layer_utils import EmbeddingLayer

torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
checkpoint,net,config=load_checkpoint()
for module in net.modules():
    if hasattr(module,'force_bfloat16'):module.force_bfloat16=False
    if hasattr(module,'use_deepspeed_evo'):module.use_deepspeed_evo=False
net=net.cuda();original={n:m.weight.detach().clone() for n,m in net.named_modules() if isinstance(m,torch.nn.Linear) and not isinstance(m,EmbeddingLayer)}
results={}
with torch.inference_mode():
 for precision in ['fp16','int8','int8-fp32-output']:
  for name,module in net.named_modules():
   if name not in original:continue
   w=original[name]
   if precision=='fp16':q=w.half().float()
   elif precision=='int8-fp32-output' and name in ['diffusion_module.sequence_head.linear','diffusion_module.to_r_update.1']:q=w
   else:
    scale=w.abs().amax(1)/127;scale=torch.where(scale==0,torch.ones_like(scale),scale);q=torch.round(w/scale[:,None]).clamp(-127,127)*scale[:,None]
   module.weight.copy_(q)
  results[precision]={}
  for name in ['monomer','atomic-motif','ligand','dna']:
   directory='oracle' if name in ['monomer','atomic-motif'] else 'oracle-pass2'
   case=json.loads((ROOT/'fixtures'/directory/(name+'.json')).read_text())
   def tensor(v):return torch.tensor(v['data'],device='cuda',dtype=torch.bool if 'bool' in v['dtype'] else torch.long if 'int' in v['dtype'] else torch.float32).reshape(v['shape'])
   f={k:tensor(v) for k,v in case['features'].items()};x=tensor(case['X_noisy_L']);t=tensor(case['t'])
   init=net.token_initializer(copy.deepcopy(f));out=net.diffusion_module(x,t,copy.deepcopy(f),**init,n_recycle=2)
   results[precision][name]={k:tensor_json(v) for k,v in out.items()}
   print(precision,name,flush=True)
atomic_json(ROOT/'fixtures/quantized-oracle.json',results)
