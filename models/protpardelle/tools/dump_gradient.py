import json
from environment import ROOT
from dump_oracle import inputs,save_fixture
from export_protpardelle import config_for,reload_bundle
from protpardelle.core.models import CoordinateDenoiser
from protpardelle.utils import dict_to_namespace
import torch
torch.set_num_threads(8)
for variant in ['cc58_epoch416','cc78_epoch1431','cc91_epoch383']:
    config=config_for(variant);model=CoordinateDenoiser(dict_to_namespace(config)).cpu().eval();cases=[]
    for precision in ['fp32','fp16','int8']:
        model.load_state_dict({k.removeprefix('struct_model.'):v for k,v in reload_bundle(ROOT/'weights'/variant/precision).items() if k.startswith('struct_model.')},strict=True)
        data=inputs(config,'allatom_motif',8);data['noisy_coords'].requires_grad_(True)
        mask=torch.zeros(1,8,37);mask[:,2:5,[0,1,2,4]]=1
        target=data['struct_crop_cond'].clone()
        with torch.enable_grad():
            out=model(**data);loss=((out-target).square()*mask[...,None]).sum();grad=torch.autograd.grad(loss,data['noisy_coords'])[0]
        cases.append({'precision':precision,'mode':'reconstruction_gradient','length':8,'inputs':data|{'target':target,'loss_mask':mask},'traces':{'gradient':grad,'denoised_coords':out}})
    save_fixture('gradient_'+variant,config,cases)
    print(variant,flush=True)
