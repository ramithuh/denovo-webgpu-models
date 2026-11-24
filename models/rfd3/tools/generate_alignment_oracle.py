#!/usr/bin/env python3
import json
import torch
from checkpoint import ROOT
from rfd3._core.utils.alignment import weighted_rigid_align
from rfd3.model.inference_sampler import centre_random_augment_around_motif
from rfd3.inference.symmetry.symmetry_utils import apply_symmetry_to_xyz_atomwise
from unittest.mock import patch
from generate_oracle import tensor_json

torch.manual_seed(412);records=[]
for kind in ['full','planar','reflection','weighted']:
    moving=torch.randn(1,9,3);target=torch.randn(1,9,3)
    if kind=='planar':moving[...,2]=0;target[...,2]=0
    if kind=='reflection':target=moving.clone();target[...,0]*=-1
    mask=torch.tensor([True]*6+[False]*3);weights=torch.arange(1,10).float() if kind=='weighted' else torch.ones(9)
    output=weighted_rigid_align(target,moving,X_exists_L=mask,w_L=weights)
    records.append(dict(name=kind,moving=tensor_json(moving),target=tensor_json(target),mask=tensor_json(mask),weights=tensor_json(weights),output=tensor_json(output)))
angles=torch.tensor([[0.3,1.4,2.1]]);translation=torch.tensor([[[0.2,-0.7,1.8]]])
x=torch.randn(1,9,3);base=torch.randn(1,9,3);fixed=torch.tensor([True]*5+[False]*4)
with patch('torch.rand', side_effect=[angles[:,i]/(2*torch.pi) for i in range(3)]),patch('torch.normal',return_value=translation):
    augmented,R=centre_random_augment_around_motif(x.clone(),base,fixed,center_option='motif')
sym={'sym_entity_id':torch.tensor([0]*12+[-1]),'sym_transform_id':torch.tensor([0]*4+[1]*4+[2]*4+[-1]),'is_sym_asu':torch.tensor([True]*4+[False]*9),'sym_transform':{str(i):(torch.tensor([[torch.cos(torch.tensor(i*2*torch.pi/3)),-torch.sin(torch.tensor(i*2*torch.pi/3)),0],[torch.sin(torch.tensor(i*2*torch.pi/3)),torch.cos(torch.tensor(i*2*torch.pi/3)),0],[0,0,1]]),torch.tensor([0.,0.,0.])) for i in range(3)}}
sx=torch.randn(1,13,3);sy=apply_symmetry_to_xyz_atomwise(sx.clone(),sym)
result=dict(alignment=records,augmentation=dict(x=tensor_json(x),base=tensor_json(base),fixed=tensor_json(fixed),angles=tensor_json(angles),translation=tensor_json(translation),output=tensor_json(augmented)),symmetry=dict(x=tensor_json(sx),output=tensor_json(sy),features={k:({j:[r.flatten().tolist(),t.tolist()] for j,(r,t) in v.items()} if k=='sym_transform' else v.tolist()) for k,v in sym.items()}))
(ROOT/'fixtures/alignment-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
