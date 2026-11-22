"""Deterministic little-endian tensors from the pinned network AND pinned sampler."""
import json, hashlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRotation
from omegaconf import OmegaConf
from oracle_common import ROOT,COMMIT,model,reload_bundle,all_atom,stats
from data.interpolant import Interpolant
OUT=ROOT/'.cache/fixtures';OUT.mkdir(parents=True,exist_ok=True)

def write(name,inputs,outputs,meta):
    arrays={**{'input.'+k:v for k,v in inputs.items()},**{'output.'+k:v for k,v in outputs.items()}}
    blob=bytearray();records={}
    for k,v in arrays.items():
        if isinstance(v,torch.Tensor):v=v.detach().cpu().numpy()
        a=np.asarray(v,dtype='<f4');raw=a.tobytes();records[k]={'shape':list(a.shape),'offset':len(blob),'count':a.size};blob.extend(raw)
    (OUT/(name+'.bin')).write_bytes(blob)
    record={'name':name,**meta,'file':'.cache/fixtures/'+name+'.bin','sha256':hashlib.sha256(blob).hexdigest(),'tensors':records}
    (ROOT/'fixtures'/f'{name}.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

def inputs(L):
    rng=np.random.default_rng(123);t=torch.from_numpy(rng.normal(size=(1,L,3)).astype('float32'))*10;t-=t.mean(dim=1,keepdim=True)
    r=torch.from_numpy(SciRotation.random(L,random_state=rng).as_matrix().astype('float32'))[None]
    return {'trans_t':t,'rotmats_t':r,'res_mask':torch.ones(1,L),'diffuse_mask':torch.ones(1,L),'res_idx':torch.arange(L,dtype=torch.float32)[None],'so3_t':torch.tensor([[0.37]]),'r3_t':torch.tensor([[0.37]])}

def coords(out):return all_atom.to_atom37(out['pred_trans'],out['pred_rotmats'])[...,[0,1,2,4,3],:]

def main(steps_list=(2,20), prefix=""):
    cases=[];drift={};fp32={}
    for precision in ['fp32','fp16','int8']:
        net,cfg=model(reload_bundle(precision))
        with torch.inference_mode():
            for L in [32,64]:
                x=inputs(L)
                for sc in [False,True]:
                    inp=dict(x)
                    if sc:inp['trans_sc']=x['trans_t']*0.7+0.3
                    out=net(inp);out['coords']=coords(out)
                    key=f'L{L}_sc{int(sc)}';name=f'{prefix}{precision}_{key}'
                    cases.append(write(name,inp,out,{'precision':precision,'length':L,'kind':'forward','selfCondition':sc})['name'])
                    if precision=='fp32':fp32[key]={k:v.clone() for k,v in out.items()}
                    else:drift[name]={k:stats(v,fp32[key][k]) for k,v in out.items()}
                for steps in steps_list:
                    cfg_i=OmegaConf.create({'min_t':0.01,'rots':{'corrupt':True,'sample_schedule':'exp','exp_rate':10},'trans':{'corrupt':True,'sample_schedule':'linear'},'sampling':{'num_timesteps':steps},'self_condition':True,'twisting':{'use':False}})
                    sampler=Interpolant(cfg_i);sampler.set_device('cpu');preds=[]
                    def wrapped(batch):
                        out=net(batch);preds.append({k:v.clone() for k,v in out.items()});return out
                    atom_traj,clean_atoms,clean_traj=sampler.sample(1,L,wrapped,trans_0=x['trans_t'],rotmats_0=x['rotmats_t'])
                    outputs={**preds[-1],'coords':atom_traj[-1][...,[0,1,2,4,3],:]}
                    for i,p in enumerate(preds):outputs.update({f'step{i}.{k}':v for k,v in p.items()})
                    key=f'L{L}_T{steps}';name=f'{prefix}{precision}_{key}'
                    cases.append(write(name,x,outputs,{'precision':precision,'length':L,'steps':steps,'kind':'sample','selfCondition':True,'seed':123})['name'])
                    if precision=='fp32':fp32[key]={k:v.clone() for k,v in outputs.items()}
                    else:drift[name]={k:stats(v,fp32[key][k]) for k,v in outputs.items() if '.' not in k}
                print(precision,L,flush=True)
    (ROOT/'fixtures'/(prefix+'matrix.json')).write_text(json.dumps({'commit':COMMIT,'cases':cases,'quantizationDrift':drift},indent=2)+'\n')
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--steps',nargs='+',type=int,default=[2,20]);parser.add_argument('--prefix',default='')
    args=parser.parse_args();main(args.steps,args.prefix)
