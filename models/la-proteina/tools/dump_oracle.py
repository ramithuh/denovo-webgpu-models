"""Real upstream forward/sampling references, never a captured operation graph."""
import argparse
import hashlib
import json
from unittest.mock import patch
import torch
from oracle_common import *
from openfold.np import residue_constants as rc

def constants():
    obj={'restypes':rc.restypes,'restype_1to3':rc.restype_1to3,'atom_types':rc.atom_types,
      'atom_mask':rc.RESTYPE_ATOM37_MASK.tolist(),'distance_bins':torch.linspace(.1,3,29).tolist()}
    (ROOT/'shared/constants.js').write_text('export default '+json.dumps(obj,separators=(',',':'))+';\n')
    from proteinfoundation.nn.feature_factory import get_time_embedding
    times=torch.tensor([0,.001,.01,.1,.25,.4999,.5,.6,.7,.8,.9,.95,.98,.99,.999,1])
    reference=write_case({'name':'time_embedding'}, {'times':times,'embedding':get_time_embedding(times,256)})
    (ROOT/'fixtures/time_embedding.json').write_text(json.dumps(reference,indent=2)+'\n')
    generator=torch.Generator().manual_seed(94)
    x=torch.randn(32,768,generator=generator);w=torch.randn(768,768,generator=generator);b=torch.randn(768,generator=generator)
    reference=write_case({'name':'linear'}, {'x':x,'w':w,'b':b,'out':torch.nn.functional.linear(x,w,b),'norm':torch.nn.functional.layer_norm(x,[768])})
    (ROOT/'fixtures/linear.json').write_text(json.dumps(reference,indent=2)+'\n')
    centering={}
    for length in [1,4,32,47,64,500]:
        x=torch.randn(length,3,generator=generator)
        centering[f'x{length}']=x;centering[f'y{length}']=x-x.mean(dim=0,keepdim=True)
    reference=write_case({'name':'centering'},centering)
    (ROOT/'fixtures/centering.json').write_text(json.dumps(reference,indent=2)+'\n')

def run_case(ld,ae,length,steps,seed,mode,trace=False):
    torch.manual_seed(seed)
    cfg=checkpoint('LD1')['hyper_parameters']['cfg_exp']
    OmegaConf.set_struct(cfg,False)
    cfg.product_flowmatcher.local_latents.dim=8
    fm=ProductSpaceFlowMatcher(cfg)
    params=OmegaConf.load(UPSTREAM/'configs/inference_base.yaml').generation.model
    if mode=='ode':
        for dm in params: params[dm].simulation_step_params.sampling_mode='vf'
    draws=[]; states=[]; velocities=[]; traces={}
    original_randn=torch.randn
    def record_randn(*a,**kw):
        x=original_randn(*a,**kw);draws.append(x.clone());return x
    def predict(batch,mode='full'):
        out=ld(batch)
        states.append({k:v.clone() for k,v in batch['x_t'].items()})
        velocities.append({k:v['v'].clone() for k,v in out.items()})
        return out
    hooks=[]
    if trace:
        for name,mod in ld.named_modules():
            if name in ['init_repr_factory','cond_factory','pair_repr_builder'] or name.startswith('transformer_layers.') and name.count('.')==1:
                hooks.append(mod.register_forward_hook(lambda m,a,out,n=name:traces.setdefault(n,out.detach().clone())))
    with patch('torch.randn',record_randn):
        x,_=fm.full_simulation({},predict,nsteps=steps,nsamples=1,n=length,self_cond=True,
              sampling_model_args=params,device=torch.device('cpu'))
    for h in hooks:h.remove()
    output=ae({'z_latent':x['local_latents'],'ca_coors_nm':x['bb_ca'],'mask':torch.ones(1,length,dtype=torch.bool),
               'residue_mask':torch.ones(1,length,dtype=torch.bool)})
    result={'initial_ca':draws[0],'initial_z':draws[1],'final_ca':x['bb_ca'],'final_z':x['local_latents'],
            'coords':output['coors_nm'],'logits':output['seq_logits'],'residue_types':output['aatype_max'],
            'atom_mask':output['atom_mask'],'masked_coords':output['coors_nm']*output['atom_mask'][...,None]}
    for i,d in enumerate(draws[2:]):result[f'draw.{i}']=d
    for i,(s,v) in enumerate(zip(states,velocities)):
        for k in s:result[f'state.{i}.{k}']=s[k];result[f'velocity.{i}.{k}']=v[k]
    for k,v in traces.items():result[f'trace.{k}']=v
    return result

def write_case(case,data):
    directory=ROOT/'.cache/fixtures';directory.mkdir(parents=True,exist_ok=True)
    blob=bytearray(); tensors={}
    for name,t in data.items():
        a=t.detach().float().cpu().contiguous().numpy().astype('<f4')
        raw=a.tobytes();tensors[name]={'shape':list(a.shape),'offset':len(blob),'length':a.size}
        blob.extend(raw)
    file=f'{case["name"]}.bin';(directory/file).write_bytes(blob)
    case.update(file=f'.cache/fixtures/{file}',byteLength=len(blob),sha256=hashlib.sha256(blob).hexdigest(),tensors=tensors)
    return case

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lengths',nargs='+',type=int,default=[32,64]);p.add_argument('--steps',nargs='+',type=int,default=[1,3]);
    p.add_argument('--precisions',nargs='+',default=['fp32','fp16','int8']);p.add_argument('--trace',action='store_true');p.add_argument('--output',default='oracles.json');args=p.parse_args()
    constants();refs={};manifest={'upstream_commit':COMMIT,'torch':torch.__version__,'cases':[],'quantization_drift':{}}
    for precision in args.precisions:
        ld,_=make_model('LD1',state=reload_bundle('LD1',precision));ae,_=make_model('AE1',state=reload_bundle('AE1',precision))
        for length in args.lengths:
            for steps in args.steps:
                for mode in ['sde','ode']:
                    key=f'L{length}_T{steps}_{mode}';name=f'{precision}_{key}';seed=2317+length
                    result=run_case(ld,ae,length,steps,seed,mode,trace=args.trace and steps==1)
                    case=write_case({'name':name,'precision':precision,'length':length,'steps':steps,'seed':seed,'mode':mode},result)
                    manifest['cases'].append(case)
                    if precision=='fp32':refs[key]={k:v.clone() for k,v in result.items() if k in ['final_ca','final_z','coords','logits']}
                    elif key in refs:manifest['quantization_drift'][name]={k:stats(result[k],v) for k,v in refs[key].items()}
                    print(name,'complete',flush=True)
        del ld,ae
    if Path(args.output).name!=args.output:raise ValueError('--output must be a filename')
    (ROOT/'fixtures'/args.output).write_text(json.dumps(manifest,indent=2)+'\n')
