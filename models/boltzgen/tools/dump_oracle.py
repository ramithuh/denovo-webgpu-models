"""Dynamic unmodified PyTorch forward oracle. Saves tensors, never ATen programs."""
from environment import ROOT, COMMIT, load_model
import argparse,json,time
import torch
from safetensors.torch import save_file
from torch.nn.attention import sdpa_kernel,SDPBackend
from export_boltzgen import reload_bundle

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',default='diverse');parser.add_argument('--precision',default='fp32');parser.add_argument('--cases',nargs='+')
    args=parser.parse_args();torch.set_num_threads(4);torch.set_grad_enabled(False)
    model,_=load_model(args.variant)
    if args.precision!='fp32':reload_bundle(model,args.variant,args.precision)
    files=[ROOT/f'weights/dynamic/{c}.json' for c in args.cases] if args.cases else sorted((ROOT/'weights/dynamic').glob('*.json'))
    for file in files:
        record=json.loads(file.read_text());options=record['options'];features={};draws={};inputs={}
        integers={'msa','method_feature','modified','mol_type','binding_type','ss_type','type_bonds','feature_asym_id','feature_residue_index','entity_id','token_index','sym_id','ref_space_uid'}
        for key,t in record['features'].items():
            name=key[2:] if key.startswith('f:') else key
            value=torch.tensor(t['data'],dtype=torch.long if name in integers else torch.float32).reshape(t['shape'])
            inputs[key]=value
            if key.startswith('f:'):features[name]=value
            else:draws[key]=value
        counter=0;original=torch.randn
        def randn(*shape,**kwargs):
            nonlocal counter
            wanted=tuple(shape[0]) if len(shape)==1 and isinstance(shape[0],(tuple,list,torch.Size)) else tuple(shape)
            value=draws[f'rng:{counter}'];counter+=1
            assert tuple(value.shape)==wanted,(value.shape,wanted)
            return value.clone().to(dtype=kwargs.get('dtype') or torch.float32,device=kwargs.get('device') or 'cpu')
        outputs={};hooks=[]
        def hook(name):
            def save(module,inp,out):
                def flatten(x,k):
                    if isinstance(x,torch.Tensor):outputs[k]=x.detach().float().contiguous().clone()
                    elif isinstance(x,(tuple,list)):
                        for i,v in enumerate(x):flatten(v,k+'.'+str(i))
                    elif isinstance(x,dict):
                        for n,v in x.items():flatten(v,k+'.'+n)
                flatten(out,name)
            return save
        for name,module in model.named_modules():
            if name in ['input_embedder','token_distance_module','template_module','msa_module','diffusion_conditioning','pairformer_module.layers.0','pairformer_module.layers.63']:hooks.append(module.register_forward_hook(hook(name)))
        torch.randn=randn;start=time.perf_counter()
        try:
            with sdpa_kernel(SDPBackend.MATH):out=model(features,recycling_steps=options.get('recyclingSteps',0),num_sampling_steps=options['numSteps'])
        finally:
            torch.randn=original
            for h in hooks:h.remove()
        assert counter==1+3*options['numSteps']
        outputs['atom_coords']=out['sample_atom_coords'].clone()
        for name,key in [('trajectory','coords_traj'),('denoised','x0_coords_traj')]:
            for i,t in enumerate(out[key]):outputs[f'{name}.{i}']=t.clone()
        assert all(torch.isfinite(t).all() for t in outputs.values())
        prefix=ROOT/f'weights/dynamic/{args.variant}_{args.precision}_{file.stem}'
        save_file(inputs,str(prefix)+'.inputs.safetensors');save_file(outputs,str(prefix)+'.outputs.safetensors')
        metadata={**options,'variant':args.variant,'precision':args.precision,'upstream_commit':COMMIT,'seconds':time.perf_counter()-start,'nAtoms':int(features['atom_pad_mask'].sum()),'tokens':features['token_pad_mask'].shape[1]}
        (ROOT/f'fixtures/dynamic_oracle_{args.variant}_{args.precision}_{file.stem}.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print('ORACLE',metadata,flush=True)
if __name__=='__main__':main()
