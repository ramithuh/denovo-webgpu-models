"""Exercise every running preset through the original sampler with a portable RNG tape.

Only RNG primitives are intercepted. Model and sampler source are never changed.
External full ProteinMPNN redesign is an explicitly excluded optional submodel.
"""
import copy
import json
import traceback
import argparse
from contextlib import ExitStack
from unittest.mock import patch
from environment import ROOT, COMMIT
from export_protpardelle import checkpoint_state, config_for, reload_bundle
import numpy as np
import torch
import yaml
from protpardelle.core.models import Protpardelle
from protpardelle.utils import dict_to_namespace
from protpardelle.data.pdb_io import load_feats_from_pdb, feats_to_pdb_str, write_coords_to_pdb

torch.set_num_threads(8)
torch.manual_seed(7301)
np.random.seed(7301)


class Recorder:
    def __init__(self):
        self.events=[]
    def wrap(self,name,original):
        def call(*args,**kwargs):
            value=original(*args,**kwargs)
            if isinstance(value,torch.Tensor):data=value.detach().cpu().clone()
            else:data=torch.as_tensor(np.asarray(value).copy())
            self.events.append({'kind':name,'data':data})
            return value
        return call
    def categorical(self,dist,sample_shape=torch.Size()):
        if sample_shape:raise ValueError('Only scalar categorical draws are used by this sampler')
        uniforms=torch.rand(dist.probs.shape[:-1],device=dist.probs.device)
        value=(uniforms.unsqueeze(-1)>dist.probs.cumsum(-1)).sum(-1).clamp(max=dist.probs.shape[-1]-1)
        self.events.append({'kind':'categorical_uniform','data':uniforms.detach().clone()})
        return value
    def context(self):
        stack=ExitStack()
        for name in ['randn','randn_like']:
            stack.enter_context(patch.object(torch,name,self.wrap(name,getattr(torch,name))))
        for name in ['rand','randint']:
            stack.enter_context(patch.object(np.random,name,self.wrap('numpy_'+name,getattr(np.random,name))))
        recorder=self
        stack.enter_context(patch.object(torch.distributions.Categorical,'sample',lambda distribution,sample_shape=torch.Size():recorder.categorical(distribution,sample_shape)))
        return stack


def make_model(name,precision='fp32'):
    config=config_for(name)
    model=Protpardelle(dict_to_namespace(config),device='cpu').eval()
    state=reload_bundle(ROOT/'weights'/name/precision)
    missing=set(model.state_dict())-set(state)
    if any(not k.startswith('struct_model.net.cond_to_patch_embedding.') for k in missing):raise ValueError(f'Missing weights {missing}')
    state={k:v for k,v in state.items() if k in model.state_dict()}
    model.load_state_dict({k:model.state_dict()[k] for k in missing}|state,strict=True)
    model.load_minimpnn(ROOT/'.cache/model_params/weights/cc58_epoch97_minimpnn.pth')
    mini=reload_bundle(ROOT/'weights/cc58_epoch97_minimpnn'/precision)
    model.mpnn_model.load_state_dict({k.removeprefix('mpnn_model.'):v for k,v in mini.items() if k.startswith('mpnn_model.')},strict=True)
    # Attaching a new PyTorch submodule after parent.eval() leaves it training.
    # All port references explicitly use evaluation mode, including MiniMPNN.
    return model.eval()


def pack(name,record):
    blob=bytearray()
    def convert(value):
        if isinstance(value,torch.Tensor):
            raw=value.detach().cpu().numpy();raw=raw.astype('<i4' if raw.dtype.kind in 'biu' else '<f4')
            result={'offset':len(blob),'length':raw.size,'shape':list(raw.shape),'dtype':'I32' if raw.dtype.kind=='i' else 'F32'};blob.extend(raw.tobytes());return result
        if isinstance(value,dict):return{k:convert(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return[convert(v) for v in value]
        if isinstance(value,np.generic):return value.item()
        return value
    meta=convert(record)
    (ROOT/'fixtures'/f'sampler_{name}.json').write_text(json.dumps(meta,indent=2)+'\n')
    (ROOT/'fixtures'/f'sampler_{name}.bin').write_bytes(blob)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case');args=parser.parse_args()
    source,_=load_feats_from_pdb(ROOT/'examples/motifs/nanobody/7eow_B_atom.pdb')
    # Fixed eight-residue input, serialized by the original output routine.
    mini={k:v[:8] for k,v in source.items()};mini['chain_index']=torch.zeros(8);mini['residue_index']=torch.arange(1,9)
    motif=ROOT/'fixtures/sampler_motif.pdb'
    motif.write_text(feats_to_pdb_str(mini['atom_positions'],aatype=mini['aatype'],atom_mask=mini['atom_mask'],residue_index=mini['residue_index'],chain_index=mini['chain_index']))
    records=[]
    tasks=[(p,None,None) for p in sorted((ROOT/'src/protpardelle/configs/running').glob('*.yaml'))]
    # Exercise every structural checkpoint in the original sampler, including
    # multichain/hotspot families and both legacy UViT checkpoints.
    for name in sorted(p.stem for p in (ROOT/'weights').iterdir() if 'minimpnn' not in p.name):
        multi=name.startswith(('cc78','cc83','cc94','cc95'))
        atom=config_for(name)['model']['task']!='backbone'
        preset='sampling_sidechain_conditional_allatom_s1' if multi and atom else 'sampling_sidechain_conditional' if multi else 'sampling_unconditional_allatom_s1' if atom else 'sampling_unconditional'
        tasks.append((ROOT/'src/protpardelle/configs/running'/f'{preset}.yaml',name,'checkpoint_'+name))
    controls=[('recurrence','sampling_sidechain_conditional','cc78_epoch1431'),('quadratic','sampling_recon_rep','cc58_epoch416'),('cubic','sampling_allatom_recon_rep_s1','cc91_epoch383'),('sequence','sampling_unconditional_allatom_s1','cc91_epoch383'),('initial','sampling_unconditional','cc58_epoch416')]
    controls.append(('stage2_translation','sampling_sidechain_conditional_allatom_recon_rep','cc91_epoch383'))
    tasks.extend((ROOT/'src/protpardelle/configs/running'/f'{preset}.yaml',variant,'control_'+name) for name,preset,variant in controls)
    for task_index,(preset,override,case_id) in enumerate(tasks):
      if args.case and args.case!=(case_id or preset.stem):continue
      for precision in ['fp32','fp16','int8']:
        torch.manual_seed(7301+task_index);np.random.seed(7301+task_index)
        config=yaml.safe_load(preset.read_text())['sampling']
        if case_id=='control_recurrence':
            config['conditional_cfg']['num_recurrence_steps']=2
            config['conditional_cfg']['reconstruction_guidance'].update(enabled=True,schedule='constant',max_scale=.1)
        if case_id=='control_quadratic':
            config['conditional_cfg']['reconstruction_guidance']['schedule']='quadratic'
            config['conditional_cfg']['crop_conditional_guidance'].update(strategy='backbone',start=.5)
            config.update(dx=1.,dy=2.,dz=3.)
        if case_id=='control_cubic':config['conditional_cfg']['reconstruction_guidance']['schedule']='cubic'
        if case_id=='control_stage2_translation':config.update(dx=1.,dy=2.,dz=3.)
        if case_id=='control_sequence':
            config.update(temperature=.6,top_p=.9,disallow_aas=[1,4,20])
            config['allatom_cfg'].update(anneal_seq_resampling_rate='cosine',skip_mpnn_proportion=0.)
        atom='allatom' in preset.stem or 'fastrelax' in preset.stem
        variant=override or ('cc89_epoch415' if 'fastrelax' in preset.stem or 'partial_diffusion_allatom' in preset.stem else 'cc91_tip_epoch480' if 'tip' in preset.stem else 'cc91_epoch383' if atom else 'cc58_epoch416')
        fixture=(case_id or preset.stem)+(f'_{precision}' if precision!='fp32' else '')
        model=make_model(variant,precision)
        cfg=copy.deepcopy(config);cc=cfg['conditional_cfg'];cc['discontiguous_motif_assignment']['fixed_motif_pos']=[[2,3,4]]
        cfg['num_steps']=4;cfg['s_churn']=.8
        pd=cfg['partial_diffusion'];pd['num_steps']=3;pd['pdb_file_path']=str(motif)
        if 'repack' in pd:pd['seq']='EVQLVESG'
        cfg['motif_file_path']=str(motif);cfg.pop('stage2_cfg');allatom=cfg.pop('allatom_cfg');allatom['use_fullmpnn']=False;allatom['use_fullmpnn_for_final']=False;cfg.update(allatom)
        cfg['noise_schedule']=model.make_sampling_noise_schedule(s_min=.001,s_max=2.)
        inputs={'seq_mask':torch.ones(1,8),'residue_index':torch.arange(1,9)[None],'chain_index':torch.zeros(1,8,dtype=torch.long)}
        if case_id=='control_initial':
            cfg['xt_start']=torch.linspace(-10,10,8*111).reshape(1,8,37,3)
            inputs['seq_mask'][:,-2:]=0
        if override and override.startswith(('cc78','cc83','cc94','cc95')):
            inputs['chain_index'][:,4:]=1
            gap=model.config.data.chain_residx_gap
            inputs['residue_index'][:,4:]=torch.arange(1,5) if gap==0 else torch.arange(5,9)+gap
            cfg['hotspots']='A2'
        if cc['enabled']:cfg['motif_placements_full']=['2/A1-3/3']
        tape=Recorder();calls=[];handles=[]
        def pre(module,args,kwargs):
            names=['noisy_coords','noise_level','seq_mask'];data=dict(zip(names,args))|kwargs
            calls.append({'inputs':{k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in data.items()},'traces':{}})
            if data['noisy_coords'].requires_grad:
                trace=calls[-1]['traces']
                data['noisy_coords'].register_hook(lambda gradient:trace.__setitem__('coordinate_gradient',gradient.detach().clone()))
        handles.append(model.struct_model.register_forward_pre_hook(pre,with_kwargs=True))
        def capture(name):return lambda m,args,out:calls[-1]['traces'].__setitem__(name,out.detach().clone())
        for i,(attn,ff) in enumerate(model.struct_model.net.transformer.layers):
            handles.append(attn.register_forward_hook(capture(f'block.{i}.attention')))
            handles.append(ff.register_forward_hook(capture(f'block.{i}.ff')))
            handles.append(ff.register_forward_hook(lambda m,args,out,i=i:calls[-1]['traces'].__setitem__(f'block.{i}.hidden',((args[0]+out)*calls[-1]['inputs']['seq_mask'][...,None]).detach().clone())))
        handles.append(model.struct_model.register_forward_hook(capture('denoised_coords')))
        mini_model=model.mpnn_model
        def mini_pre(module,args,kwargs):
            values=dict(zip(['denoised_coords','noise_level','seq_mask','residue_index','seq_self_cond','seq_crop_cond'],args))|kwargs
            calls[-1]['mini_inputs']={k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in values.items()}
        handles.append(mini_model.register_forward_pre_hook(mini_pre,with_kwargs=True))
        handles.extend([mini_model.noise_block.register_forward_hook(capture('mpnn.noise')),mini_model.mpnn_net.mpnn.features.edge_embedding.register_forward_pre_hook(lambda m,a:calls[-1]['traces'].__setitem__('mpnn.features.raw',a[0].detach().clone())),mini_model.mpnn_net.mpnn.features.register_forward_hook(lambda m,a,o:calls[-1]['traces'].update({'mpnn.features':o[0].detach().clone(),'mpnn.neighbors':o[1].detach().clone()})),mini_model.proj_out.register_forward_hook(capture('mpnn.logits')),mini_model.register_forward_hook(capture('mpnn.logprobs'))])
        for i,layer in enumerate(mini_model.mpnn_net.mpnn.encoder_layers):handles.append(layer.register_forward_hook(lambda m,a,o,i=i:calls[-1]['traces'].update({f'mpnn.encoder.{i}.node':o[0].detach().clone(),f'mpnn.encoder.{i}.edge':o[1].detach().clone()})))
        for i,layer in enumerate(mini_model.mpnn_net.mpnn.decoder_layers):handles.append(layer.register_forward_hook(capture(f'mpnn.decoder.{i}')))
        try:
            with tape.context():result=model.sample(**inputs,**cfg,dummy_fill_mode=model.config.data.dummy_fill_mode)
            first_calls=calls[:];calls.clear();stage2=None
            if config['stage2_cfg']['enabled']:
                stage2_path=ROOT/'.cache/stage2_input.pdb'
                write_coords_to_pdb(result['xt_traj'][-1][0],str(stage2_path),aatype=result['s'][0],chain_index=inputs['chain_index'][0])
                cfg2=copy.deepcopy(cfg)
                cfg2.update({k:v for k,v in config['stage2_cfg'].items() if k not in ['enabled','rewind_steps','noise_schedule']})
                cfg2.update(num_steps=4,s_churn=.8,jump_steps=False,uniform_steps=True,gt_aatype=result['s'],stage2=True,motif_all_atom_stage1=result['motif_all_atom'],motif_idx_stage1=result['motif_idx'])
                cfg2['partial_diffusion'].update(enabled=True,num_steps=3,pdb_file_path=str(stage2_path))
                with tape.context():stage2=model.sample(**inputs,**cfg2,dummy_fill_mode=model.config.data.dummy_fill_mode)
            record={'upstream_commit':COMMIT,'variant':variant,'preset':preset.stem,'scope':'original_sampler','precision':precision,'length':8,'num_steps':4,'noise_schedule':{'s_min':.001,'s_max':2.},'external_fullmpnn':False,'stage2_requested':config['stage2_cfg']['enabled'],'config':config,'inputs':inputs,'initial_coordinates':cfg.get('xt_start'),'hotspots':cfg.get('hotspots'),'rng_tape':tape.events,'denoiser_calls':first_calls,'stage2_denoiser_calls':calls,'outputs':result,'stage2_outputs':stage2}
            pack(fixture,record)
            summary={'preset':fixture,'source_preset':preset.stem,'variant':variant,'precision':precision,'upstream_status':'executed','denoiser_calls':len(first_calls)+len(calls),'random_events':len(tape.events),'port_status':'pending'}
        except Exception as error:
            summary={'preset':fixture,'variant':variant,'upstream_status':'error','error':str(error),'traceback':traceback.format_exc(),'port_status':'pending'}
        finally:
            for h in handles:h.remove()
        records.append(summary);print(json.dumps(summary),flush=True)
    if args.case:
        old=json.loads((ROOT/'fixtures/sampler_inventory.json').read_text());updated={x['preset']:x for x in records};records=[updated.pop(x['preset'],x) for x in old]+list(updated.values())
    (ROOT/'fixtures/sampler_inventory.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
