"""Original MiniMPNN graph, encoder, decoder and sequence-logprob oracles."""
import json
from environment import ROOT
from export_protpardelle import config_for, reload_bundle, checkpoint_state
from dump_oracle import save_fixture
import torch
from protpardelle.core.models import MiniMPNN
from protpardelle.utils import dict_to_namespace
from protpardelle.data.pdb_io import load_feats_from_pdb

torch.set_num_threads(8)
torch.set_grad_enabled(False)
name='cc58_epoch97_minimpnn'
config=config_for(name)
model=MiniMPNN(dict_to_namespace(config)).cpu().eval()
source,_=load_feats_from_pdb(ROOT/'examples/motifs/nanobody/7eow_B_atom.pdb')
cases=[]
original=checkpoint_state(ROOT/'.cache/model_params/weights'/f'{name}.pth')
model.load_state_dict({k.removeprefix('mpnn_model.'):v for k,v in original.items() if k.startswith('mpnn_model.')},strict=True)
reference={};metrics=[]
for length,mode in [(8,'structure_sequence'),(32,'sequence_conditioned'),(64,'graph_top32')]:
    crop=torch.nn.functional.one_hot(source['aatype'][:length],21).float()[None] if mode=='sequence_conditioned' else None
    reference[mode]=model(source['atom_positions'][:length].unsqueeze(0),torch.tensor([3.]),torch.ones(1,length),torch.arange(1,length+1)[None],seq_crop_cond=crop).clone()
for precision in ('fp32','fp16','int8'):
    bundle=reload_bundle(ROOT/'weights'/name/precision)
    model.load_state_dict({k.removeprefix('mpnn_model.'):v for k,v in bundle.items() if k.startswith('mpnn_model.')},strict=True)
    for length,mode in [(8,'structure_sequence'),(32,'sequence_conditioned'),(64,'graph_top32')]:
        coords=source['atom_positions'][:length].unsqueeze(0)
        mask=torch.ones(1,length)
        residue_index=torch.arange(1,length+1)[None]
        level=torch.tensor([3.])
        crop=torch.nn.functional.one_hot(source['aatype'][:length],21).float()[None] if mode=='sequence_conditioned' else None
        traces={}
        def hook(name):
            return lambda m,args,out:traces.__setitem__(name,out.detach().clone())
        handles=[model.noise_block.register_forward_hook(hook('mpnn.noise')),
                 model.mpnn_net.mpnn.features.edge_embedding.register_forward_pre_hook(lambda m,args:traces.__setitem__('mpnn.features.raw',args[0].clone())),
                 model.mpnn_net.mpnn.features.register_forward_hook(lambda m,args,out:traces.update({'mpnn.features':out[0].clone(),'mpnn.neighbors':out[1].clone()})),
                 model.proj_out.register_forward_hook(hook('mpnn.logits'))]
        for i,layer in enumerate(model.mpnn_net.mpnn.encoder_layers):
            handles.append(layer.register_forward_hook(lambda m,args,out,i=i:traces.update({f'mpnn.encoder.{i}.node':out[0].clone(),f'mpnn.encoder.{i}.edge':out[1].clone()})))
        for i,layer in enumerate(model.mpnn_net.mpnn.decoder_layers):
            handles.append(layer.register_forward_hook(hook(f'mpnn.decoder.{i}')))
        output=model(coords,level,mask,residue_index,seq_crop_cond=crop)
        metrics.append({'variant':name,'mode':mode,'precision':precision,'output_relRMS':float(((output.double()-reference[mode].double()).square().sum()/reference[mode].double().square().sum()).sqrt())})
        traces['mpnn.logprobs']=output
        for h in handles:h.remove()
        data={'noisy_coords':coords,'noise_level':level.expand(1,length),'seq_mask':mask,'residue_index':residue_index,'chain_index':torch.zeros_like(residue_index),'seq_crop_cond':crop,'neighbor_indices':traces['mpnn.neighbors']}
        cases.append({'mode':mode,'precision':precision,'length':length,'inputs':data,'traces':traces})
save_fixture(name,config,cases)
path=ROOT/'fixtures'/f'oracle_{name}.json'
meta=json.loads(path.read_text());meta['scope']='minimpnn';path.write_text(json.dumps(meta,indent=2)+'\n')
print(f'{len(cases)} MiniMPNN traces')
(ROOT/'fixtures/minimpnn_export_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
