"""CPU FP32 upstream traces for every mode, and identical quantization oracles."""
import argparse, hashlib, json, time
import numpy as np
import torch
from reference import ROOT, COMMIT, model_for, batch, relrms
from export_models import reload_bundle
from independent_oracle import dump_independent
from antifold.esm.inverse_folding.features import GVPInputFeaturizer, DihedralFeatures
from antifold.esm.inverse_folding.util import get_rotation_frames, nan_to_num, rbf

class Writer:
    def __init__(self,path): self.path=path;self.data=bytearray();self.records={}
    def put(self,name,value):
        if isinstance(value,torch.Tensor): value=value.detach().cpu().numpy()
        a=np.asarray(value)
        a=np.ascontiguousarray(a,dtype='<i4' if a.dtype.kind in 'biu' else '<f4')
        self.records[name]=dict(offset=len(self.data),shape=list(a.shape),dtype='I32' if a.dtype.kind=='i' else 'F32',length=a.size)
        self.data.extend(a.tobytes())
    def finish(self,meta):
        self.path.with_suffix('.bin').write_bytes(self.data)
        self.path.with_suffix('.json').write_text(json.dumps(dict(**meta,tensors=self.records,byteLength=len(self.data),sha256=hashlib.sha256(self.data).hexdigest()),indent=2)+'\n')

def trace(model,alphabet,case,writer):
    c,p,cf,t=batch(case,alphabet)
    coords=nan_to_num(c); cm=torch.isfinite(c).all(-1).all(-1)
    ns,nv=GVPInputFeaturizer.get_node_features(coords,cm)
    graph=model.encoder.gvp_encoder.embed_graph
    (es,ev),ei=graph.get_edge_features(coords,cm,p)
    valid=(ei[0,0]>=0)
    # Keep the upstream edge order. JS tests align valid edges by (src,dst).
    for name,val in dict(coords=c[0],padding=p[0],confidence=cf[0],tokens=t[0],
        coordMask=cm[0],nodeScalar=ns[0],nodeVector=nv[0],edgeScalar=es[0,valid],
        edgeVector=ev[0,valid],edgeIndex=ei[0,:,valid].T,
        dihedrals=DihedralFeatures._dihedrals(coords)[0],frames=get_rotation_frames(coords)[0],
        confidenceRBF=rbf(cf,0.,1.)[0]).items(): writer.put(name,val)
    handles=[]
    def hook(name,fn=lambda o:o):
        return lambda m,i,o: writer.put(name,fn(o))
    handles.append(graph.register_forward_hook(hook('graph.nodeScalar',lambda o:o[0][0])))
    handles.append(graph.register_forward_hook(hook('graph.nodeVector',lambda o:o[0][1])))
    handles.append(graph.register_forward_hook(hook('graph.edgeScalar',lambda o:o[1][0])))
    handles.append(graph.register_forward_hook(hook('graph.edgeVector',lambda o:o[1][1])))
    for i,layer in enumerate(model.encoder.gvp_encoder.encoder_layers):
        handles.append(layer.register_forward_hook(hook(f'gvp.{i}.scalar',lambda o:o[0][0])))
        handles.append(layer.register_forward_hook(hook(f'gvp.{i}.vector',lambda o:o[0][1])))
    for i,layer in enumerate(model.encoder.layers): handles.append(layer.register_forward_hook(hook(f'encoder.{i}',lambda o:o[:,0])))
    for i,layer in enumerate(model.decoder.layers): handles.append(layer.register_forward_hook(hook(f'decoder.{i}',lambda o:o[0][:,0])))
    try:
        enc=model.encoder(c,p,cf,return_all_hiddens=True)
        writer.put('encoder.input',enc['encoder_states'][0][:,0])
        writer.put('encoder.output',enc['encoder_out'][0][:,0])
        for k,v in enc['encoder_embedding'][0].items(): writer.put('embedding.'+k,v[0])
        logits,extra=model.decoder(t[:,:-1],enc)
        writer.put('decoder.input',extra['inner_states'][1][:,0])
        logits=logits[0].T.contiguous()
        writer.put('logits',logits)
        logp=logits.log_softmax(-1)
        lp=logp.gather(-1,t[:,1:].T).squeeze(-1)
        writer.put('logProbabilities',lp)
        canonical=torch.tensor([alphabet.get_idx(a) for a in 'ACDEFGHIKLMNPQRSTVWY'])
        cp=logits[:,canonical].softmax(-1)
        writer.put('canonicalProbabilities',cp)
        writer.put('entropyPerplexity',torch.exp(-(cp*cp.log()).sum(-1)))
        return logits.numpy(),enc
    finally:
        for h in handles:h.remove()

def replay(model,alphabet,case,enc,writer):
    length=min(6,len(case['sequence']))
    uniforms=np.array([.13,.87,.42,.64,.21,.95],dtype=np.float32)[:length]
    fixed={1:alphabet.get_idx(case['sequence'][1]),4:alphabet.get_idx(case['sequence'][4])} if length>=5 else {}
    tokens=torch.full((1,length+1),alphabet.mask_idx,dtype=torch.long);tokens[0,0]=alphabet.get_idx('<cath>')
    state={};all_logits=[];temperature=.7
    for i in range(length):
        logits,_=model.decoder(tokens[:,:i+1],enc,incremental_state=state)
        row=logits[0,:,-1];all_logits.append(row)
        probs=(row/temperature).softmax(-1)
        token=int(torch.searchsorted(probs.cumsum(-1),torch.tensor(uniforms[i])))
        tokens[0,i+1]=fixed.get(i,min(token,len(alphabet)-1))
    writer.put('replay.uniforms',uniforms);writer.put('replay.logits',torch.stack(all_logits));writer.put('replay.tokens',tokens[0,1:])
    return dict(length=length,temperature=temperature,fixed=fixed)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--variants',nargs='+',default=['antifold','esmif1']);ap.add_argument('--precisions',nargs='+',default=['fp32','fp16','int8']);ap.add_argument('--cases',nargs='+');args=ap.parse_args()
    cases=json.loads((ROOT/'fixtures/cases.json').read_text());metrics=json.loads((ROOT/'fixtures/export_metrics.json').read_text()) if (ROOT/'fixtures/export_metrics.json').exists() else {}
    for variant in args.variants:
        model,alphabet=model_for(variant)
        for precision in args.precisions:
            model.load_state_dict(reload_bundle(ROOT/'weights'/variant/precision),strict=True)
            for case in cases:
                if case['variant']!=variant or args.cases and case['name'] not in args.cases:continue
                name=case['name'];writer=Writer(ROOT/'fixtures'/f'oracle_{name}_{precision}')
                start=time.perf_counter();logits,enc=trace(model,alphabet,case,writer)
                replay_info=replay(model,alphabet,case,enc,writer)
                independent=dump_independent(case,logits,alphabet,writer)
                writer.finish(dict(case=name,variant=variant,precision=precision,upstream_commit=COMMIT,replay=replay_info,independent=independent))
                if precision=='fp32':
                    # Compare round-tripped FP32 to original checkpoint, not itself.
                    original,_=model_for(variant);c,p,cf,t=batch(case,alphabet)
                    baseline=original(c,p,cf,t[:,:-1])[0][0].T.numpy()
                    assert np.array_equal(logits,baseline),name
                    del original
                    (ROOT/'.cache'/f'baseline_{name}.bin').write_bytes(baseline.tobytes())
                else:baseline=np.fromfile(ROOT/'.cache'/f'baseline_{name}.bin',dtype=np.float32).reshape(logits.shape)
                metrics[f'{name}/{precision}']=dict(logits_relRMS=relrms(logits,baseline),seconds=time.perf_counter()-start)
                print(name,precision,metrics[f'{name}/{precision}'],flush=True)
    (ROOT/'fixtures/export_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')

    from measure_exports import measure
    measure()
