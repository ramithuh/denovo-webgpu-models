#!/usr/bin/env python3
"""Independent upstream full-model traces. Storage rounding uses F32 compute in all modes."""
import argparse, hashlib, json
import numpy as np
import torch
from upstream import ROOT, COMMIT, load_models, restore
from export_esm3 import load_bundle
from esm.utils.structure.affine3d import build_affine3d_from_coordinates
from esm.utils.constants.esm3 import SEQUENCE_VOCAB

def inputs():
    seq='NLYIQWLKDGGPSSGRPPPS'
    tok=torch.tensor([[0]+[SEQUENCE_VOCAB.index(c) for c in seq]+[2]])
    coords=torch.full((1,22,3,3),float('nan'))
    for line in (ROOT/'fixtures/1l2y.pdb').read_text().splitlines():
        if not line.startswith('ATOM'):continue
        name=line[12:16].strip()
        if name not in ['N','CA','C']:continue
        n=int(line[22:26]);coords[0,n,['N','CA','C'].index(name)]=torch.tensor([float(line[k:k+8]) for k in [30,38,46]])
    synthetic=torch.tensor([[0]+[SEQUENCE_VOCAB.index(c) for c in 'ACDEFGHIKLMN']+[2]])
    masked=tok.clone();masked[:,[2,4,8,13]]=32
    multi=dict(sequence_tokens=synthetic,ss8_tokens=torch.tensor([[0]+[3+i%8 for i in range(12)]+[0]]),sasa_tokens=torch.tensor([[0]+[3+i%16 for i in range(12)]+[0]]),function_tokens=torch.zeros(1,14,8,dtype=torch.long),residue_annotation_tokens=torch.zeros(1,14,16,dtype=torch.long))
    multi['function_tokens'][0,3]=torch.arange(4,12);multi['residue_annotation_tokens'][0,4,:3]=torch.tensor([7,29,122])
    inverse=dict(sequence_tokens=tok.clone(),structure_coords=coords,per_res_plddt=torch.tensor([[0]+[1.]*20+[0]]))
    inverse['sequence_tokens'][:,1:-1]=32
    return {'sequence':dict(sequence_tokens=tok),'masked':dict(sequence_tokens=masked),'inverse':inverse,'multitrack':multi}

class Writer:
    def __init__(self,precision):
        self.file=ROOT/'weights'/f'oracle-{precision}.bin';self.f=self.file.open('wb');self.records={}
    def tensor(self,name,t):
        t=t.detach().float().cpu().contiguous();a=t.numpy().astype('<f4');raw=a.tobytes();self.records[name]=dict(shape=list(t.shape),offset=self.f.tell(),bytes=len(raw));self.f.write(raw)
    def close(self):self.f.close()

def stats(a,b):
    d=(a.double()-b.double());return dict(relRms=float((d.square().sum()/b.double().square().sum().clamp_min(1e-30)).sqrt()),maxAbs=float(d.abs().max()))

@torch.inference_mode()
def execute(models,precision):
    m,e,d=[models[k] for k in ['esm','encoder','decoder']];w=Writer(precision);cases={};finals={}
    for name,raw in inputs().items():
        case={k:v.tolist()[0] for k,v in raw.items()};kw={k:v.clone() for k,v in raw.items()};traces={};hooks=[]
        def hook(label):return lambda mod,args,out:traces.update({label:(out[0] if isinstance(out,tuple) else out).detach()})
        if 'structure_coords' in kw:
            frames,mask=build_affine3d_from_coordinates(kw['structure_coords']);traces['encoder.frames']=torch.cat([frames.tensor,mask[...,None]],-1)
            edges,_=e.find_knn_edges(kw['structure_coords'],torch.zeros_like(mask),mask,torch.zeros_like(mask,dtype=torch.long),16);traces['encoder.edges']=edges
            hooks.append(e.relative_positional_embedding.register_forward_hook(hook('encoder.embedding')))
            for i,b in enumerate(e.transformer.blocks):hooks.append(b.register_forward_hook(hook(f'encoder.transformer.blocks.{i}')))
            hooks.append(e.pre_vq_proj.register_forward_hook(hook('encoder.pre_vq')))
            _,tokens=e.encode(kw['structure_coords']);traces['encoder.tokens']=tokens;kw['structure_tokens']=tokens
            for h in hooks:h.remove()
            hooks=[]
        hooks.append(m.encoder.register_forward_hook(hook('esm.encoder')))
        hooks.append(m.transformer.norm.register_forward_hook(hook('esm.normalized')))
        for i,b in enumerate(m.transformer.blocks):hooks.append(b.register_forward_hook(hook(f'esm.transformer.blocks.{i}')))
        result=m(**kw)
        for h in hooks:h.remove()
        for field in ['sequence_logits','structure_logits','secondary_structure_logits','sasa_logits','function_logits','residue_logits','embeddings']:
            k='ss8_logits' if field=='secondary_structure_logits' else field;traces[k]=getattr(result,field);finals[name+'.'+k]=traces[k].clone()
        for k,t in traces.items():w.tensor(name+'/'+k,t)
        cases[name]=dict(input=case,outputs=list(traces));print(precision,name,'forward complete',flush=True)
        if name=='sequence':
            st=result.structure_logits.argmax(-1);st[:,0]=4098;st[:,-1]=4097;traces={};hooks=[]
            hooks.append(d.embed.register_forward_hook(hook('decoder.embedding')))
            for i,b in enumerate(d.decoder_stack.blocks):hooks.append(b.register_forward_hook(hook(f'decoder.decoder_stack.blocks.{i}')))
            hooks.append(d.decoder_stack.norm.register_forward_hook(hook('decoder.normalized')))
            hooks.append(d.affine_output_projection.proj.register_forward_hook(hook('decoder.projection')))
            hooks.append(d.pairwise_classification_head.register_forward_hook(hook('decoder.pairwise_logits')))
            dec=d.decode(st)
            for h in hooks:h.remove()
            traces.update(coordinates=dec['bb_pred'],plddt=dec['plddt'],affine=dec['tensor7_affine'],predicted_aligned_error=dec['predicted_aligned_error'],ptm=dec['ptm'])
            for k,t in traces.items():w.tensor('decode/'+k,t)
            cases['decode']=dict(tokens=st[0].tolist(),outputs=list(traces));finals['decode.coordinates']=dec['bb_pred'].clone();finals['decode.plddt']=dec['plddt'].clone()
            print(precision,'decode complete',flush=True)
    w.close()
    # JSON disallows NaN; missing coordinate components are represented by null.
    def clean(v):
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    info=dict(model='esm3-sm-open-v1',upstream_commit=COMMIT,precision=precision,compute='float32; TF32 disabled; storage-dequantized matching precision',binary='../weights/'+w.file.name,sha256=hashlib.sha256(w.file.read_bytes()).hexdigest(),cases=clean(cases),tensors=w.records)
    (ROOT/'fixtures'/f'oracle-{precision}.json').write_text(json.dumps(info,indent=2)+'\n')
    return finals

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--precisions',nargs='+',default=['fp32','fp16','int8']);a=p.parse_args();models=load_models();original=execute(models,'fp32');metrics={}
    for precision in a.precisions:
        if precision=='fp32':continue
        restore(models,load_bundle(ROOT/'weights'/precision));out=execute(models,precision);metrics[precision]={k:stats(v,original[k]) for k,v in out.items()}
    (ROOT/'fixtures/quantization.json').write_text(json.dumps(metrics,indent=2)+'\n')
