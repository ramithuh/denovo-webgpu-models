"""Execute the unmodified pinned Denoiser, saving exact noise draws and every stage.
No ATen/FX trace or op graph is captured. Binary tensors live only in .cache/fixtures.
"""
import json,hashlib,types,torch
from oracle_common import *

def canonicalize_quat(q):
    c0 = q[..., 0:1].abs() > 1e-7
    c1 = (~c0) & (q[..., 1:2].abs() > 1e-7)
    c2 = (~c0) & (~c1) & (q[..., 2:3].abs() > 1e-7)
    c3 = (~c0) & (~c1) & (~c2) & (q[..., 3:4].abs() > 1e-7)
    neg = (
        (c0 & (q[..., 0:1] < 0))
        | (c1 & (q[..., 1:2] < 0))
        | (c2 & (q[..., 2:3] < 0))
        | (c3 & (q[..., 3:4] < 0))
    )
    return torch.where(neg, -q, q)

def patch_canonical_orientations(m):
    orig = m.pair_feature_net._encode_orientations
    def wrapped(self, rots, mask):
        return canonicalize_quat(orig(rots, mask))
    m.pair_feature_net._encode_orientations = types.MethodType(wrapped, m.pair_feature_net)

def dump(precisions=('fp32','fp16','int8')):
    folder=ROOT/'.cache/fixtures';folder.mkdir(parents=True,exist_ok=True)
    drift={};baseline={}
    for precision in precisions:
        m=model(reload_bundle(precision));patch_canonical_orientations(m);cases=[]
        for L in (24,48):
            for mode in ('unconditional','single_motif','multi_motif'):
                steps=2 if L==24 else 3;scale=0.6 if mode=='unconditional' else 0.4
                m.single_feature_net.n_timestep=steps
                name=f'{precision}_{mode}_L{L}_T{steps}';tensors={};chunks=[];offset=0
                def save(key,t):
                    nonlocal offset
                    if not isinstance(t,torch.Tensor):return
                    t=t.detach().cpu().float().contiguous();data=t.numpy().astype('<f4').tobytes()
                    tensors[key]={'shape':list(t.shape),'offset':offset,'byteLength':len(data)};chunks.append(data);offset+=len(data)
                f=features(L,mode)
                for k,v in f.items():save('features.'+k,v)
                gen=torch.Generator().manual_seed(17000+L)
                x=torch.randn(1,L,3,generator=gen);save('initial',x)
                beta,alpha,cum=schedule(steps);save('betas',beta)
                traj=[]
                with torch.inference_mode():
                    for t in range(steps,0,-1):
                        label=f'step{t}.';hooks=[]
                        def hook(key,extract=lambda x:x):
                            return lambda mod,args,out:save(label+key,extract(out))
                        hooks.append(m.single_feature_net.register_forward_hook(hook('single')))
                        hooks.append(m.pair_feature_net.register_forward_hook(hook('pair')))
                        hooks.append(m.pair_transform_net.register_forward_hook(hook('pair_final')))
                        for i,layer in enumerate(m.structure_net.net):
                            hooks.append(layer.register_forward_hook(hook(f'structure{i}.single',lambda x:x[0])))
                            hooks.append(layer.register_forward_hook(hook(f'structure{i}.rots',lambda x:x[2].rots)))
                            hooks.append(layer.register_forward_hook(hook(f'structure{i}.trans',lambda x:x[2].trans)))
                        ts=frame(x,f);save(label+'input_rots',ts.rots);save(label+'input_trans',x)
                        save(label+'orientations',m.pair_feature_net._encode_orientations(ts.rots,f['residue_mask']))
                        z=m(ts,torch.tensor([t]),f)['z'];save(label+'z',z)
                        for h in hooks:h.remove()
                        noise=torch.randn(1,L,3,generator=gen) if t>1 else torch.zeros_like(x);save(label+'noise',noise)
                        wz=(1-alpha[t])/torch.sqrt(1-cum[t])
                        mean=(1/torch.sqrt(alpha[t]))*(x-wz*z)*f['residue_mask'][...,None]
                        x=(mean+scale*torch.sqrt(beta[t])*noise)*f['residue_mask'][...,None] if t>1 else mean
                        save(label+'output',x);traj.append(x.clone())
                save('final',x);blob=b''.join(chunks);fn=name+'.bin';(folder/fn).write_bytes(blob)
                case={'name':name,'length':L,'mode':mode,'steps':steps,'scale':scale,'file':'.cache/fixtures/'+fn,'sha256':hashlib.sha256(blob).hexdigest(),'byteLength':len(blob),'tensors':tensors};cases.append(case)
                key=f'{mode}_L{L}_T{steps}'
                if precision=='fp32':baseline[key]=x.clone()
                if key in baseline:drift.setdefault(precision,{})[key]=stats(x,baseline[key])
                print(name,'bytes',len(blob),flush=True)
        (ROOT/f'fixtures/oracle_{precision}.json').write_text(json.dumps({'commit':COMMIT,'precision':precision,'torch':torch.__version__,'cases':cases},indent=2)+'\n')
    (ROOT/'fixtures/quantization_drift.json').write_text(json.dumps(drift,indent=2)+'\n')
if __name__=='__main__':dump()
