"""Independent pinned PyTorch forwards and complete dynamic-step sampler trajectories."""
import hashlib,json,time,torch
from oracle_common import ROOT,COMMIT,model,reload_bundle,forward,sample,schedule,stats
CASES=[(32,16,None,.45),(64,16,None,.45),(32,4,None,.45),(64,8,None,.45),(32,1,None,.45),(32,4,[0,0,0],.3)]
DEST=ROOT/'.cache/fixtures'; DEST.mkdir(parents=True,exist_ok=True)
def save(name,t):
    a=t.detach().contiguous().numpy().astype('<f4'); b=a.tobytes(); (DEST/name).parent.mkdir(parents=True,exist_ok=True); (DEST/name).write_bytes(b)
    return {'file':'.cache/fixtures/'+name,'shape':list(a.shape),'byteLength':len(b),'sha256':hashlib.sha256(b).hexdigest()}
if __name__=='__main__':
    originals={}
    for p in ['fp32','fp16','int8']:
        m=model(reload_bundle(p)); cases=[]
        with torch.inference_mode():
            for L,T,cath,scale in CASES:
                name=f'L{L}_T{T}'+('_cath' if cath else '')
                noise=torch.randn(T,1,L,3,generator=torch.Generator().manual_seed(5))
                traces={}; handles=[]
                # One independent forward plus module-level traces for debugging.
                for modname in ['cond_factory','transition_c_2','init_repr_factory','pair_repr_builder','transformer_layers.0','transformer_layers.14','coors_3d_decoder']:
                    handles.append(m.get_submodule(modname).register_forward_hook(lambda mod,inp,out,key=modname:traces.__setitem__(key,out.detach().clone())))
                x=noise[0]*.2; y=forward(m,x,.37,cath)
                for h in handles: h.remove()
                start=time.time(); trajectory,inputs=sample(m,noise,T,scale,cath)
                case={'name':name,'seed':5,'length':L,'steps':T,'noiseScale':scale,'cathIndices':cath,'times':schedule(T),
                      'noise':save(f'{name}_noise.bin',noise),'forwardInput':save(f'{name}_input.bin',x),
                      'forward':save(f'{p}_{name}_forward.bin',y),'trajectory':save(f'{p}_{name}_trajectory.bin',trajectory),
                      'samplerInputs':save(f'{p}_{name}_sampler_inputs.bin',inputs),
                      'traces':{k:save(f'{p}_{name}_{k}.bin',v) for k,v in traces.items()},'seconds':time.time()-start}
                if p=='fp32': originals[name]=trajectory
                case['quantizationDrift']=stats(trajectory[-1],originals[name][-1]);cases.append(case)
                print(p,name,case['seconds'],case['quantizationDrift'],flush=True)
        if p=='fp32':
            with torch.inference_mode():
                noise=torch.randn(8,1,64,3,generator=torch.Generator().manual_seed(1298))
                x=noise[0]*.2;y=forward(m,x,.37);trajectory,inputs=sample(m,noise,8)
                regression={'name':'L64_T8_seed1298','seed':1298,'length':64,'steps':8,'noiseScale':.45,'times':schedule(8),
                    'noise':save('sensitivity/L64_T8_noise.bin',noise),
                    'forwardInput':save('sensitivity/L64_T8_input.bin',x),
                    'forward':save('sensitivity/fp32_L64_T8_forward.bin',y),
                    'trajectory':save('sensitivity/fp32_L64_T8_trajectory.bin',trajectory),
                    'samplerInputs':save('sensitivity/fp32_L64_T8_sampler_inputs.bin',inputs)}
                (ROOT/'fixtures/sensitivity_regression.json').write_text(json.dumps(regression,indent=2)+'\n')
        (ROOT/f'fixtures/oracle_{p}.json').write_text(json.dumps({'commit':COMMIT,'torch':torch.__version__,'reference':'pinned upstream classes and sampler functions, CPU FP32 arithmetic; quantized weights decoded to FP32','cases':cases},indent=2)+'\n')

    from dump_numeric_oracle import dump
    dump()
