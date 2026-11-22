"""Execute original dynamic upstream forward + sampler with saved random draws."""
import json, hashlib, math, types
from unittest.mock import patch
import torch
from oracle_common import *

def dump(precision,state,cfg):
    quant={n:dequant(*quantize(n,t,precision)) for n,t in state.items()}
    model=upstream_model(quant,cfg)
    from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
    fm=R3NFlowMatcher(zero_com=True,scale_ref=1.)
    out=ROOT/'.cache/fixtures';out.mkdir(parents=True,exist_ok=True)
    data=bytearray(); tensors={}
    def put(name,t):
        a=t.detach().cpu().float().contiguous().numpy().astype('<f4')
        tensors[name]=dict(offset=len(data),length=a.size,shape=list(a.shape))
        data.extend(a.tobytes()); return name
    meta=dict(precision=precision,commit=COMMIT,torch=torch.__version__,tensors=tensors,denoise=[],samples=[])
    with torch.inference_mode():
        for L in [32,64]:
            torch.manual_seed(121+L)
            x=fm.sample_reference(L,shape=(1,),dtype=torch.float32,mask=torch.ones(1,L,dtype=torch.bool))
            sc=torch.randn_like(x)*.3
            for conditioned in [False,True]:
                ident=f'denoise_L{L}_sc{int(conditioned)}'
                batch=dict(x_t=x,t=torch.tensor([.61 if conditioned else .37]),mask=torch.ones(1,L,dtype=torch.bool))
                if conditioned:batch.update(x_sc=sc,cath_code=[['1.1.1.x']])
                hooks=[]; traces={}
                if L==32 and not conditioned:
                    for name in ['cond_factory','transition_c_1','transition_c_2','init_repr_factory','pair_repr_builder']+[f'transformer_layers.{i}' for i in range(15)]:
                        def hook(m,args,y,n=name):traces[n]=put(ident+'.'+n,y)
                        hooks.append(model.get_submodule(name).register_forward_hook(hook))
                y=model(batch)['coors_pred']
                for h in hooks:h.remove()
                meta['denoise'].append(dict(id=ident,length=L,t=float(batch['t'][0]),x=put(ident+'.x',x),
                    xSC=put(ident+'.sc',sc) if conditioned else None,fold=[1,1,1] if conditioned else None,
                    output=put(ident+'.output',y),trace=traces))
            for steps in ([2,3,4] if L==32 else [2,3]):
                ident=f'sample_L{L}_T{steps}'
                schedule='uniform' if (L==32)==(steps==2) else 'log'
                mode='vf' if steps in [2,4] else 'sc'
                guided=steps==4
                if guided:schedule='power'
                draws=[];trajectory=[];predictions=[];clean=[];inputs=[]
                randn=torch.randn
                def record_randn(*args,**kw):
                    y=randn(*args,**kw);draws.append(put(f'{ident}.noise{len(draws)}',y));return y
                def predict(batch):
                    inputs.append(put(f'{ident}.input{len(inputs)}',batch['x_t']))
                    y=model(batch)['coors_pred']
                    predictions.append(put(f'{ident}.pred{len(predictions)}',y))
                    xp=batch['x_t']+(1.-batch['t'][...,None,None])*y
                    if guided:
                        ub=batch.copy();ub.pop('cath_code')
                        uy=model(ub)['coors_pred']
                        up=batch['x_t']+(1.-batch['t'][...,None,None])*uy
                        xp=1.6*xp+(1.-1.6)*up
                    clean.append(put(f'{ident}.clean{len(clean)}',xp))
                    return xp,fm.xt_dot(xp,batch['x_t'],batch['t'],batch['mask'])
                step_original=fm.simulation_step
                def step(*args,**kw):
                    y,t=step_original(*args,**kw)
                    trajectory.append(put(f'{ident}.step{len(trajectory)}',y));return y,t
                fm.simulation_step=step
                torch.manual_seed(3300+L+steps)
                with patch('torch.randn',record_randn):
                    y=fm.full_simulation(predict_clean_n_v=predict,dt=1/steps,nsamples=1,n=L,
                       mask=torch.ones(1,L,dtype=torch.bool),device='cpu',dtype=torch.float32,self_cond=not guided,
                       schedule_mode=schedule,schedule_p=2.,sampling_mode=mode,sc_scale_noise=.45,
                       sc_scale_score=1.,gt_mode='1/t',gt_p=1.,gt_clamp_val=10.,cath_code=[['1.1.1.x']] if guided else None)
                fm.simulation_step=step_original
                meta['samples'].append(dict(id=ident,length=L,steps=steps,schedule=schedule,schedulePower=2.,mode=mode,
                    noiseScale=.45,scoreScale=1.,gtMode='1/t',gtPower=1.,gtClamp=10.,selfCondition=not guided,fold=[1,1,1] if guided else None,guidanceWeight=1.6 if guided else 1.,
                    times=fm.get_schedule(schedule,steps,p1=2.).tolist(),noise=draws,trajectory=trajectory,
                    predictions=predictions,clean=clean,inputs=inputs,output=put(ident+'.output',y)))
    binary=out/f'oracle_{precision}.bin';binary.write_bytes(data)
    meta['binary']=f'.cache/fixtures/{binary.name}';meta['sha256']=hashlib.sha256(data).hexdigest()
    (ROOT/'fixtures'/f'oracle_{precision}.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(precision,len(data),'bytes',flush=True)
if __name__=='__main__':
    torch.set_num_threads(8)
    state,cfg=checkpoint()
    for precision in ['fp32','fp16','int8']:dump(precision,state,cfg)
