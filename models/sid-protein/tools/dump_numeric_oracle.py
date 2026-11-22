import torch,json
from oracle_common import ROOT,upstream_namespace

def dump():
    from dump_oracle import save
    ns=upstream_namespace();g=torch.Generator().manual_seed(233);cases=[]
    for width in [512,768]:
        x=torch.randn(3,width,generator=g);w=torch.randn(width,generator=g);b=torch.randn(width,generator=g)
        cases.append({'width':width,'input':save(f'norm{width}_input.bin',x),'weight':save(f'norm{width}_weight.bin',w),'bias':save(f'norm{width}_bias.bin',b),'output':save(f'norm{width}_output.bin',torch.nn.functional.layer_norm(x,[width],w,b))})
    t=torch.tensor([.37]);pos=torch.cat([ns['get_index_embedding'](torch.arange(1,33).float()[None],128),torch.zeros(1,32,4)],-1)
    out={'norms':cases,'time':save('time_embedding.bin',ns['get_time_embedding'](t,256)),'position':save('position_embedding.bin',pos)}
    (ROOT/'fixtures/operators.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':dump()
