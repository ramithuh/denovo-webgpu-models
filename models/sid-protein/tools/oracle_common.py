"""Instantiate pinned upstream neural classes without evaluation-only dependencies.

AST selection removes imports, never rewrites forward methods. All active operations
come from the pinned files. Unused OpenFold triangle/evaluation classes are not built.
The release omits the CATH vocabulary: numerical C/A/T indices are exposed explicitly.
"""
from pathlib import Path
import ast, hashlib, json, math, os, random, typing
import torch, einops, yaml
from loguru import logger
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / '.cache/upstream'
COMMIT = 'dfef6a29d975451144e8b0449c12cbea28b4ae15'
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

def upstream_namespace():
    ns = dict(torch=torch, einops=einops, math=math, os=os, random=random,
              logger=logger, checkpoint=checkpoint, F=torch.nn.functional,
              nn=torch.nn, Tensor=torch.Tensor, einsum=torch.einsum,
              rearrange=einops.rearrange, **vars(typing))
    base = UP / 'training/proteina/proteinfoundation'
    for f in ['utils/ff_utils/idx_emb_utils.py', 'nn/feature_factory.py',
              'nn/alphafold3_pytorch_utils/modules.py', 'nn/pair_bias_attn/pair_bias_attn.py',
              'nn/protein_transformer.py']:
        tree = ast.parse((base / f).read_text(), filename=f)
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(base/f), 'exec'), ns)
    def mapping(self, unused):
        # This supplies only sizes. No label-to-index vocabulary is invented.
        for level, count in [('C',5),('A',43),('T',1472)]:
            setattr(self, 'class_mapping_'+level, {})
            setattr(self, 'num_classes_'+level, count)
    ns['FoldEmbeddingSeqFeat'].create_mapping = mapping
    def fold(self, batch):
        ids = batch.get('cath_indices', [5,43,1472])
        x = torch.cat([getattr(self, 'embedding_'+level).weight[idx] for level,idx in zip('CAT',ids)])
        return x[None,None,:].expand(batch['x_t'].shape[0],batch['x_t'].shape[1],-1)
    ns['FoldEmbeddingSeqFeat'].forward = fold
    return ns

def state():
    from download_weights import SHA256, digest
    p=ROOT/'weights/16-step-checkpoint.pt'
    assert digest(p)==SHA256
    raw=torch.load(p,map_location='cpu',weights_only=True)
    assert all(k.startswith('model.nn.') for k in raw)
    return {k[len('model.nn.'):]:v for k,v in raw.items()}

def model(s=None):
    cfg=yaml.safe_load((UP/'training/proteina/configs/experiment_config/model/nn/ca_af3_200M_no_tri.yaml').read_text())
    m=upstream_namespace()['ProteinTransformerAF3'](**cfg).eval().requires_grad_(False)
    m.load_state_dict(state() if s is None else s,strict=True)
    return m

def quantize(t,p):
    if t.ndim != 2 or p=='fp32': return t.clone(),None
    if p=='fp16': return t.half(),None
    scale=t.abs().amax(1).clamp_min(1e-30)/127
    return (t/scale[:,None]).round().clamp(-127,127).to(torch.int8),scale

def reload_bundle(p):
    folder=ROOT/'weights'/p
    m=json.loads((folder/'manifest.json').read_text()); out={}; cache={}
    for name,r in m['tensors'].items():
        parts=[]
        for part in r['parts']:
            sh=part['shard']
            if sh not in cache:
                spec=next(x for x in m['shards'] if x['file']==sh)
                data=(folder/sh).read_bytes()
                assert len(data)==spec['byteLength'] and hashlib.sha256(data).hexdigest()==spec['sha256'],sh
                cache[sh]=load_file(str(folder/sh))
            t=cache[sh][part['key']].float()
            if part['dtype']=='I8': t=t*cache[sh][part['quant']['scale_tensor']][:,None]
            parts.append(t)
        out[name]=torch.cat(parts,0)
    return out

def stats(a,b):
    a,b=a.double(),b.double()
    return {'relRms':float(((a-b).square().sum()/b.square().sum().clamp_min(1e-30)).sqrt()),'maxAbs':float((a-b).abs().max())}

def forward(m,x,t,cath=None, hooks=None):
    b={'x_t':x,'t':torch.tensor([t],dtype=torch.float32),'mask':torch.ones(x.shape[:2],dtype=torch.bool)}
    if cath is not None: b['cath_indices']=cath
    out=m(b)['coors_pred']
    return x+(1-b['t'][:,None,None])*out

def schedule(n,t_init=.37):
    if n==1: return [float(torch.tensor(t_init))]
    return [float(1-10**(-(s/400)*2)) for s in torch.round(torch.linspace(30,400,n))]

def center(x): return x-x.mean(-2,keepdim=True)

# Execute original sampler functions, substituting only the source of Gaussian draws.
def sample(m,noise,steps,scale=.45,cath=None,t_init=.37,gamma=1):
    import __future__
    ns={'torch':torch,'einops':einops}
    selected=[('training/proteina/proteinfoundation/utils/align_utils/align_utils.py',{'mean_w_mask'}),
              ('training/proteina/proteina_utils.py',{'_force_zero_com','_apply_mask','_extend_t','_mask_and_zero_com','interpolate','sample_reference'}),
              ('generate_distilled_proteina.py',{'generate_multistep','generate_onestep'})]
    for file,names in selected:
        tree=ast.parse((UP/file).read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(tree,str(UP/file),'exec',flags=__future__.annotations.compiler_flag),ns)
    draws=iter(noise)
    def reference(n,shape,device,mask):return ns['_mask_and_zero_com'](next(draws),mask)
    ns['sample_reference']=reference
    traj=[];inputs=[]
    class Wrapper:
        def __call__(self,batch):
            if cath is not None:batch['cath_indices']=cath
            out=m(batch)['coors_pred']; x=batch['x_t']+(1-batch['t'][:,None,None])*out
            inputs.append(batch['x_t'].clone());traj.append(x.clone());return x,None
        @property
        def model(self):return self
        predict_clean=__call__
    L=noise.shape[-2];mask=torch.ones((1,L),dtype=torch.bool);batch={'mask':mask}
    args=(Wrapper(),batch,1,L,mask,torch.zeros(1,L,3))
    if steps==1:ns['generate_onestep'](*args,torch.device('cpu'),gamma,t_init)
    else:ns['generate_multistep'](*args,steps,scale,torch.device('cpu'))
    return torch.stack(traj),torch.stack(inputs)
