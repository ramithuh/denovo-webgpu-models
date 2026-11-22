"""Oracle adapter around the unmodified, pinned upstream ESMC implementation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/esm'))
import torch
from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from download_weights import COMMIT, REVISION
SEQUENCES={'trp_cage':'NLYIQWLKDGGPSSGRPPPS','ubiquitin':'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'}
AA='ACDEFGHIKLMNPQRSTVWY'
CONFIG=dict(d_model=960,n_heads=15,n_layers=30,ffn_size=2560,residual_scale=(30/36)**.5,layer_norm_eps=1e-5,vocab_size=64,max_sequence_length=1022)
def setup():
 torch.set_num_threads(8);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
def load_upstream():
 setup()
 tokenizer=EsmSequenceTokenizer()
 with torch.device('meta'): model=ESMC(960,15,30,tokenizer,use_flash_attn=False)
 state=torch.load(ROOT/'.cache/esmc_300m_2024_12_v0.pth',map_location='cpu',weights_only=True,mmap=True)
 model.load_state_dict(state,strict=True,assign=True)
 model=model.float().eval()
 for block in model.transformer.blocks:
  block.attn.rotary.device=torch.device('cpu');block.attn.rotary.reset_parameters()
 return model
@torch.inference_mode()
def trace(model,sequence=None,tokens=None,sequence_id=None,attentions=False,row=None):
 if tokens is None: tokens=model._tokenize([sequence])
 else: tokens=torch.tensor(tokens,dtype=torch.long).reshape(1,-1)
 sid=(tokens!=1) if sequence_id is None else torch.tensor(sequence_id,dtype=torch.long).reshape(1,-1)
 result={'tokens':tokens.int(),'sequence_id':sid.int()}
 hooks=[]
 def save(name,value):
  if row is not None: value=value[:,row:row+1]
  result[name]=value.detach().clone()
 hooks.append(model.embed.register_forward_hook(lambda m,a,o:save('embedding',o)))
 for i,block in enumerate(model.transformer.blocks):
  hooks.append(block.register_forward_hook(lambda m,a,o,i=i:save(f'block.{i}',o[0])))
 for i,stage in enumerate(model.sequence_head):
  hooks.append(stage.register_forward_hook(lambda m,a,o,i=i:save(f'head.{i}',o)))
 try:
  out=model(sequence_tokens=tokens,sequence_id=sid,output_attentions=attentions)
  save('final_ln',out.embeddings);save('logits',out.sequence_logits)
  if row is None:
   result['residue_embeddings']=out.embeddings[:,1:-1].clone()
   result['pooled']=out.embeddings[:,1:-1].mean(1)
  if attentions:
   for i,a in enumerate(out.attentions): result[f'attention.{i}']=a
 finally:
  for h in hooks:h.remove()
 return {k:v.cpu().contiguous() for k,v in result.items()}
def metrics(a,b):
 a=a.double();b=b.double();return dict(relRms=float(((a-b).square().sum()/b.square().sum().clamp_min(1e-30)).sqrt()),maxAbs=float((a-b).abs().max()))
