"""Pinned local upstream modules; sequence-only specialization, no remote code."""
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from esm.models.esm3 import ESM3
from model.transformer import Transformer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

SEQUENCES = {
    'trp_cage': 'NLYIQWLKDGGPSSGRPPPS',
    'ubiquitin': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
}
CONFIG = dict(d_model=1536, backbone_heads=24, backbone_layers=23,
              backbone_total_layers=48, head_heads=6, head_layers=12,
              ffn_size=4096, head_ffn_size=2048, residual_scale=(48/36)**0.5,
              layer_norm_eps=1e-5, head_outer_norm_eps=1e-12, prior=0.05,
              max_sequence_length=1024, special_tokens=False)

def setup():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)

def load_upstream(cache=ROOT / '.cache'):
    setup()
    with torch.device('meta'):
        backbone = ESM3(1536, 24, 256, 48, None, None, None,
                       SimpleNamespace(sequence=SimpleNamespace(mask_token_id=32)))
        head = Transformer(1536)
    base = torch.load(cache / 'esm3_sm_open_v1.pth', map_location='cpu', weights_only=True, mmap=True)
    backbone.load_state_dict(base, strict=True, assign=True)
    backbone = backbone.float().eval()
    # Rotary nonpersistent buffers were created on meta; regenerate them on CPU.
    for block in backbone.transformer.blocks:
        block.attn.rotary.device = torch.device('cpu')
        block.attn.rotary.reset_parameters()
    ckpt = torch.load(cache / 'dyna1.pt', map_location='cpu', weights_only=True, mmap=True)
    assert all(k.startswith('classifier.') for k in ckpt), list(ckpt)[:5]
    head.load_state_dict({k.removeprefix('classifier.'):v for k,v in ckpt.items()}, strict=True, assign=True)
    return backbone, head.float().eval()

def tokenize(sequence):
    return EsmSequenceTokenizer().encode(sequence, add_special_tokens=False, return_tensors='pt')

@torch.inference_mode()
def trace(backbone, head, sequence):
    tokens = tokenize(sequence)
    result = {'tokens': tokens.int()}
    hooks = [backbone.encoder.register_forward_hook(lambda m,a,o: result.update(embedding=o.detach()))]
    for i, b in enumerate(backbone.transformer.blocks):
        if i < 23:
            hooks.append(b.register_forward_hook(lambda m,a,o,i=i: result.update({f'backbone.{i}':o.detach()})))
    for i, b in enumerate(head.encoder.layers):
        hooks.append(b.register_forward_hook(lambda m,a,o,i=i: result.update({f'head.{i}':o.detach()})))
    try:
        _, hidden = backbone(sequence_tokens=tokens, sequence_id=torch.ones_like(tokens, dtype=torch.bool))
        result['final_embeddings'] = hidden[22]
        result['logits'] = head(hidden[22], torch.zeros_like(tokens, dtype=torch.bool)).reshape(-1)
        result['probs'] = torch.sigmoid(result['logits'] - torch.tensor(__import__('math').log(.05/.95)))
    finally:
        for h in hooks: h.remove()
    return {k:v.detach().cpu().contiguous() for k,v in result.items()}

@torch.inference_mode()
def inference_state(backbone, head):
    # Constant missing tracks are evaluated in their original FP32 addition order.
    t = torch.arange(64).reshape(1,64)
    emb = backbone.encoder(t, torch.full_like(t,4096), torch.ones_like(t).float(),
        torch.zeros_like(t).float(), torch.zeros_like(t), torch.zeros_like(t),
        torch.zeros(1,64,8,dtype=torch.long), torch.zeros(1,64,16,dtype=torch.long))[0]
    sd = {'embedding.weight':emb.contiguous()}
    for i,b in enumerate(backbone.transformer.blocks[:23]):
        for k,v in b.state_dict().items():
            if not k.startswith('geom_attn.'):
                sd[f'backbone.{i}.{k}'] = v.detach().contiguous()
    sd.update({f'head.{k}':v.detach().contiguous() for k,v in head.state_dict().items()})
    return sd
