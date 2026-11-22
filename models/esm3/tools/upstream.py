"""Strictly load the pinned upstream architecture and original cached checkpoints."""
import os, sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'd49ee4606ea4a95a8bc7388171a6e21fefdf1c26'
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(ROOT / '.cache/upstream'))
import torch
from esm.models.esm3 import ESM3
from esm.models.vqvae import StructureTokenEncoder, StructureTokenDecoder
CACHE = Path('/home/ruh/.cache/huggingface/hub/models--EvolutionaryScale--esm3-sm-open-v1/snapshots/2feda652c51c57f9797fc1b01f8d6181b38fb488/data/weights')
FILES = {'esm':'esm3_sm_open_v1.pth','encoder':'esm3_structure_encoder_v0.pth','decoder':'esm3_structure_decoder_v0.pth'}
CONFIG = dict(d_model=1536, heads=24, layers=48, geom_heads=256, ffn=4096, residual_scale=(48/36)**.5, max_length=1024,
              encoder=dict(d_model=1024,layers=2,geom_heads=128,ffn=4096,knn=16,codes=4096,code_dim=128),
              decoder=dict(d_model=1280,layers=30,heads=20,ffn=3584))
def setup():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def load_models():
    setup()
    with torch.device('meta'):
        models = dict(esm=ESM3(1536,24,256,48,None,None,None,SimpleNamespace(sequence=SimpleNamespace(mask_token_id=32))),
                      encoder=StructureTokenEncoder(1024,1,128,2,128,4096),
                      decoder=StructureTokenDecoder(1280,20,30))
    for name,model in models.items():
        sd = torch.load(CACHE / FILES[name],map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(sd,strict=True,assign=True)
        model.float().eval()
        for mod in model.modules():
            if hasattr(mod,'rotary'):
                mod.rotary.device=torch.device('cpu')
                mod.rotary.reset_parameters()
    from esm.utils.constants.physics import BB_COORDINATES
    models['decoder'].affine_output_projection.bb_local_coords=torch.tensor(BB_COORDINATES).float()
    return models

def state(models):
    return {f'{prefix}.{key}':value.detach() for prefix,model in models.items() for key,value in model.state_dict().items()}

def restore(models,sd):
    for prefix,model in models.items():
        model.load_state_dict({k[len(prefix)+1:]:v for k,v in sd.items() if k.startswith(prefix+'.')},strict=True,assign=True)
