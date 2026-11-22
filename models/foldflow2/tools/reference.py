"""Pinned upstream eager oracle; no tracing or captured operation graphs."""
import pathlib, sys, dataclasses, copy, os
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('GEOMSTATS_BACKEND','pytorch')
sys.path.insert(0,str(ROOT/'.cache/upstream'))
# Python >=3.11 disallows the old upstream dataclass instance defaults.
_original_get_field=dataclasses._get_field
def _compat(cls,name,tp,kw):
    default=getattr(cls,name,None)
    if dataclasses.is_dataclass(default) and not isinstance(default,type):
        setattr(cls,name,dataclasses.field(default_factory=lambda d=default:copy.deepcopy(d)))
    return _original_get_field(cls,name,tp,kw)
dataclasses._get_field=_compat
import torch, numpy as np
from omegaconf import OmegaConf
from foldflow.models.ff2flow.flow_model import FF2Model
from foldflow.models.ff2flow.ff2_dependencies import FF2Dependencies
from foldflow.models.components.sequence.frozen_esm import ESM_REGISTRY
from openfold.utils.rigid_utils import Rigid
import esm

def load_esm():
    data=torch.load(ROOT/'weights/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False)
    model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',data,None)
    # FP32 arithmetic oracle; disable the inference factory's optional half cast.
    model.half=lambda: model
    return model,alphabet

def load_model():
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    ESM_REGISTRY['esm2_650M']=load_esm
    flow=OmegaConf.load(ROOT/'.cache/upstream/runner/config/flow_matcher/default.yaml')
    cfg=OmegaConf.create({'flow_matcher':flow,'model':OmegaConf.load(ROOT/'.cache/upstream/runner/config/model/ff2.yaml')})
    cfg.flow_matcher.so3.inference_scaling=10.0
    deps=FF2Dependencies(cfg)
    # Dependencies require encoder/decoder configurations before building adapters.
    _=deps.bb_encoder; _=deps.bb_decoder
    model=FF2Model.from_ckpt(torch.load(ROOT/'weights/ff2_reft.pth',map_location='cpu',weights_only=False),deps)
    model.float(); model.eval()
    return model

if __name__=='__main__':
    model=load_model()
    print('loaded',len(model.state_dict()),sum(x.numel() for x in model.state_dict().values()),flush=True)
