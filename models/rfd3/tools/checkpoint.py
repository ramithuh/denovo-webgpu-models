"""Pinned, eager FP32 reference model. Used only by offline development tools."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.cache/upstream/foundry/models/rfd3'
COMMIT = 'a13b299a05897b951ea28b65e977eebb984380bc'
DEFAULT_CHECKPOINT = '/home/ruh/.foundry/checkpoints/rfd3_latest.ckpt'
# Prefer the pinned source over any installed editable checkout.
sys.path.insert(0, str(ROOT / '.cache/python-deps'))
sys.path.insert(0, str(UPSTREAM / 'src'))
os.environ['RFD3_LOW_MEMORY_MODE'] = '0'


def load_checkpoint(path=DEFAULT_CHECKPOINT):
    import torch
    import hydra
    from omegaconf import OmegaConf
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    config = OmegaConf.to_container(checkpoint['train_cfg'].model.net, resolve=True)
    import rfd3.model.RFD3_diffusion_module  # import-time constants require a real device
    # Instantiate architecture on meta: no redundant allocation or randomized weight copy.
    with torch.device('meta'):
        net = hydra.utils.instantiate(OmegaConf.create(config), _recursive_=False)
    state = {k.removeprefix('shadow.'): v for k, v in checkpoint['model'].items() if k.startswith('shadow.')}
    rms = state.pop('use_rmsnorm')
    if not bool(rms):
        raise ValueError('Checkpoint requires LayerNorm; exporter expects RMSNorm')
    net.load_state_dict(state, strict=True, assign=True)
    net.eval()
    return checkpoint, net, config
