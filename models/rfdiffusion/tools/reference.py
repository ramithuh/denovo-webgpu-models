"""Pinned upstream import environment; DGL executes its original graph operators."""
import os, sys, pathlib, contextlib, inspect
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ['DGLBACKEND']='pytorch'
os.environ['DGLDEFAULTDIR']=str(ROOT/'.cache/dgl')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
os.environ['XDG_CACHE_HOME']=str(ROOT/'.cache')
sys.path[:0]=[str(ROOT/'.cache/upstream'),str(ROOT/'.cache/upstream/env/SE3Transformer')]
import torch
# NVTX is profiling only, unavailable in CPU-only PyTorch builds.
torch.cuda.nvtx.range=lambda *args,**kwargs:contextlib.nullcontext()
torch.set_num_threads(4)
from rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule

def checkpoint():
    return torch.load(ROOT/'weights/Base_ckpt.pt',map_location='cpu',weights_only=False)

def model(state=None):
    c=checkpoint(); config=c['config_dict']; kwargs={k:v for k,v in config['model'].items() if k in inspect.signature(RoseTTAFoldModule).parameters};kwargs.update(d_t1d=config['preprocess']['d_t1d'],d_t2d=config['preprocess']['d_t2d'],T=config['diffuser']['T'])
    m=RoseTTAFoldModule(**kwargs).eval();m.load_state_dict(state or c['model_state_dict'],strict=True);return m
