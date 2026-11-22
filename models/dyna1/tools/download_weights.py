#!/usr/bin/env python3
"""Download and hash-check the official checkpoints into the ignored local cache."""
import hashlib, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=[
 ('dyna1.pt','https://huggingface.co/gelnesr/Dyna-1/resolve/main/dyna1.pt','f6531b77aaa2593932adb4731278b17c26aabad7a8ffff4da92e8805c558735e'),
 ('esm3_sm_open_v1.pth','https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1/resolve/main/data/weights/esm3_sm_open_v1.pth','5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9'),
]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
if __name__=='__main__':
 cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
 for name,url,sha in FILES:
  p=cache/name
  if not p.exists():
   partial=p.with_suffix('.partial');urllib.request.urlretrieve(url,partial)
   if digest(partial)!=sha:raise ValueError(f'Checkpoint checksum mismatch: {name}')
   partial.replace(p)
  if digest(p)!=sha:raise ValueError(f'Checkpoint checksum mismatch: {name}')
  print(name,sha)
