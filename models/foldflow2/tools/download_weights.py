#!/usr/bin/env python3
"""Download only FF2 ReFT and its required frozen ESM2 encoder; verify pinned hashes."""
import hashlib,pathlib,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
ASSETS={
 'esm2_t33_650M_UR50D.pt':('https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt','ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c'),
 'ff2_reft.pth':('https://github.com/DreamFold/FoldFlow/releases/download/0.2.0/ff2_reft.pth','73f6d969c6757f4864d19c7a942d698ac5334de3034fcb40a0a9423d2057b902'),
}
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while b:=f.read(8<<20):h.update(b)
 return h.hexdigest()
def main():
 for name,(url,sha) in ASSETS.items():
  p=ROOT/'weights'/name;p.parent.mkdir(exist_ok=True)
  if not p.exists():
   tmp=p.with_suffix('.download');urllib.request.urlretrieve(url,tmp)
   if digest(tmp)!=sha:tmp.unlink();raise RuntimeError(f'SHA256 failure: {name}')
   tmp.replace(p)
  if digest(p)!=sha:raise RuntimeError(f'SHA256 failure: {name}')
  print(name,sha)
if __name__=='__main__':main()
