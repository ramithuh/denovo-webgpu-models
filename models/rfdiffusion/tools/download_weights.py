#!/usr/bin/env python3
"""Official IPD checkpoint, transported over HTTPS and pinned by SHA-256."""
import hashlib,pathlib,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
URL='https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt'
SHA256='0fcf7d7c32b4848030aca3a051e6768de194616f96ba6c38186351a33bfc6eca'
LENGTH=483616107

def verify(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if path.stat().st_size!=LENGTH or h.hexdigest()!=SHA256:raise ValueError('Base_ckpt.pt size or SHA-256 mismatch')
    return h.hexdigest()

def main():
    path=ROOT/'weights/Base_ckpt.pt';path.parent.mkdir(exist_ok=True)
    if path.exists():print(verify(path));return
    temp=path.with_suffix('.partial')
    with urllib.request.urlopen(URL) as response,temp.open('wb') as out:
        for block in iter(lambda:response.read(8*1024*1024),b''):out.write(block)
    verify(temp);temp.replace(path);print(SHA256)
if __name__=='__main__':main()
