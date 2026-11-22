"""Fetch only the official epoch-40 checkpoint; never training/evaluation corpora."""
from pathlib import Path
import hashlib, urllib.request
ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/aqlaboratory/genie2/releases/download/v1.0.0/epoch.40.ckpt'
SHA256 = 'c1f6a4e864f039878b209484bb9780a374858890ad42160ea80a2635d422f859'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def download():
    p=ROOT/'weights/epoch.40.ckpt';p.parent.mkdir(exist_ok=True)
    if p.exists():
        if digest(p)!=SHA256: raise RuntimeError('Existing checkpoint SHA-256 mismatch')
        return p
    tmp=p.with_suffix('.download')
    try:
        urllib.request.urlretrieve(URL,tmp)
        if digest(tmp)!=SHA256: raise RuntimeError('Downloaded checkpoint SHA-256 mismatch')
        tmp.replace(p)
    finally: tmp.unlink(missing_ok=True)
    return p
if __name__=='__main__': print(download(), SHA256)
