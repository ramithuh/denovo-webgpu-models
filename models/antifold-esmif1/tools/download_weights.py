"""Download only the two official checkpoints; verify before atomic promotion."""
import hashlib, os, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'antifold':('https://github.com/oxpig/AntiFold/releases/download/weights-v1/model.pt','d5c442fa0372c28f4d0026d2f551b6f8ba7e7a127cb6837813a88093ed233e9e'),
 'esmif1':('https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt','be4ba36edec22a9bfaa4946ff6b2815f1f19d8a3d7e0eada8b796d5a0eae9fd4')}
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
if __name__=='__main__':
    (ROOT/'.cache').mkdir(exist_ok=True)
    for name,(url,sha) in SOURCES.items():
        p=ROOT/'.cache'/f'{name}.pt'
        if p.exists():
            if digest(p)!=sha: raise RuntimeError(f'Checksum mismatch: {p}')
        else:
            tmp=p.with_suffix('.download')
            urllib.request.urlretrieve(url,tmp)
            if digest(tmp)!=sha: tmp.unlink(); raise RuntimeError(f'Checksum mismatch: {url}')
            os.replace(tmp,p)
        print(name,sha)
