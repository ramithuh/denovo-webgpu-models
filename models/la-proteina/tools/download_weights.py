"""Download only the pinned LD1 and AE1 resources; verify NVIDIA's SHA-256."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    'LD1': ('LD1_ucond_notri_512.ckpt', '6d9ec013f73be6bc305876a02f2aa5dbf3183a938332bac9262aa30f8c963f66', 2926616286),
    'AE1': ('AE1_ucond_512.ckpt', '898da17022cdeaaaea7caace41c8b6fe7bfcb78be4876b0113127eb9bb1527e6', 4100184649),
}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def download(model):
    name, sha, size = CHECKPOINTS[model]
    folder = ROOT / 'weights'
    folder.mkdir(exist_ok=True)
    target = folder / name
    if target.exists():
        if target.stat().st_size == size and digest(target) == sha:
            print(f'{model}: verified existing {target.name}', flush=True)
            return
        raise ValueError(f'Existing checkpoint is corrupt: {target}')
    api = f'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/{name.lower()}/1.0/files'
    with urllib.request.urlopen(api) as r: meta = json.load(r)
    i = meta['filepath'].index(name)
    assert base64.b64decode(meta['sha256_base64'][i]).hex() == sha
    partial = target.with_suffix('.ckpt.partial')
    h = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(meta['urls'][i], timeout=120) as r, partial.open('wb') as f:
        while block := r.read(8 << 20):
            f.write(block)
            h.update(block)
            count += len(block)
    if count != size or h.hexdigest() != sha:
        raise ValueError(f'{name}: SHA-256/size mismatch')
    partial.replace(target)
    print(f'{model}: verified {count} bytes, sha256={sha}', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs='*', choices=['LD1', 'AE1'], default=['LD1','AE1'])
    for model in p.parse_args().models: download(model)
