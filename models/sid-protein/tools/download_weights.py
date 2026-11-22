"""Download only the released 16-step checkpoint and verify the HF LFS SHA-256."""
from pathlib import Path
import hashlib, urllib.request
ROOT = Path(__file__).resolve().parents[1]
URL = 'https://huggingface.co/LY-Xie/DistilledProteina/resolve/main/16-step-checkpoint.pt'
SHA256 = '57062dbf5b8a22068b65484b3cee389cd99f1b67bd4e2d280c55366ee22d80e7'
SIZE = 764630872

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    dest = ROOT / 'weights/16-step-checkpoint.pt'
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        part = dest.with_suffix('.partial')
        try:
            urllib.request.urlretrieve(URL, part)
            if part.stat().st_size != SIZE or digest(part) != SHA256: raise ValueError('Checkpoint checksum mismatch')
            part.replace(dest)
        finally:
            part.unlink(missing_ok=True)
    assert dest.stat().st_size == SIZE and digest(dest) == SHA256
    print(dest, SHA256)
if __name__ == '__main__': main()
