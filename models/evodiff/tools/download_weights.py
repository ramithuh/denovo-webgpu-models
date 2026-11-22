"""Fetch the exact Zenodo OA_DM_38M archive; verify both published MD5 and SHA256."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / '.cache/oaar-38M.tar'
URL = 'https://zenodo.org/records/8045076/files/oaar-38M.tar?download=1'
MD5 = '7d8959d06cc4373c220413ef00f99fab'
SHA256 = 'cad6774580b700f563bdc63227c336351e8c0c7d0a89014122aa2dce4f768aa0'

def verify(path):
    md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5.update(chunk)
            sha.update(chunk)
    if path.stat().st_size != 454881175 or md5.hexdigest() != MD5 or sha.hexdigest() != SHA256:
        raise ValueError('Checkpoint size/checksum mismatch')

def download():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PATH.exists():
        partial = PATH.with_suffix('.partial')
        urllib.request.urlretrieve(URL, partial)
        verify(partial)
        partial.replace(PATH)
    verify(PATH)
    return PATH

if __name__ == '__main__':
    print(download())
