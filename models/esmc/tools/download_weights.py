"""Download immutable upstream code/checkpoint; verify before use. Writes only here."""
import hashlib, tarfile, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
COMMIT = '26b0bc2b771e3e419ea74f445a5f35cc094a1509'
REVISION = '7f10b20ae75017b2dbc884070e03434515709a8d'
ASSETS = {
 'esm-mit.tar.gz': (f'https://api.github.com/repos/Biohub/esm/tarball/{COMMIT}', 'ba9464e99eaca217927a1d4d56c64770f9639fb5dc1d3f6d7e211f4c326f37cc'),
 'esmc_300m_2024_12_v0.pth': (f'https://huggingface.co/biohub/esmc-300m-2024-12/resolve/{REVISION}/data/weights/esmc_300m_2024_12_v0.pth', '323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea'),
}
def digest(path):
 with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 cache=ROOT/'.cache'; cache.mkdir(exist_ok=True)
 for name,(url,sha) in ASSETS.items():
  path=cache/name
  if not path.exists() or digest(path)!=sha:
   tmp=path.with_suffix('.download')
   print('Downloading',name,flush=True)
   with urllib.request.urlopen(url) as src, tmp.open('wb') as out:
    while chunk:=src.read(1024*1024): out.write(chunk)
   assert digest(tmp)==sha, f'SHA256 mismatch: {name}'
   tmp.replace(path)
  assert digest(path)==sha
  print('Verified',name,sha,flush=True)
 # Extract the archive intact; never patch upstream modules.
 source=cache/'esm'
 if not source.exists():
  source.mkdir()
  with tarfile.open(cache/'esm-mit.tar.gz') as tar:
   for member in tar.getmembers():
    member.name='/'.join(member.name.split('/')[1:])
    if member.name: tar.extract(member,source,filter='data')
if __name__=='__main__': main()
