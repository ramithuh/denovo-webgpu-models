"""Check the cached upstream tree against its verified archive without modifying it."""
import hashlib, importlib.metadata, json, tarfile
from download_weights import ROOT, COMMIT, REVISION, ASSETS, digest

def main():
 for name,(_,sha) in ASSETS.items():
  assert digest(ROOT/'.cache'/name)==sha, f'Asset checksum mismatch: {name}'
 count=0
 with tarfile.open(ROOT/'.cache/esm-mit.tar.gz') as tar:
  for member in tar.getmembers():
   relative='/'.join(member.name.split('/')[1:])
   if member.isfile() and relative.endswith('.py'):
    assert (ROOT/'.cache/esm'/relative).read_bytes()==tar.extractfile(member).read(), f'Upstream source changed: {relative}'
    count+=1
 result=dict(upstream_commit=COMMIT,weight_revision=REVISION,verified_python_files=count,
             assets={name:sha for name,(_,sha) in ASSETS.items()},
             oracle_environment={p:importlib.metadata.version(p) for p in ['torch','numpy','transformers','tokenizers','safetensors']})
 (ROOT/'fixtures/provenance.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
