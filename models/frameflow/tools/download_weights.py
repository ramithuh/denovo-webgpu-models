"""Download only the official weights archive; extract only the selected checkpoint."""
import hashlib, pathlib, tarfile, urllib.request, shutil
ROOT=pathlib.Path(__file__).resolve().parents[1]
URL='https://zenodo.org/records/12776473/files/weights.tar.gz?download=1'
SHA256='c36f992dbc36682049a9aa06cf204146125fcff22a3e5316339982330942b12e'
CHECKPOINT_SHA256='7d0906e0184e87d1101a2be4c24c530d434e9fac01c1d777d65a63fccd1c958a'
MD5='f08937bbf1bbf4889771e6eee2451c53'
def digest(path,kind='sha256'):
    h=hashlib.new(kind)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    folder=ROOT/'weights';folder.mkdir(exist_ok=True)
    archive=folder/'official.tar.gz'
    checkpoint=folder/'published.ckpt'
    if checkpoint.exists() and digest(checkpoint)==CHECKPOINT_SHA256:
        print('Verified checkpoint',CHECKPOINT_SHA256);return
    if not archive.exists():
        temp=folder/'official.tar.gz.part'
        with urllib.request.urlopen(URL) as r,temp.open('wb') as f:shutil.copyfileobj(r,f)
        if digest(temp)!=SHA256:raise RuntimeError('Archive SHA-256 mismatch')
        temp.replace(archive)
    if digest(archive)!=SHA256 or digest(archive,'md5')!=MD5:raise RuntimeError('Archive checksum mismatch')
    with tarfile.open(archive) as t:
        member=t.getmember('weights/pdb/published.ckpt')
        if not member.isfile():raise RuntimeError('Checkpoint must be a regular file')
        with t.extractfile(member) as source,(folder/'published.ckpt').open('wb') as dest:shutil.copyfileobj(source,dest)
    if digest(checkpoint)!=CHECKPOINT_SHA256:raise RuntimeError('Checkpoint SHA-256 mismatch')
    print('checkpoint SHA-256',CHECKPOINT_SHA256)
if __name__=='__main__':main()
