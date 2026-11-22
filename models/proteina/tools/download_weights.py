"""Download exactly the pinned NGC checkpoint and verify its SHA-256."""
import hashlib, pathlib, urllib.request, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = 'proteina_v1.2_DFS_200M_notri.ckpt'
URL = 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/proteina_v1.2_dfs_200m_notri/1.0/files?redirect=true&path=' + NAME
SHA256 = 'b5f4c1457fac0e7f54b909245b36b95ddd6bfd314ddd51973a202a3a5a2b6ce9'
def verify(path):
    with open(path, 'rb') as f: actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual != SHA256: raise ValueError(f'Checkpoint SHA-256 mismatch: {actual}')
    return actual
def main():
    path = ROOT / 'weights' / NAME
    path.parent.mkdir(exist_ok=True)
    if path.exists(): verify(path); print(path); return
    tmp = path.with_suffix('.partial')
    try:
        with urllib.request.urlopen(URL) as response, open(tmp, 'wb') as f:
            while chunk := response.read(8 << 20): f.write(chunk)
        verify(tmp); os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    print(path)
if __name__ == '__main__': main()
