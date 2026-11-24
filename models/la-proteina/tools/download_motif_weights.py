"""Download the pinned La-Proteina motif checkpoints (AE3 + LD4..LD7) into weights/ on /home/ruh/flame and verify NVIDIA's SHA-256."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = (ROOT / 'weights').resolve()

MOTIF_CHECKPOINTS = {
    'AE3': ('AE3_motif.ckpt', '63b1358c5459e968628094fdc9a2a6a95ac003606fcf7b1a3a21174458a69734', 4100197925),
    'LD4': ('LD4_motif_idx_aa.ckpt', '8bd61de905c62bbe6b26c2669694e413e547c23d7a91515de965371cddd7db7d', 2931464466),
    'LD5': ('LD5_motif_idx_tip.ckpt', 'dd0a888c2fe69715c1fb246339375f464c2d442a46566199191371ddcde9f441', 2928462198),
    'LD6': ('LD6_motif_uidx_aa.ckpt', '9c52db67d2fec4b30134cd8c15132732f188bf1e82d72c9adc3c941086dcf589', 2928525858),
    'LD7': ('LD7_motif_uidx_tip.ckpt', '6b4a5f15dfabbe5594550c9d3042b3770bb9ee11632101b392cdda6a304fb393', 2928525538),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(model: str) -> str:
    if not str(WEIGHTS_DIR).startswith('/home/ruh/flame/'):
        raise RuntimeError(f'Refusing to store multi-GB checkpoints outside /home/ruh/flame/: {WEIGHTS_DIR}')
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    name, sha, size = MOTIF_CHECKPOINTS[model]
    target = WEIGHTS_DIR / name
    if target.exists():
        if target.stat().st_size == size and digest(target) == sha:
            msg = f'{model}: verified existing {target.name} ({size} bytes)'
            print(msg, flush=True)
            return msg
        raise ValueError(f'Existing checkpoint is corrupt: {target}')
    api = f'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/{name.lower()}/1.0/files'
    with urllib.request.urlopen(api, timeout=30) as r:
        meta = json.load(r)
    i = meta['filepath'].index(name)
    pub_sha = base64.b64decode(meta['sha256_base64'][i]).hex()
    if pub_sha != sha:
        raise ValueError(f'{name}: publisher SHA-256 mismatch ({pub_sha} != {sha})')
    partial = WEIGHTS_DIR / f'{name}.partial'
    h = hashlib.sha256()
    count = 0
    print(f'{model}: downloading {name} ({size / 1e9:.2f} GB) -> {target} ...', flush=True)
    with urllib.request.urlopen(meta['urls'][i], timeout=180) as r, partial.open('wb') as f:
        while block := r.read(8 << 20):
            f.write(block)
            h.update(block)
            count += len(block)
    if count != size or h.hexdigest() != sha:
        partial.unlink(missing_ok=True)
        raise ValueError(f'{name}: SHA-256/size mismatch ({count} vs {size}, {h.hexdigest()} vs {sha})')
    partial.replace(target)
    msg = f'{model}: verified {count} bytes, sha256={sha}'
    print(msg, flush=True)
    return msg


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('models', nargs='*', choices=list(MOTIF_CHECKPOINTS))
    p.add_argument('--jobs', type=int, default=5)
    args = p.parse_args()
    selected = args.models or list(MOTIF_CHECKPOINTS)
    with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, len(selected)))) as pool:
        futures = {pool.submit(download, m): m for m in selected}
        for fut in as_completed(futures):
            fut.result()
