"""Download the published archive, verify it, and extract regular files safely.

The publisher supplies MD5. Initial --bootstrap verifies that digest and records
SHA-256; subsequent runs require the committed SHA-256 lock as well.
"""
import argparse
import concurrent.futures
import hashlib
import json
import tarfile
import time
import urllib.request
from pathlib import Path
from environment import ROOT

URL = 'https://zenodo.org/records/16817230/files/protpardelle-1c.tar.gz?download=1'
SIZE = 4294232741
MD5 = '59fc043a9d981d6c74baa2d051ea164a'
ARCHIVE = ROOT / '.cache/protpardelle-1c.tar.gz'
LOCK = ROOT / 'tools/weights.lock.json'


def download(workers):
    chunk = 256 * 1024 * 1024
    parts = ROOT / '.cache/download-parts'
    parts.mkdir(exist_ok=True)
    def part(i):
        start, end = i * chunk, min(SIZE, (i + 1) * chunk) - 1
        dest = parts / f'{i:03}.part'
        if dest.exists() and dest.stat().st_size == end - start + 1:
            return dest
        for attempt in range(6):
            try:
                req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=120) as response, dest.open('wb') as out:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('Server did not honor exact byte range')
                    while data := response.read(1024 * 1024):
                        out.write(data)
                if dest.stat().st_size != end - start + 1:
                    raise ValueError('Truncated range')
                print(f'downloaded range {i}', flush=True)
                return dest
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(45, 5 * (attempt + 1)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(part, range((SIZE + chunk - 1) // chunk)))
    tmp = ARCHIVE.with_suffix('.tmp')
    with tmp.open('wb') as out:
        for path in paths:
            with path.open('rb') as inp:
                while data := inp.read(8 * 1024 * 1024):
                    out.write(data)
    tmp.replace(ARCHIVE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if not LOCK.exists() and not args.bootstrap:
        raise RuntimeError('Missing SHA-256 lock; bootstrap only from publisher-verified archive')
    if not ARCHIVE.exists() or ARCHIVE.stat().st_size != SIZE:
        download(args.workers)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with ARCHIVE.open('rb') as inp:
        while data := inp.read(8 * 1024 * 1024):
            sha.update(data)
            md5.update(data)
    if md5.hexdigest() != MD5:
        raise ValueError('Publisher MD5 mismatch')
    record = {'url': URL, 'byteLength': SIZE, 'publisher_md5': MD5, 'sha256': sha.hexdigest(), 'license': 'CC-BY-4.0'}
    if LOCK.exists():
        if json.loads(LOCK.read_text())['sha256'] != record['sha256']:
            raise ValueError('Pinned SHA-256 mismatch')
    else:
        LOCK.write_text(json.dumps(record, indent=2) + '\n')
    with tarfile.open(ARCHIVE) as archive:
        for entry in archive:
            relative = Path(entry.name)
            if relative.parts[:2] != ('protpardelle-1c', 'model_params'):
                continue
            target = ROOT / '.cache' / Path(*relative.parts[1:])
            if not target.resolve().is_relative_to(ROOT / '.cache/model_params'):
                raise ValueError('Archive path escape')
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif entry.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(entry) as inp, target.open('wb') as out:
                    while data := inp.read(8 * 1024 * 1024):
                        out.write(data)
            else:
                raise ValueError('Archive contains a non-regular file')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
