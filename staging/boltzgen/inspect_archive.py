"""Inspect the ZIP index of a published BoltzGen checkpoint, without loading any tensors."""

import argparse
import gzip
import hashlib
import io
import json
import re
import struct
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
RELEASE = json.loads((ROOT / "release.json").read_text())
EOCD = struct.Struct("<4s4H2IH")
MAX_INDEX_BYTES = 2 * 1024 * 1024


def read_range(url, start, end, size):
    # The query also distinguishes differently ranged responses in HTTP caches.
    request = Request(f"{url}?bytes={start}-{end}", headers={
        "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
    })
    with urlopen(request, timeout=30) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
            raise ValueError("Expected a partial response matching the pinned object size and requested range")
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError("Incomplete or oversized range response")
    return data


def fetch_index(name, size):
    url = f"https://huggingface.co/boltzgen/boltzgen-1/resolve/{RELEASE['revision']}/{name}.ckpt"
    footer = read_range(url, size - EOCD.size, size - 1, size)
    signature, disk, central_disk, on_disk, entries, central_size, start, comment = EOCD.unpack(footer)
    if (signature != b'PK\x05\x06' or disk or central_disk or comment or entries != on_disk
            or not 0 < central_size <= size - start <= MAX_INDEX_BYTES):
        raise ValueError("Unexpected ZIP footer for this release")
    return read_range(url, start, size - 1, size)


class ArchiveIndex(io.BytesIO):
    """Expose the index at its original offset for ZipFile's ZIP64 locator."""

    def __init__(self, index, size):
        super().__init__(index)
        self.offset = size - len(index)

    def seek(self, offset, whence=0):
        return self.offset + super().seek(offset - self.offset if whence == 0 else offset, whence)

    def tell(self):
        return self.offset + super().tell()


def summarize(index, size):
    if not index.startswith(b"PK\x01\x02"):
        raise ValueError("Expected an isolated ZIP central directory")
    with ZipFile(ArchiveIndex(index, size)) as archive:
        entries = archive.infolist()
    storages = [item for item in entries if re.fullmatch(r"[^/]+/data/[0-9]+", item.filename)]
    if not storages:
        raise ValueError("No numbered PyTorch storage records")
    return {
        "entries": len(entries),
        "storages": len(storages),
        "storage_bytes": sum(item.file_size for item in storages),
        "largest_storage_bytes": max(item.file_size for item in storages),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true", help="check the pinned upstream indexes over HTTP")
    args = parser.parse_args()
    for name, expected in RELEASE["checkpoints"].items():
        index = (fetch_index(name, expected["file_bytes"]) if args.remote
                 else gzip.decompress((ROOT / f"{name}.index.gz").read_bytes()))
        if hashlib.sha256(index).hexdigest() != expected["index_sha256"]:
            raise ValueError(f"{name}: archived index digest differs from the reference")
        measured = summarize(index, expected["file_bytes"])
        if measured != expected["layout"]:
            raise ValueError(f"{name}: unexpected storage layout {measured}")
        print(f"{name}: {measured['storages']:,} storages, {measured['storage_bytes']:,} bytes, "
              f"largest {measured['largest_storage_bytes']:,} bytes")


if __name__ == "__main__":
    main()
