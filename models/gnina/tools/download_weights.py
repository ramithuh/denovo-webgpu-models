"""Fetch and verify the release sources and all default checkpoints."""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / ".cache/upstream"
COMMIT = "97fa6bcc465d62b40d87ccc85d3c53e9921a7cc0"
MOLGRID = "4674cfa0835ae777c0eb9a75562bfb16e28d1bbe"
MODELS = {
    "dense_1.3": "2fc61b1f75d5d79b0874d3f3fdb15c7eb2806e09e22f22d83b4f38ec6787d7b3",
    "dense_1.3_PT_KD_3": "87480e37828480b4c0a3eda41d5ca82e9a8f9e3ad48cbbfe83807f67a6046bc7",
    "crossdock_default2018_KD_4": "9f125b93875c78540558d9909e961e0dce1a38ff35feda5cdf3bb4e09686f68e",
}

def checkpoint(name):
    path = UP / "gnina/gninasrc/lib/models" / (name + ".pt")
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODELS[name]:
        raise ValueError(f"Checkpoint integrity failure: {name}")
    return path

def fetch():
    UP.mkdir(parents=True, exist_ok=True)
    for repo, tag, sha in [("gnina", "v1.3", COMMIT), ("libmolgrid", "v0.5.5", MOLGRID)]:
        path = UP / repo
        if not path.exists():
            subprocess.run(["git", "clone", "--depth=1", "--branch", tag,
                            f"https://github.com/gnina/{repo}", path], check=True)
        if subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip() != sha:
            raise ValueError(f"Incorrect revision: {path}")
    for name in MODELS:
        checkpoint(name)

if __name__ == "__main__":
    fetch()
