#!/usr/bin/env python3
"""Fetch pinned official checkpoints from upstream get_model_params.sh URLs."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

CHECKPOINTS = {
    "ligandmpnn_v_32_010_25.pt": "161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970",
    "ligandmpnn_v_32_020_25.pt": "42caaa5cdb380867d2b31c30de4ab53ea89171279b3a0732ddf94bc4d3cb6981",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--outdir", type=Path, default=Path(__file__).resolve().parents[1] / ".cache"
    )
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, expected in CHECKPOINTS.items():
        target = args.outdir / name
        if (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest() == expected
        ):
            print(name, "verified cached checkpoint")
            continue
        with urlopen(
            "https://files.ipd.uw.edu/pub/ligandmpnn/" + name, timeout=120
        ) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Official checkpoint checksum changed: " + name)
        temporary = target.with_suffix(".partial")
        temporary.write_bytes(data)
        temporary.replace(target)
        print(name, len(data), "bytes, SHA256 verified")


if __name__ == "__main__":
    main()
