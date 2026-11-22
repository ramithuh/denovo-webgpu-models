"""Print the CPU output from the pinned upstream SymGen on fixed test points."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


REVISION = "86507b6538f51fce57b5a72477165f03999ed7ae"
SOURCE_SHA256 = "9f9a69890777cba6cc4a54e482693346a65e637c7d5a065ee4658674845383a5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the pinned upstream symmetry.py")
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("The SHA-256 of symmetry.py differs from the reference revision")
    spec = importlib.util.spec_from_file_location("upstream_symmetry", args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Two residues, each with two asymmetric test points, off the rotation axes.
    points = torch.tensor([
        [[1.25, -2.5, 0.75], [3.5, 1.125, -0.25]],
        [[-2.0, 0.625, 1.5], [0.875, -1.25, 2.25]],
    ], dtype=torch.float32)
    expected = {}
    for name in ("c1", "d1", "c3", "d4", "d5"):
        symmetry = module.SymGen(name, recenter=False, radius=0)
        coords = torch.full((symmetry.order * len(points), *points.shape[1:]), 99.0)
        coords[:len(points)] = points
        seq = torch.zeros(len(coords), dtype=torch.long)
        output, _ = symmetry.apply_symmetry(coords, seq)
        expected[name] = output.reshape(symmetry.order, -1).tolist()
    print(json.dumps({
        "upstreamCommit": REVISION,
        "sourceSha256": SOURCE_SHA256,
        "unit": points.flatten().tolist(),
        "expected": expected,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
