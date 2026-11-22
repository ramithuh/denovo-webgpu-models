#!/usr/bin/env python3
"""Strict upstream checkpoint -> browser Safetensors bundles. No optimizer tensors.
Shard cap includes the serialized header. Geometry buffers retain their precision.
"""

import argparse
from collections import OrderedDict, Counter
import hashlib
import json
from pathlib import Path
import shutil
import torch
from safetensors.torch import save, load_file
from upstream import ROOT, CHECKPOINT, load

DTYPES = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.int8: "I8",
    torch.int32: "I32",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def entries(model, precision):
    parameters = dict(model.named_parameters())
    for name, original in model.state_dict().items():
        t = original.detach().cpu().contiguous()
        # Only learned matrices are quantized; geometry tables and RBF centers are constants.
        matrix = name in parameters and t.ndim == 2
        if not t.is_floating_point():
            t = t.to(torch.int32)
        else:
            t = t.float()
        rec = {"dtype": DTYPES[t.dtype], "shape": list(t.shape)}
        if precision == "fp16" and matrix:
            t = t.half()
            rec["dtype"] = "F16"
        if precision == "int8" and matrix:
            scale = t.abs().amax(dim=1).clamp_min(1.27e-10) / 127
            t = (t / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            key = name + ".__scale"
            rec.update(
                dtype="I8",
                quant={
                    "scheme": "symmetric_per_row",
                    "axis": 0,
                    "scale_tensor": key,
                    "scale_dtype": "F32",
                    "scale_shape": list(scale.shape),
                },
            )
            yield name, {name: t, key: scale}, rec
        else:
            yield name, {name: t}, rec


def reload(directory):
    m = json.loads((directory / "manifest.json").read_text())
    shards = {s["file"]: load_file(str(directory / s["file"])) for s in m["shards"]}
    result = OrderedDict()
    for n, r in m["tensors"].items():
        t = shards[r["shard"]][n]
        if "quant" in r:
            t = t.float() * shards[r["shard"]][r["quant"]["scale_tensor"]][:, None]
        result[n] = t.float() if t.is_floating_point() else t
    return result


def errors(got, ref):
    a, b = got.double(), ref.double()
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    assert torch.equal(torch.isinf(a), torch.isinf(b))
    mask = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[mask], b[mask]
    return (
        float(((a - b) ** 2).sum()),
        float((b * b).sum()),
        float((a - b).abs().max()) if a.numel() else 0.0,
    )


def export(model, params, checkpoint, directory, precision, cap):
    directory.mkdir(parents=True, exist_ok=True)
    groups = []
    current = {}
    records = []
    # Reserve a generous header budget, then enforce the real serialized cap below.
    payload = 0
    for n, ts, r in entries(model, precision):
        size = sum(t.numel() * t.element_size() for t in ts.values())
        if current and payload + size + len(current) * 256 + 4096 > cap:
            groups.append((current, records))
            current = {}
            records = []
            payload = 0
        current.update(ts)
        records.append((n, r))
        payload += size
    if current:
        groups.append((current, records))
    manifest = {
        "format": "lasermpnn-safetensors-v1",
        "model": "laser-0p1A-nothing-heldout",
        "precision": precision,
        "source": {
            "repo": "polizzilab/LASErMPNN",
            "revision": "e70f2c6d765416f7e29d51bfd6d4e08496438878",
            "checkpoint": checkpoint.name,
            "sha256": sha(checkpoint.read_bytes()),
            "license": "MIT",
        },
        "total_params": sum(p.numel() for p in model.parameters()),
        "num_tensors": len(model.state_dict()),
        "config": {
            "model": params["model_params"],
            "ligand_encoder": model.ligand_encoder_params["model_params"],
            "alphabet": "ARNDCEQGHILKMFPSTWYVX",
        },
        "shards": [],
        "tensors": {},
        "total_bytes": 0,
    }
    for i, (ts, rs) in enumerate(groups):
        raw = save(ts, metadata={"format": manifest["format"], "precision": precision})
        if len(raw) > cap:
            raise ValueError(
                f"Shard exceeds {cap} byte cap; cannot split a logical tensor"
            )
        name = f"shard-{i:03d}.safetensors"
        (directory / name).write_bytes(raw)
        manifest["shards"].append(
            {
                "file": name,
                "bytes": len(raw),
                "sha256": sha(raw),
                "num_tensors": len(rs),
                "num_raw_tensors": len(ts),
            }
        )
        manifest["total_bytes"] += len(raw)
        for n, r in rs:
            manifest["tensors"][n] = {"shard": name, **r}
    manifest["dtype_counts"] = dict(
        Counter(r["dtype"] for r in manifest["tensors"].values())
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    state = reload(directory)
    model.load_state_dict(state, strict=True)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CHECKPOINT)
    ap.add_argument("--outdir", type=Path, default=ROOT / "weights")
    ap.add_argument(
        "--precisions",
        nargs="+",
        choices=["fp32", "fp16", "int8"],
        default=["fp32", "fp16", "int8"],
    )
    ap.add_argument("--max-shard-mb", type=float, default=25.0)
    a = ap.parse_args()
    if not a.ckpt.exists() and a.ckpt == CHECKPOINT:
        a.ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "model_weights" / a.ckpt.name, a.ckpt)
    model, params = load(a.ckpt)
    original = {n: t.clone() for n, t in model.state_dict().items()}
    stats = {}
    parameter_names = set(dict(model.named_parameters()))
    for precision in a.precisions:
        model.load_state_dict(original, strict=True)
        m = export(
            model,
            params,
            a.ckpt,
            a.outdir / precision,
            precision,
            int(a.max_shard_mb * 1_000_000),
        )
        err = ref = mx = learned_err = learned_ref = 0.0
        for n, t in reload(a.outdir / precision).items():
            e, r, x = errors(t, original[n])
            err += e
            ref += r
            mx = max(mx, x)
            if n in parameter_names:
                learned_err += e
                learned_ref += r
        stats[precision] = {
            "bytes": m["total_bytes"],
            "shards": len(m["shards"]),
            "state_relRMS": (err / max(ref, 1e-30)) ** 0.5,
            "weight_relRMS": (learned_err / max(learned_ref, 1e-30)) ** 0.5,
            "weight_maxAbs": mx,
        }
        print(precision, stats[precision], flush=True)
    (a.outdir / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
