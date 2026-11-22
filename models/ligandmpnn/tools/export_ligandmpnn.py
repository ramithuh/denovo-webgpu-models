#!/usr/bin/env python3
"""Export official LigandMPNN checkpoints; no changes to upstream model code."""

import argparse, hashlib, json, sys
from collections import Counter, OrderedDict
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_utils import ProteinMPNN


def load_model(path, sidechains=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(
        k_neighbors=ckpt["num_edges"],
        atom_context_num=ckpt["atom_context_num"],
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=sidechains,
    )
    model = ProteinMPNN(**config).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, config


def reload_bundle(folder):
    m = json.loads((folder / "manifest.json").read_text())
    shards = {s["file"]: load_file(str(folder / s["file"])) for s in m["shards"]}
    result = OrderedDict()
    for k, r in m["tensors"].items():
        t = shards[r["shard"]][k].float()
        if "quant" in r:
            t = t * shards[r["shard"]][r["quant"]["scale_tensor"]][:, None]
        result[k] = t
    return result


def export(model, config, checkpoint, out, precision, cap):
    state = model.state_dict()
    groups = []
    for name, w in state.items():
        w = w.detach().cpu().float().contiguous()
        rec = {"dtype": "F32", "shape": list(w.shape)}
        tensors = {name: w}
        if w.ndim == 2 and precision == "fp16":
            tensors = {name: w.half()}
            rec["dtype"] = "F16"
        elif w.ndim == 2 and precision == "int8":
            scale = (w.abs().amax(1) / 127).clamp(min=1e-12)
            tensors = {
                name: torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8),
                name + ".__scale": scale,
            }
            rec.update(
                dtype="I8",
                quant={
                    "scheme": "symmetric_per_row",
                    "axis": 0,
                    "scale_tensor": name + ".__scale",
                    "scale_dtype": "F32",
                    "scale_shape": list(scale.shape),
                },
            )
        groups.append((name, tensors, rec))
    # Account conservatively for safetensors JSON header, and check actual file sizes.
    batches = []
    batch = []
    size = 0
    for g in groups:
        n = sum(t.numel() * t.element_size() for t in g[1].values()) + 1024 * len(g[1])
        if n > cap:
            raise ValueError("Single logical tensor exceeds shard cap")
        if size + n > cap:
            batches.append(batch)
            batch = []
            size = 0
        batch.append(g)
        size += n
    if batch:
        batches.append(batch)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        format="ligandmpnn-safetensors-v1",
        model=checkpoint.stem,
        precision=precision,
        source={
            "repo": "dauparas/LigandMPNN",
            "license": "MIT",
            "checkpoint": checkpoint.name,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        total_params=sum(t.numel() for t in state.values()),
        num_tensors=len(state),
        config=config,
        shards=[],
        tensors={},
    )
    for i, b in enumerate(batches):
        filename = f"shard-{i:03d}.safetensors"
        flat = {k: t for _, ts, _ in b for k, t in ts.items()}
        save_file(
            flat,
            str(out / filename),
            metadata={"format": manifest["format"], "precision": precision},
        )
        raw = (out / filename).read_bytes()
        assert len(raw) <= cap
        manifest["shards"].append(
            dict(
                file=filename,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                num_tensors=len(b),
                num_raw_tensors=len(flat),
            )
        )
        for k, _, r in b:
            manifest["tensors"][k] = dict(shard=filename, **r)
    manifest["total_bytes"] = sum(s["bytes"] for s in manifest["shards"])
    manifest["dtype_counts"] = dict(
        Counter(r["dtype"] for r in manifest["tensors"].values())
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def rel(a, b):
    return float(
        torch.sqrt(
            ((a.double() - b.double()) ** 2).sum()
            / b.double().square().sum().clamp(min=1e-30)
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--outdir", type=Path, default=ROOT / "weights")
    p.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "fp16", "int8"],
        choices=["fp32", "fp16", "int8"],
    )
    p.add_argument("--max-shard-mb", type=float, default=25)
    args = p.parse_args()
    torch.set_num_threads(2)
    from dump_oracle import build_input

    features, _ = build_input()
    paths = (
        [args.ckpt]
        if args.ckpt
        else [ROOT / ".cache" / f"ligandmpnn_v_32_{v}_25.pt" for v in ["010", "020"]]
    )
    results = {}
    for checkpoint in paths:
        model, config = load_model(checkpoint)
        original = model.state_dict()
        with torch.inference_mode():
            ref = model.score(features, True)
        stats = {}
        for precision in args.precisions:
            dest = args.outdir / checkpoint.stem / precision
            m = export(
                model,
                config,
                checkpoint,
                dest,
                precision,
                int(args.max_shard_mb * 1_000_000),
            )
            sd = reload_bundle(dest)
            quant, _ = load_model(checkpoint)
            quant.load_state_dict(sd, strict=True)
            with torch.inference_mode():
                got = quant.score(features, True)
            stats[precision] = dict(
                bytes=m["total_bytes"],
                shards=len(m["shards"]),
                weight_relRMS=rel(
                    torch.cat([sd[k].flatten() for k in original]),
                    torch.cat([v.flatten() for v in original.values()]),
                ),
                logits_relRMS=rel(got["logits"], ref["logits"]),
                log_probs_relRMS=rel(got["log_probs"], ref["log_probs"]),
            )
        results[checkpoint.stem] = stats
    (ROOT / "fixtures/export_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
