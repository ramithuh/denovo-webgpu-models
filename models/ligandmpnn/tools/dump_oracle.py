#!/usr/bin/env python3
"""Deterministic, unmodified upstream FP32 traces on a real Zn/DNA holo structure.
A recorded torch.multinomial result and an interior CDF uniform permit exact JS
replay without pretending that JS implements PyTorch's random-number generator.
"""

import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import parse_PDB, featurize, get_score
from export_ligandmpnn import load_model


def build_input():
    source = ROOT / "inputs/1BC8.pdb"
    p, *_ = parse_PDB(str(source), parse_all_atoms=True)
    # Keep 32 authentic residues closest to either zinc, in original file order.
    zn = p["Y"][p["Y_t"] == 30]
    d = ((p["X"][:, 1, None] - zn[None]) ** 2).sum(-1).min(-1).values
    ids = torch.sort(torch.argsort(d)[:32]).values
    addresses = [str(p["chain_letters"][i]) + str(int(p["R_idx"][i])) for i in ids]
    keep = set(addresses)
    lines = [
        l
        for l in source.read_text().splitlines()
        if (
            l.startswith(("ATOM  ", "HETATM"))
            and (
                l[21].strip() + l[22:26].strip() in keep
                or l[17:20].strip()
                not in {
                    "ALA",
                    "ARG",
                    "ASN",
                    "ASP",
                    "CYS",
                    "GLN",
                    "GLU",
                    "GLY",
                    "HIS",
                    "ILE",
                    "LEU",
                    "LYS",
                    "MET",
                    "PHE",
                    "PRO",
                    "SER",
                    "THR",
                    "TRP",
                    "TYR",
                    "VAL",
                }
            )
        )
    ]
    crop = ROOT / "fixtures/holo_1BC8.pdb"
    crop.write_text("\n".join(lines) + "\nEND\n")
    p, *_ = parse_PDB(str(crop), parse_all_atoms=True)
    L = len(p["S"])
    p["chain_mask"] = torch.ones(L)
    p["chain_mask"][:2] = 0
    f = featurize(p, number_of_ligand_atoms=25, model_type="ligand_mpnn")
    g = torch.Generator().manual_seed(42)
    f.update(
        batch_size=1,
        temperature=0.3,
        randn=torch.randn(1, L, generator=g),
        symmetry_residues=[[]],
        symmetry_weights=[[]],
        bias=torch.zeros(1, L, 21),
    )
    f["bias"][:, :, 1] = -1e8
    f["bias"][:, :, 0] = 0.2
    return f, dict(
        source="inputs/1BC8.pdb",
        structure="holo_1BC8.pdb",
        length=L,
        residues=addresses,
        options=dict(
            fixedPositions=addresses[:2],
            designChains=["C"],
            omitAAs="C",
            biasAA={"A": 0.2},
            temperature=0.3,
        ),
        elements=sorted(set(p["Y_t"].tolist())),
        parsed=p,
    )


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    f, meta = build_input()
    parsed = meta.pop("parsed")
    tensors = {}

    def put(name, t):
        if torch.is_tensor(t):
            tensors[name] = t.detach().cpu().clone()

    for k, v in parsed.items():
        put("parsed." + k, v)
    for k, v in f.items():
        put("input." + k, v)
    n, ca, c = parsed["X"][:, 0], parsed["X"][:, 1], parsed["X"][:, 2]
    b, d = ca - n, c - ca
    put(
        "parsed.CB",
        -0.58273431 * torch.cross(b, d, dim=-1) + 0.56802827 * b - 0.54067466 * d + ca,
    )
    model, _ = load_model(ROOT / ".cache/ligandmpnn_v_32_010_25.pt")
    meta["periodic_table"] = model.features.periodic_table_features.tolist()
    meta["side_chain_atom_types"] = model.features.side_chain_atom_types.tolist()
    meta["torch_version"] = torch.__version__
    meta["upstream_commit"] = "26ec57ac976ade5379920dbd43c7f97a91cf82de"
    prefix = "score"
    calls = {}
    handles = []
    capturing = True

    def hook(name):
        def h(mod, args, out):
            if not capturing:
                return
            n = calls.get(name, 0)
            calls[name] = n + 1
            if isinstance(out, tuple):
                names = (
                    ["V", "E", "E_idx", "Y_nodes", "Y_edges", "Y_m"]
                    if name == "features"
                    else ["h_V", "h_E"]
                )
                for key, v in zip(names, out):
                    put(f"{prefix}.{name}.{n}.{key}", v)
            else:
                put(f"{prefix}.{name}.{n}", out)

        return h

    for name, mod in model.named_modules():
        if (
            name == "features"
            or name in ["W_out", "V_C_norm"]
            or name.startswith(
                (
                    "encoder_layers.",
                    "decoder_layers.",
                    "context_encoder_layers.",
                    "y_context_encoder_layers.",
                )
            )
            and name.count(".") == 1
        ):
            handles.append(mod.register_forward_hook(hook(name)))
    # Raw geometry inputs to learned projections, including all RBFs and angles.
    for name in ["edge_embedding", "node_project_down", "y_edges", "y_nodes"]:
        handles.append(
            getattr(model.features, name).register_forward_pre_hook(
                lambda mod, args, n=name: put(f"{prefix}.geometry.{n}", args[0])
            )
        )
    with torch.inference_mode():
        out = model.score(f, True)
        for k, v in out.items():
            put("score.output." + k, v)
        capturing = False
        h, e, idx = model.encode(f)
        put("score.encoded.h_V", h)
        put("score.encoded.h_E", e)
        capturing = True
        for case, ties, side in [
            ("sample", [[]], False),
            ("tied", [[2, 3], [4, 5]], False),
            ("sidechain", [[]], True),
            ("no_context", [[]], False),
        ]:
            prefix = case
            calls.clear()
            model.features.use_side_chains = side
            f["symmetry_residues"] = ties
            f["symmetry_weights"] = [[1 / max(1, len(g))] * len(g) for g in ties]
            if case == "no_context":
                f["Y_m"] = torch.zeros_like(f["Y_m"])
            original = torch.multinomial
            draws = []
            uniforms = []

            def record(probs, *args, **kwargs):
                chosen = original(probs, *args, **kwargs)
                draws.append(chosen)
                cdf = probs.cumsum(-1)
                ix = int(chosen[0, 0])
                lo = float(cdf[0, ix - 1]) if ix else 0
                uniforms.append((lo + float(cdf[0, ix])) / 2)
                return chosen

            torch.multinomial = record
            try:
                out = model.sample(f)
            finally:
                torch.multinomial = original
            for k, v in out.items():
                put(case + ".output." + k, v)
            put(case + ".uniforms", torch.tensor(uniforms))
            put(case + ".draws", torch.cat(draws))
            for name, mask in [
                ("global", f["mask"] * f["chain_mask"]),
                ("ligand", f["mask_XY"] * f["chain_mask"]),
            ]:
                put(
                    case + "." + name + "_score",
                    get_score(out["S"], out["log_probs"], mask)[0],
                )
    for h in handles:
        h.remove()
    meta["tensors"] = {}
    offset = 0
    with (ROOT / "fixtures/oracle_ligandmpnn.bin").open("wb") as file:
        for name, t in tensors.items():
            dtype = "<f4" if t.is_floating_point() else "<i4"
            a = t.numpy().astype(dtype)
            raw = a.tobytes()
            meta["tensors"][name] = dict(
                dtype="F32" if dtype == "<f4" else "I32",
                shape=list(a.shape),
                byteOffset=offset,
                byteLength=len(raw),
            )
            file.write(raw)
            offset += len(raw)
    (ROOT / "fixtures/oracle_ligandmpnn.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    # Portable architecture constants, not learned weights.
    target = ROOT / "shared/ligandmpnn"
    target.mkdir(parents=True, exist_ok=True)
    (target / "constants.js").write_text(
        "export const PERIODIC_TABLE = "
        + json.dumps(meta["periodic_table"])
        + ";\nexport const SIDE_CHAIN_TYPES = "
        + json.dumps(meta["side_chain_atom_types"])
        + ";\n"
    )
    print(
        f"{len(tensors)} tensors, {offset:,} bytes; residues {meta['residues']}; elements {meta['elements']}"
    )


if __name__ == "__main__":
    main()
