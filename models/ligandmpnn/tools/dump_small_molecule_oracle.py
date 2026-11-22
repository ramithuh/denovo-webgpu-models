#!/usr/bin/env python3
"""Independent acetazolamide + zinc fixture from RCSB 3HS4 (real coordinates)."""

import json
import torch
from dump_oracle import ROOT
from data_utils import parse_PDB, featurize, get_score
from export_ligandmpnn import load_model


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source = ROOT / ".cache/3HS4.pdb"
    p, *_ = parse_PDB(str(source), parse_all_atoms=True)
    zn = p["Y"][p["Y_t"] == 30]
    distance = ((p["X"][:, 1, None] - zn[None]) ** 2).sum(-1).min(-1).values
    ids = torch.sort(torch.argsort(distance)[:32]).values
    addresses = [str(p["chain_letters"][i]) + str(int(p["R_idx"][i])) for i in ids]
    lines = [
        l
        for l in source.read_text().splitlines()
        if (
            l.startswith("HETATM")
            or l.startswith("ATOM  ")
            and l[21].strip() + l[22:26].strip() in addresses
        )
    ]
    target = ROOT / "fixtures/holo_3HS4.pdb"
    target.write_text("\n".join(lines) + "\nEND\n")
    p, *_ = parse_PDB(str(target), parse_all_atoms=True)
    L = len(p["S"])
    p["chain_mask"] = torch.ones(L)
    p["chain_mask"][:2] = 0
    f = featurize(p, number_of_ligand_atoms=25, model_type="ligand_mpnn")
    f.update(
        batch_size=1,
        temperature=0.3,
        randn=torch.randn(1, L),
        symmetry_residues=[[]],
        symmetry_weights=[[]],
        bias=torch.zeros(1, L, 21),
    )
    model, _ = load_model(ROOT / ".cache/ligandmpnn_v_32_010_25.pt")
    tensors = {}
    for k, v in p.items():
        if torch.is_tensor(v):
            tensors["parsed." + k] = v
    for k, v in f.items():
        if torch.is_tensor(v):
            tensors["input." + k] = v
    with torch.inference_mode():
        for k, v in model.score(f, True).items():
            tensors["score." + k] = v
        uniforms = []
        original = torch.multinomial

        def draw(probs, *a, **kw):
            result = original(probs, *a, **kw)
            cdf = probs.cumsum(-1)
            i = int(result[0, 0])
            uniforms.append((float(cdf[0, i]) + (float(cdf[0, i - 1]) if i else 0)) / 2)
            return result

        torch.multinomial = draw
        try:
            out = model.sample(f)
        finally:
            torch.multinomial = original
        for k, v in out.items():
            tensors["sample." + k] = v
        tensors["uniforms"] = torch.tensor(uniforms)
        for name, mask in [
            ("global", f["mask"] * f["chain_mask"]),
            ("ligand", f["mask_XY"] * f["chain_mask"]),
        ]:
            tensors[name + "_score"] = get_score(out["S"], out["log_probs"], mask)[0]
    meta = dict(
        source="https://www.rcsb.org/structure/3HS4",
        structure=target.name,
        residues=addresses,
        length=L,
        options=dict(fixedPositions=addresses[:2], temperature=0.3),
        tensors={},
    )
    offset = 0
    with (ROOT / "fixtures/oracle_small_molecule.bin").open("wb") as file:
        for k, t in tensors.items():
            floating = t.is_floating_point()
            a = t.detach().numpy().astype("<f4" if floating else "<i4")
            raw = a.tobytes()
            meta["tensors"][k] = dict(
                dtype="F32" if floating else "I32",
                shape=list(a.shape),
                byteOffset=offset,
                byteLength=len(raw),
            )
            file.write(raw)
            offset += len(raw)
    (ROOT / "fixtures/oracle_small_molecule.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(L, addresses, offset, "bytes")


if __name__ == "__main__":
    main()
