#!/usr/bin/env python3
"""CPU FP32 trace of upstream on its real, protonated 4jnj holo example.
The optional portable categorical sampler uses recorded uniform draws and inverse CDF;
the network, preprocessing, decoding loop and rotamer builder are unmodified upstream.
"""

import argparse
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from upstream import ROOT, load, batch_for
from export_lasermpnn import reload, errors


@contextmanager
def categorical_replay(draws, choices, fixed_choices=None):
    original = torch.distributions.Categorical.sample

    def sample(self, sample_shape=torch.Size()):
        if sample_shape:
            raise ValueError("Only scalar batch samples are supported")
        u = torch.rand(self.probs.shape[:-1])
        idx = (
            (u.unsqueeze(-1) > self.probs.cumsum(-1))
            .sum(-1)
            .clamp_max(self.probs.shape[-1] - 1)
        )
        if fixed_choices is not None:
            idx = fixed_choices[len(choices)].clone()
        draws.append(u.clone())
        choices.append(idx.clone())
        return idx

    torch.distributions.Categorical.sample = sample
    try:
        yield
    finally:
        torch.distributions.Categorical.sample = original


def trace(model, batch, fixed_choices=None):
    tensors = {}
    calls = {}
    hooks = []

    def put(name, value):
        if isinstance(value, torch.Tensor):
            tensors[name] = value.detach().cpu().clone()
        elif hasattr(value, "scalars"):
            put(name + ".scalars", value.scalars)
            put(name + ".vectors", value.vectors)
        elif isinstance(value, (tuple, list)):
            for i, x in enumerate(value):
                put(name + f".{i}", x)

    def hook(name, repeated=False):
        def run(m, args, out):
            if repeated:
                calls.setdefault(name, []).append(out.detach().cpu().clone())
            else:
                put(name, out)

        return run

    modules = dict(model.named_modules())
    names = [
        "ligand_encoder",
        "ligand_encoder_output_gvp",
        "backbone_frame_vec_norm",
        "prot_prot_edge_input_layer",
        "lig_prot_edge_input_layer",
    ]
    names += [f"ligand_encoder.gat_layers.{i}" for i in range(3)]
    names += [f"protein_encoder_layers.{i}" for i in range(3)]
    for name in names:
        hooks.append(modules[name].register_forward_hook(hook(name)))
    for name in (
        ["sequence_output_layer"]
        + [f"chi_prediction_layers.{i}" for i in range(4)]
        + [f"chi_offset_prediction_layers.{i}" for i in range(4)]
    ):
        hooks.append(modules[name].register_forward_hook(hook(name, True)))
    draws = []
    choices = []
    try:
        with torch.inference_mode(), categorical_replay(draws, choices, fixed_choices):
            out = model.sample(
                batch,
                sequence_sample_temperature=0.1,
                chi_angle_sample_temperature=0.1,
                disable_pbar=True,
            )
            for f in fields(out):
                put("output." + f.name, getattr(out, f.name))
            seq_lp = (
                out.sequence_logits.log_softmax(-1)
                .gather(1, out.sampled_sequence_indices[:, None])
                .squeeze(1)
            )
            sampled_bins = torch.zeros((batch.num_residues, 4), dtype=torch.long)
            for step, node in enumerate(batch.decoding_order[0].long()):
                sampled_bins[node] = torch.cat(choices[step * 5 + 1 : step * 5 + 5])
            chi_lp = (
                out.chi_logits.log_softmax(-1)
                .gather(-1, sampled_bins[:, :, None])
                .squeeze(-1)
            )
            chi_lp[out.sampled_chi_degrees.isnan()] = 0
            put("output.sequence_log_probs", seq_lp)
            put("output.chi_log_probs", chi_lp)
            put("output.total_log_likelihood", seq_lp.sum() + chi_lp.sum())
            coords = model.rotamer_builder.build_rotamers(
                batch.backbone_coords,
                out.sampled_chi_degrees,
                out.sampled_sequence_indices,
                add_nonrotatable_hydrogens=True,
            )
            put("output.coords", coords)
            put(
                "output.nh_coords",
                model.rotamer_builder.impute_backbone_nh_coords(
                    coords.float(),
                    out.sampled_sequence_indices,
                    batch.phi_psi_angles[:, 0:1],
                ),
            )
            put(
                "output.cleaned_coords",
                model.rotamer_builder.cleanup_titratable_hydrogens(
                    coords.float().clone(),
                    out.sampled_sequence_indices,
                    tensors["output.nh_coords"],
                    batch,
                    model.hbond_network_detector,
                ),
            )
    finally:
        for h in hooks:
            h.remove()
    for name, values in calls.items():
        put(name, torch.cat([v.reshape(1, -1) for v in values], 0))
    put("rng.categorical_uniforms", torch.stack(draws))
    put("rng.categorical_indices", torch.stack(choices))
    return tensors, choices


def save(tensors, meta, prefix):
    raw = bytearray()
    records = {}
    for n, t in tensors.items():
        dtype = "<f4" if t.is_floating_point() else "<i4"
        a = t.numpy().astype(dtype)
        b = a.tobytes()
        records[n] = {
            "dtype": "F32" if dtype == "<f4" else "I32",
            "shape": list(a.shape),
            "byteOffset": len(raw),
            "byteLength": len(b),
        }
        raw.extend(b)
    meta.update(tensors=records, binary_sha256=hashlib.sha256(raw).hexdigest())
    prefix.with_suffix(".bin").write_bytes(raw)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            meta,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-quantization", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(42)
    model, params = load()
    data, batch = batch_for(model, params)
    original = {n: t.clone() for n, t in model.state_dict().items()}
    inputs = {}
    for prefix, obj in [("input", batch), ("ligand", batch.ligand_data)]:
        for k, v in vars(obj).items():
            if isinstance(v, torch.Tensor):
                inputs[prefix + "." + k] = v
    inputs["rng.state_before_sampling"] = torch.get_rng_state().to(torch.int32)
    tensors, choices = trace(model, batch)
    meta = {
        "format": "lasermpnn-oracle-v1",
        "seed": 42,
        "torch": torch.__version__,
        "length": batch.num_residues,
        "ligand_atoms": len(batch.ligand_data.lig_coords),
        "example": "example_pdbs/4jnj-1_prot.pdb",
        "sequenceTemperature": 0.1,
        "chiTemperature": 0.1,
        "sampling": "recorded uniforms, inverse CDF; upstream network and decode loop unchanged",
        "residues": [vars(r) for r in data.residue_identifiers],
    }
    save({**inputs, **tensors}, meta, ROOT / "fixtures/oracle_lasermpnn")
    # Generate browser geometry/element vocabulary from upstream authoritative constants.
    from LASErMPNN.utils import constants as c

    constants = {
        "alphabet": "ARNDCEQGHILKMFPSTWYVX",
        "residueNames": [c.aa_idx_to_long[i] for i in range(21)],
        "atomNames": [
            c.hydrogen_extended_dataset_atom_order[c.aa_idx_to_short[i]]
            for i in range(20)
        ],
        "elements": c.atomic_number_to_atom,
        "maxAtoms": c.MAX_NUM_PROTONATED_RESIDUE_ATOMS,
        "maxHeavyAtoms": c.MAX_NUM_RESIDUE_ATOMS,
    }
    (ROOT / "fixtures/constants.json").write_text(
        json.dumps(constants, indent=2) + "\n"
    )
    print(
        "Oracle:",
        meta["length"],
        "residues,",
        meta["ligand_atoms"],
        "ligand atoms,",
        len(inputs) + len(tensors),
        "tensors",
        flush=True,
    )
    if not a.skip_quantization:
        metrics = json.loads((ROOT / "weights/metrics.json").read_text())
        for p in ["fp32", "fp16", "int8"]:
            model.load_state_dict(reload(ROOT / "weights" / p), strict=True)
            torch.manual_seed(42)
            got, _ = trace(model, batch, choices)
            metrics[p]["conditional_forward"] = {}
            for n in [
                "ligand_encoder.scalars",
                "protein_encoder_layers.2.0.scalars",
                "sequence_output_layer",
                *[f"chi_prediction_layers.{i}" for i in range(4)],
                "output.sampled_chi_degrees",
                "output.coords",
            ]:
                e, r, m = errors(got[n], tensors[n])
                metrics[p]["conditional_forward"][n] = {
                    "relRMS": (e / max(r, 1e-30)) ** 0.5,
                    "maxAbs": m,
                }
            print(p, metrics[p]["conditional_forward"], flush=True)
            if p == "fp16":
                keys = [
                    "sequence_output_layer",
                    *[f"chi_prediction_layers.{i}" for i in range(4)],
                    "output.sampled_chi_degrees",
                    "output.coords",
                    "output.sampled_sequence_indices",
                ]
                save(
                    {k: got[k] for k in keys},
                    {
                        "format": "lasermpnn-oracle-v1",
                        "precision": p,
                        "sampling": "FP32 categorical choices replayed; continuous offsets recomputed",
                    },
                    ROOT / "fixtures/oracle_lasermpnn_fp16",
                )
        (ROOT / "weights/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (ROOT / "fixtures/export_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
