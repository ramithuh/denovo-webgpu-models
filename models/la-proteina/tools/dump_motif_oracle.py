"""Deterministic PyTorch CPU FP32 + quantized (FP16/INT8) oracles for La-Proteina motif scaffolding.

Covers:
1. AE3 encoder -> decoder round-trip across >= 2 distinct lengths (1prw L=40 multi-segment + 2kl8 L=79 monomer)
   in fp32, fp16, int8, with intermediate layer traces and reconstruction RMSD / seq recovery.
2. LD4 (indexed all-atom), LD5 (indexed tip-atom), LD6 (unindexed all-atom), LD7 (unindexed tip-atom)
   paired with AE3 decoder across >= 2 distinct lengths & multi-step SDE/ODE trajectories in fp32, fp16, int8,
   plus optional longer trajectory (--long-steps) verifying motif atom RMSD preservation & sequence recovery.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from motif_common import (
    COMMIT,
    MOTIF_MODEL_META,
    ROOT,
    UPSTREAM,
    ProductSpaceFlowMatcher,
    checkpoint_cfg_exp,
    compute_motif_rmsd_angstrom,
    extract_full_structure_for_ae,
    extract_motif_condition,
    make_motif_model,
    reload_motif_bundle,
    stats,
)

AE3_CASES = [
    ('1prw_L40', UPSTREAM / 'motif_benchmark_pdb_files/1prw_aa.pdb'),
    ('2kl8_L79', UPSTREAM / 'motif_benchmark_pdb_files/2kl8_aa.pdb'),
]

MOTIF_TASKS = [
    {
        'task': '1ycr_L26',
        'pdb': UPSTREAM / 'motif_benchmark_pdb_files/1ycr_aa.pdb',
        'contig': '8/B19-27/9',
        'motif_only': True,
    },
    {
        'task': '1ycr_L40',
        'pdb': UPSTREAM / 'motif_benchmark_pdb_files/1ycr_aa.pdb',
        'contig': '15/B19-27/16',
        'motif_only': True,
    },
    {
        'task': '5trv_L27',
        'pdb': UPSTREAM / 'motif_benchmark_pdb_files/5trv_aa.pdb',
        'contig': '4/A46-48/1/A50-55/1/A57-58/2/A61-65/4',
        'motif_only': False,
    },
    {
        'task': '1prw_L52',
        'pdb': UPSTREAM / 'motif_benchmark_pdb_files/1prw_aa.pdb',
        'contig': '4/A1-20/4/B1-20/4',
        'motif_only': True,
    },
]


def write_case(case: dict, data: dict) -> dict:
    directory = ROOT / '.cache/fixtures'
    directory.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    tensors = {}
    for name, t in data.items():
        a = t.detach().float().cpu().contiguous().numpy().astype('<f4')
        raw = a.tobytes()
        tensors[name] = {'shape': list(a.shape), 'offset': len(blob), 'length': a.size}
        blob.extend(raw)
    file = f'{case["name"]}.bin'
    (directory / file).write_bytes(blob)
    case.update(
        file=f'.cache/fixtures/{file}',
        byteLength=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(),
        tensors=tensors,
    )
    return case


def run_ae3_case(ae_pair: dict, pdb_path: Path, seed: int, trace: bool = False):
    torch.manual_seed(seed)
    batch = extract_full_structure_for_ae(pdb_path, center_ca=True)
    enc = ae_pair['encoder']
    dec = ae_pair['decoder']

    draws = []
    traces = {}
    orig_randn_like = torch.randn_like

    def record_randn_like(t, *a, **kw):
        eps = orig_randn_like(t, *a, **kw)
        draws.append(eps.clone())
        return eps

    hooks = []
    if trace:
        for name, mod in enc.named_modules():
            if name in ('init_repr_factory', 'cond_factory', 'pair_rep_factory') or (
                name.startswith('transformer_layers.') and name.count('.') == 1
            ):
                hooks.append(
                    mod.register_forward_hook(
                        lambda m, a, out, n=f'enc.{name}': traces.setdefault(n, out.detach().clone())
                    )
                )
        for name, mod in dec.named_modules():
            if name in ('init_repr_factory', 'cond_factory', 'pair_rep_factory') or (
                name.startswith('transformer_layers.') and name.count('.') == 1
            ):
                hooks.append(
                    mod.register_forward_hook(
                        lambda m, a, out, n=f'dec.{name}': traces.setdefault(n, out.detach().clone())
                    )
                )

    with patch('torch.randn_like', record_randn_like):
        enc_out = enc(batch)

    mask = batch['mask']
    ca_nm = batch['coords_nm'][:, :, 1, :] * mask[..., None]
    dec_out = dec({
        'z_latent': enc_out['z_latent'],
        'ca_coors_nm': ca_nm,
        'residue_mask': mask,
        'mask': mask,
    })
    for h in hooks:
        h.remove()

    # Compute reconstruction metrics vs ground truth
    gt_mask = batch['coord_mask'] & mask[..., None]
    err_ang = ((batch['coords_nm'] - dec_out['coors_nm']) * 10.0) * gt_mask[..., None]
    rmsd_ang = (err_ang.square().sum(dim=-1).sum() / gt_mask.sum().clamp_min(1)).sqrt().item()
    seq_rec = (dec_out['aatype_max'] == batch['residue_type']).float().mean().item()

    result = {
        'coords_ang': batch['coords'],
        'coords_nm': batch['coords_nm'],
        'coord_mask': batch['coord_mask'].float(),
        'residue_type': batch['residue_type'].float(),
        'chain_breaks': batch['chain_breaks_per_residue'].float(),
        'chains': batch['chains'].float(),
        'eps': draws[0],
        'mean': enc_out['mean'],
        'log_scale': enc_out['log_scale'],
        'z_latent': enc_out['z_latent'],
        'ca_nm': ca_nm,
        'coords': dec_out['coors_nm'],
        'logits': dec_out['seq_logits'],
        'residue_types': dec_out['aatype_max'].float(),
        'atom_mask': dec_out['atom_mask'].float(),
        'masked_coords': dec_out['coors_nm'] * dec_out['atom_mask'][..., None],
    }
    for k, v in traces.items():
        result[f'trace.{k}'] = v
    meta = {
        'length': int(mask.shape[1]),
        'reconstruction_rmsd_ang': rmsd_ang,
        'seq_recovery': seq_rec,
    }
    return result, meta


def run_motif_ld_case(
    model_name: str,
    ld,
    ae_dec,
    task_spec: dict,
    steps: int,
    seed: int,
    mode: str,
    trace: bool = False,
):
    torch.manual_seed(seed)
    meta_cfg = MOTIF_MODEL_META[model_name]
    cond = extract_motif_condition(
        pdb_path=task_spec['pdb'],
        contig_string=task_spec['contig'],
        motif_only=task_spec['motif_only'],
        atom_selection_mode=meta_cfg['motif_atom_mode'],
        coors_to_nm=True,
        center_motif=True,
    )
    length = cond['length']

    cfg_exp = OmegaConf.create(checkpoint_cfg_exp(model_name))
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.product_flowmatcher.local_latents.dim = 8
    fm = ProductSpaceFlowMatcher(cfg_exp)

    params = OmegaConf.load(UPSTREAM / 'configs/inference_base.yaml').generation.model
    # Motif inference configs set bb_ca.simulation_step_params.center_every_step = False!
    params.bb_ca.simulation_step_params.center_every_step = False
    if mode == 'ode':
        for dm in params:
            params[dm].simulation_step_params.sampling_mode = 'vf'

    batch = {
        'nsamples': 1,
        'nres': length,
        'motif_mask': cond['motif_mask'].clone(),
        'x_motif': cond['x_motif'].clone(),
        'seq_motif_mask': cond['seq_motif_mask'].clone(),
        'seq_motif': cond['seq_motif'].clone(),
        'mask': cond['mask'].clone(),
    }
    # Important: ProductSpaceFlowMatcher.full_simulation squeezes batch dims of size 1 if dim>0!
    # Wait: in generate.py DataLoader(dataset, batch_size=1) wraps GenDataset's [1, L, ...] into [1, 1, L, ...]!
    # So when full_simulation squeezes dim 0 of size 1, [1, 1, L, ...] becomes [1, L, ...]!
    batch_wrapped = {
        k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    draws = []
    states = []
    velocities = []
    traces = {}
    orig_randn = torch.randn

    def record_randn(*a, **kw):
        x = orig_randn(*a, **kw)
        draws.append(x.clone())
        return x

    def predict(b, mode='full'):
        out = ld(b)
        states.append({k: v.clone() for k, v in b['x_t'].items()})
        velocities.append({k: v['v'].clone() for k, v in out.items()})
        return out

    hooks = []
    if trace:
        for name, mod in ld.named_modules():
            if name in ('init_repr_factory', 'cond_factory', 'pair_repr_builder', 'motif_uidx_factory') or (
                name.startswith('transformer_layers.') and name.count('.') == 1
            ):
                def _hook(m, a, out, n=name):
                    val = out[0] if isinstance(out, tuple) else out
                    traces.setdefault(n, val.detach().clone())

                hooks.append(mod.register_forward_hook(_hook))

    with patch('torch.randn', record_randn):
        x, _ = fm.full_simulation(
            batch=batch_wrapped,
            predict_for_sampling=predict,
            nsteps=steps,
            nsamples=1,
            n=length,
            self_cond=True,
            sampling_model_args=params,
            device=torch.device('cpu'),
        )
    for h in hooks:
        h.remove()

    mask = torch.ones((1, length), dtype=torch.bool)
    dec_out = ae_dec({
        'z_latent': x['local_latents'],
        'ca_coors_nm': x['bb_ca'],
        'residue_mask': mask,
        'mask': mask,
    })

    motif_metrics = compute_motif_rmsd_angstrom(
        gen_coords_nm=dec_out['coors_nm'],
        gen_atom_mask=dec_out['atom_mask'],
        gen_aatype=dec_out['aatype_max'],
        motif_mask_raw=cond['motif_mask_raw'],
        x_motif_raw_nm=cond['x_motif_raw'],
        motif_aatype_raw=cond['residue_type_raw'],
        motif_indices=cond['motif_indices'],
        unindexed=(meta_cfg['motif_indexing'] == 'unindexed'),
    )

    result = {
        'motif_mask': cond['motif_mask'].float(),
        'x_motif': cond['x_motif'],
        'seq_motif_mask': cond['seq_motif_mask'].float(),
        'seq_motif': cond['seq_motif'].float(),
        'initial_ca': draws[0],
        'initial_z': draws[1],
        'final_ca': x['bb_ca'],
        'final_z': x['local_latents'],
        'coords': dec_out['coors_nm'],
        'logits': dec_out['seq_logits'],
        'residue_types': dec_out['aatype_max'].float(),
        'atom_mask': dec_out['atom_mask'].float(),
        'masked_coords': dec_out['coors_nm'] * dec_out['atom_mask'][..., None],
    }
    for i, d in enumerate(draws[2:]):
        result[f'draw.{i}'] = d
    for i, (s, v) in enumerate(zip(states, velocities)):
        for k in s:
            result[f'state.{i}.{k}'] = s[k]
            result[f'velocity.{i}.{k}'] = v[k]
    for k, v in traces.items():
        result[f'trace.{k}'] = v

    case_meta = {
        'model': model_name,
        'task': task_spec['task'],
        'pdb': str(task_spec['pdb'].relative_to(UPSTREAM)),
        'contig': task_spec['contig'],
        'motif_only': task_spec['motif_only'],
        'resolved_contig': cond['resolved_contig'],
        'motif_indices': cond['motif_indices'],
        'n_motif_res': cond['n_motif_res'],
        'motif_indexing': meta_cfg['motif_indexing'],
        'motif_atom_mode': meta_cfg['motif_atom_mode'],
        'length': length,
        'steps': steps,
        'seed': seed,
        'mode': mode,
        **motif_metrics,
    }
    return result, case_meta


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['LD4', 'LD5', 'LD6', 'LD7'])
    p.add_argument('--precisions', nargs='+', default=['fp32', 'fp16', 'int8'])
    p.add_argument('--tasks', nargs='+', default=['1ycr_L26', '1prw_L52'])
    p.add_argument('--steps', nargs='+', type=int, default=[1, 3])
    p.add_argument('--modes', nargs='+', default=['sde', 'ode'])
    p.add_argument('--include-ae3', action='store_true', default=True)
    p.add_argument('--no-ae3', action='store_false', dest='include_ae3')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--output', default='.cache/fixtures/motif_oracles.json')
    args = p.parse_args()

    selected_tasks = [t for t in MOTIF_TASKS if t['task'] in args.tasks]
    assert selected_tasks, f'No matching tasks in {args.tasks}'

    manifest = {
        'upstream_commit': COMMIT,
        'torch': torch.__version__,
        'ae3_cases': [],
        'cases': [],
        'quantization_drift': {},
    }
    ae3_refs = {}
    ld_refs = {}

    for precision in args.precisions:
        ae3_state, ae3_cfg = reload_motif_bundle('AE3', precision, with_config=True)
        ae3_pair, _ = make_motif_model('AE3', state=ae3_state, cfg=ae3_cfg)
        del ae3_state
        if args.include_ae3:
            for idx, (label, pdb_path) in enumerate(AE3_CASES):
                key = f'ae3_{label}'
                name = f'motif_{precision}_{key}'
                seed = 4101 + idx
                res, meta = run_ae3_case(ae3_pair, pdb_path, seed=seed, trace=(args.trace and precision == 'fp32'))
                case = write_case({'name': name, 'model': 'AE3', 'precision': precision, 'seed': seed, **meta}, res)
                manifest['ae3_cases'].append(case)
                if precision == 'fp32':
                    ae3_refs[key] = {k: res[k].clone() for k in ('mean', 'log_scale', 'z_latent', 'coords', 'logits')}
                elif key in ae3_refs:
                    manifest['quantization_drift'][name] = {k: stats(res[k], v) for k, v in ae3_refs[key].items()}
                print(
                    f"{name}: L={meta['length']} rmsd={meta['reconstruction_rmsd_ang']:.3f}A seq_rec={meta['seq_recovery']:.3f}",
                    flush=True,
                )

        for model_name in args.models:
            ld_state, ld_cfg = reload_motif_bundle(model_name, precision, with_config=True)
            ld, _ = make_motif_model(model_name, state=ld_state, cfg=ld_cfg)
            del ld_state
            for task_spec in selected_tasks:
                for steps in args.steps:
                    for mode in args.modes:
                        key = f"{model_name.lower()}_{task_spec['task']}_T{steps}_{mode}"
                        name = f'motif_{precision}_{key}'
                        seed = 5100 + sum(ord(c) for c in key)
                        res, meta = run_motif_ld_case(
                            model_name=model_name,
                            ld=ld,
                            ae_dec=ae3_pair['decoder'],
                            task_spec=task_spec,
                            steps=steps,
                            seed=seed,
                            mode=mode,
                            trace=(args.trace and precision == 'fp32' and steps == 1 and mode == 'sde'),
                        )
                        case = write_case({'name': name, 'precision': precision, **meta}, res)
                        manifest['cases'].append(case)
                        if precision == 'fp32':
                            ld_refs[key] = {k: res[k].clone() for k in ('final_ca', 'final_z', 'coords', 'logits')}
                        elif key in ld_refs:
                            manifest['quantization_drift'][name] = {
                                k: stats(res[k], v) for k, v in ld_refs[key].items()
                            }
                        print(
                            f"{name}: L={meta['length']} T={steps} {mode} motifRMSD={meta['motif_rmsd_aligned_ang']:.3f}A seqRec={meta['motif_seq_recovery']:.2f}",
                            flush=True,
                        )
            del ld
        del ae3_pair

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {out_path} ({len(manifest["ae3_cases"])} AE3 cases, {len(manifest["cases"])} LD4-LD7 cases)', flush=True)
