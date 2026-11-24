"""Shared utilities for La-Proteina motif scaffolding (AE3 + LD4..LD7) export & oracle generation.

Imports the pinned upstream modules directly without replacing any neural or flow-matching arithmetic.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sys
import types
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.cache/upstream'
COMMIT = 'cde5de3ead6e4d76f367da6dc5174be9913ef6ca'
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

torch.set_num_threads(8)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False


def _unused(*args, **kwargs):
    raise RuntimeError('Unused training/conditional dependency entered motif oracle')


sys.modules.setdefault('torch_scatter', types.SimpleNamespace(scatter_mean=_unused))
sys.modules.setdefault('lightning', types.SimpleNamespace(LightningModule=torch.nn.Module))

from omegaconf import OmegaConf  # noqa: E402
from openfold.np import residue_constants as rc  # noqa: E402
from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher  # noqa: E402
from proteinfoundation.nn.feature_factory import FeatureFactory, XmotifBulkTipSeqFeat  # noqa: E402
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer  # noqa: E402
from proteinfoundation.nn.local_latents_transformer_unindexed import LocalLatentsTransformerMotifUidx  # noqa: E402
from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer  # noqa: E402
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer  # noqa: E402
from proteinfoundation.utils.align_utils import kabsch_align_ind, mean_w_mask  # noqa: E402

from download_motif_weights import MOTIF_CHECKPOINTS  # noqa: E402

# Exact SIDECHAIN_TIP_ATOMS table from proteinfoundation/utils/constants.py (without graphein import).
SIDECHAIN_TIP_ATOMS = {
    'ALA': ['CA', 'CB'],
    'ARG': ['CD', 'CZ', 'NE', 'NH1', 'NH2'],
    'ASP': ['CB', 'CG', 'OD1', 'OD2'],
    'ASN': ['CB', 'CG', 'ND2', 'OD1'],
    'CYS': ['CA', 'CB', 'SG'],
    'GLU': ['CG', 'CD', 'OE1', 'OE2'],
    'GLN': ['CG', 'CD', 'NE2', 'OE1'],
    'GLY': [],
    'HIS': ['CB', 'CG', 'CD2', 'CE1', 'ND1', 'NE2'],
    'ILE': ['CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['CB', 'CG', 'CD1', 'CD2'],
    'LYS': ['CE', 'NZ'],
    'MET': ['CG', 'CE', 'SD'],
    'PHE': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['CA', 'CB', 'CG', 'CD', 'N'],
    'SER': ['CA', 'CB', 'OG'],
    'THR': ['CA', 'CB', 'CG2', 'OG1'],
    'TRP': ['CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'NE1'],
    'TYR': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': ['CB', 'CG1', 'CG2'],
}

MOTIF_MODEL_META = {
    'AE3': {
        'scope': 'AE3 motif autoencoder (encoder + decoder)',
    },
    'LD4': {
        'scope': 'LD4 indexed all-atom motif flow-matching transformer',
        'motif_indexing': 'indexed',
        'motif_atom_mode': 'all_atom',
        'x_motif_feat_mode': 'all_atom',
        'center_every_step': False,
    },
    'LD5': {
        'scope': 'LD5 indexed tip-atom motif flow-matching transformer',
        'motif_indexing': 'indexed',
        'motif_atom_mode': 'tip_atoms',
        'x_motif_feat_mode': 'tip',
        'center_every_step': False,
    },
    'LD6': {
        'scope': 'LD6 unindexed all-atom motif flow-matching transformer',
        'motif_indexing': 'unindexed',
        'motif_atom_mode': 'all_atom',
        'x_motif_feat_mode': 'all_atom',
        'center_every_step': False,
    },
    'LD7': {
        'scope': 'LD7 unindexed tip-atom motif flow-matching transformer',
        'motif_indexing': 'unindexed',
        'motif_atom_mode': 'tip_atoms',
        'x_motif_feat_mode': 'tip',
        'center_every_step': False,
    },
}


from functools import lru_cache


def checkpoint(model: str):
    return torch.load(
        ROOT / 'weights' / MOTIF_CHECKPOINTS[model][0],
        map_location='cpu',
        weights_only=False,
        mmap=True,
    )


@lru_cache(maxsize=8)
def checkpoint_cfg_exp(model: str):
    ckpt = checkpoint(model)
    return OmegaConf.to_container(ckpt['hyper_parameters']['cfg_exp'], resolve=False)


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double(), b.double()
    diff = a - b
    return {
        'relRMS': (diff.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30)).item(),
        'maxAbs': diff.abs().max().item(),
    }


def quantize(t: torch.Tensor, precision: str):
    t = t.detach().float().contiguous()
    if t.ndim != 2 or precision == 'fp32':
        return t, None
    if precision == 'fp16':
        return t.half(), None
    if precision != 'int8':
        raise ValueError(precision)
    scale = t.abs().amax(1).clamp_min(1e-30) / 127
    return (t / scale[:, None]).round().clamp(-127, 127).to(torch.int8), scale


def dequantize(t: torch.Tensor, scale):
    return t.float() if scale is None else t.float() * scale[:, None]


@contextmanager
def _patch_x_motif_creator(x_motif_feat_mode: str):
    """Implements upstream README lines 153-160 (feature_factory.py lines 1845-1851) without editing .cache/upstream."""
    if x_motif_feat_mode != 'tip':
        yield
        return
    orig_get_creator = FeatureFactory.get_creator

    def patched_get_creator(self, f, **kwargs):
        if self.mode == 'seq' and f == 'x_motif':
            return XmotifBulkTipSeqFeat(**kwargs)
        return orig_get_creator(self, f, **kwargs)

    with patch.object(FeatureFactory, 'get_creator', patched_get_creator):
        yield


def model_state(model: str):
    ckpt = checkpoint(model)
    if model == 'AE3':
        hp = ckpt['hyper_parameters']
        cfg = OmegaConf.to_container(hp['cfg_ae'].nn_ae, resolve=True)
        state = {
            k: v
            for k, v in ckpt['state_dict'].items()
            if k.startswith('encoder.') or k.startswith('decoder.')
        }
        assert any(k.startswith('encoder.') for k in state), 'AE3 encoder weights missing'
        assert any(k.startswith('decoder.') for k in state), 'AE3 decoder weights missing'
        return cfg, state
    if model in ('LD4', 'LD5', 'LD6', 'LD7'):
        cfg = OmegaConf.to_container(ckpt['hyper_parameters']['cfg_exp'].nn, resolve=True)
        cfg['latent_dim'] = 8
        meta = MOTIF_MODEL_META[model]
        cfg['motif_indexing'] = meta['motif_indexing']
        cfg['motif_atom_mode'] = meta['motif_atom_mode']
        cfg['x_motif_feat_mode'] = meta['x_motif_feat_mode']
        cfg['center_every_step'] = meta['center_every_step']
        prefix = 'nn.'
        state = {k[len(prefix):]: v for k, v in ckpt['state_dict'].items() if k.startswith(prefix)}
        assert state, f'{model}: empty nn state_dict'
        return cfg, state
    raise ValueError(f'Unknown motif model: {model}')


def make_motif_model(model: str, precision: str = 'fp32', state: dict = None, cfg: dict = None):
    if cfg is None or state is None:
        loaded_cfg, original = model_state(model)
        if cfg is None:
            cfg = loaded_cfg
        if state is None:
            state = {k: dequantize(*quantize(v, precision)) for k, v in original.items()}
    if model == 'AE3':
        enc = EncoderTransformer(**cfg)
        dec = DecoderTransformer(**cfg)
        enc_state = {k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')}
        dec_state = {k[len('decoder.'):]: v for k, v in state.items() if k.startswith('decoder.')}
        assert len(enc_state) + len(dec_state) == len(state), 'Unexpected non-encoder/decoder key in AE3'
        enc.load_state_dict(enc_state, strict=True)
        dec.load_state_dict(dec_state, strict=True)
        return {'encoder': enc.eval(), 'decoder': dec.eval()}, cfg
    x_motif_feat_mode = cfg.get('x_motif_feat_mode', MOTIF_MODEL_META[model]['x_motif_feat_mode'])
    with _patch_x_motif_creator(x_motif_feat_mode):
        if cfg['name'] == 'local_latents_transformer':
            net = LocalLatentsTransformer(**cfg)
        elif cfg['name'] == 'local_latents_transformer_motif_uidx':
            net = LocalLatentsTransformerMotifUidx(**cfg)
        else:
            raise ValueError(f"Unsupported nn architecture {cfg['name']}")
    net.load_state_dict(state, strict=True)
    return net.eval(), cfg


def reload_motif_bundle(model: str, precision: str, with_config: bool = False):
    from safetensors.torch import load_file

    folder = ROOT / 'weights' / model.lower() / precision
    manifest = json.loads((folder / 'manifest.json').read_text())
    shards = {s['file']: load_file(folder / s['file']) for s in manifest['shards']}
    out = {}
    for name, entry in manifest['tensors'].items():
        parts = []
        for part in entry['parts']:
            s = shards[part['shard']]
            parts.append(dequantize(s[part['key']], s.get(part.get('scale_tensor', ''))))
        out[name] = torch.cat(parts, dim=0).reshape(entry['shape'])
    return (out, manifest['config']) if with_config else out


def parse_pdb_model1_atoms(pdb_path: Path):
    """Parse model-1 non-HETATM ATOM records matching biotite load_structure(model=1) & hetero==False."""
    atoms = []
    in_model = False
    saw_model = False
    for line in pdb_path.read_text().splitlines():
        if line.startswith('MODEL'):
            if saw_model:
                break
            saw_model = True
            in_model = True
            continue
        if line.startswith('ENDMDL') and in_model:
            break
        if not line.startswith('ATOM  '):
            continue
        altloc = line[16] if len(line) > 16 else ' '
        if altloc not in (' ', 'A', '1'):
            continue
        ins_code = line[26] if len(line) > 26 else ' '
        if ins_code != ' ':
            raise ValueError(f'Insertion codes unsupported in {pdb_path}: {line}')
        atoms.append({
            'atom_name': line[12:16].strip(),
            'res_name': line[17:20].strip(),
            'chain_id': line[21].strip(),
            'res_id': int(line[22:26].strip()),
            'coord': [float(line[30:38]), float(line[38:46]), float(line[46:54])],
        })
    return atoms


def select_motif_atoms(available_atoms: list[int], atom_selection_mode: str, residue_name: str = None) -> list[int]:
    backbone_atoms = [0, 1, 2, 4]  # N, CA, C, O
    if atom_selection_mode == 'ca':
        return [1] if 1 in available_atoms else []
    if atom_selection_mode == 'bb3o':
        return [i for i in backbone_atoms if i in available_atoms]
    if atom_selection_mode == 'all_atom':
        return list(available_atoms)
    if atom_selection_mode == 'tip_atoms':
        if residue_name is None:
            raise ValueError('residue_name required for tip_atoms')
        out = []
        for name in SIDECHAIN_TIP_ATOMS.get(residue_name, []):
            if name in rc.atom_order and rc.atom_order[name] in available_atoms:
                out.append(rc.atom_order[name])
        return out
    raise ValueError(f'Unknown atom_selection_mode: {atom_selection_mode}')


def extract_full_structure_for_ae(pdb_path: Path, center_ca: bool = True) -> dict:
    """Build AE3 encoder/decoder input batch from a PDB file matching upstream transforms."""
    atoms = parse_pdb_model1_atoms(pdb_path)
    unique_residues = []
    seen = set()
    for a in atoms:
        key = (a['chain_id'], a['res_id'])
        if key not in seen:
            seen.add(key)
            unique_residues.append(key)
    n_res = len(unique_residues)
    coords_ang = torch.zeros((n_res, 37, 3), dtype=torch.float32)
    coord_mask = torch.zeros((n_res, 37), dtype=torch.bool)
    residue_type = torch.full((n_res,), rc.restype_num, dtype=torch.int64)
    chain_ids = []
    for i, (chain_id, res_id) in enumerate(unique_residues):
        res_atoms = [a for a in atoms if a['chain_id'] == chain_id and a['res_id'] == res_id]
        res_1 = rc.restype_3to1.get(res_atoms[0]['res_name'], 'UNK')
        residue_type[i] = rc.restype_order.get(res_1, rc.restype_num)
        chain_ids.append(chain_id)
        for a in res_atoms:
            if a['atom_name'] in rc.atom_order:
                idx = rc.atom_order[a['atom_name']]
                coord_mask[i, idx] = True
                coords_ang[i, idx] = torch.tensor(a['coord'], dtype=torch.float32)

    coords_nm = coords_ang / 10.0
    if center_ca:
        ca_nm = coords_nm[:, 1, :]
        mask_1d = torch.ones((n_res,), dtype=torch.bool)
        com_nm = mean_w_mask(ca_nm, mask_1d, keepdim=True)  # [1, 3]
        coords_nm = (coords_nm - com_nm[:, None, :]) * coord_mask[..., None]
        coords_ang = coords_nm * 10.0

    # ChainBreakPerResidueTransform uses 4.0 Å cutoff on CA-CA distance
    ca_ang = coords_ang[:, 1, :]
    if n_res > 1:
        ca_dists = torch.norm(ca_ang[1:] - ca_ang[:-1], dim=1)
        chain_breaks = torch.cat([ca_dists > 4.0, torch.tensor([False], dtype=torch.bool)])
    else:
        chain_breaks = torch.tensor([False], dtype=torch.bool)

    unique_chains = sorted(set(chain_ids))
    chain_map = {c: idx for idx, c in enumerate(unique_chains)}
    chains = torch.tensor([chain_map[c] for c in chain_ids], dtype=torch.int64)
    mask = torch.ones((1, n_res), dtype=torch.bool)
    mask_coords = torch.ones((1, n_res, 37, 3), dtype=torch.bool)
    return {
        'coords': coords_ang.unsqueeze(0),
        'coords_nm': coords_nm.unsqueeze(0),
        'coord_mask': coord_mask.unsqueeze(0),
        'residue_type': residue_type.unsqueeze(0),
        'chain_breaks_per_residue': chain_breaks.unsqueeze(0),
        'chains': chains.unsqueeze(0),
        'mask': mask,
        'mask_dict': {
            'coords': mask_coords,
            'residue_type': mask,
        },
    }


def extract_motif_condition(
    pdb_path: Path,
    contig_string: str = None,
    motif_only: bool = False,
    atom_selection_mode: str = 'all_atom',
    motif_atom_spec: str = None,
    coors_to_nm: bool = True,
    center_motif: bool = True,
) -> dict:
    """Exact pure-PyTorch equivalent of motif_utils.parse_motif + GenDataset.generate_motif_info for deterministic contigs."""
    atoms = parse_pdb_model1_atoms(pdb_path)
    alphabet = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'

    if motif_atom_spec is not None:
        parsed = []
        for match in re.finditer(r'([A-Za-z])(\d+): \[([^\]]+)\]', motif_atom_spec):
            parsed.append((match.group(1), int(match.group(2)), [x.strip() for x in match.group(3).split(',')]))
        unique_residues = []
        seen = set()
        for c, r, _ in parsed:
            if (c, r) not in seen:
                seen.add((c, r))
                unique_residues.append((c, r))
        n_motif = len(unique_residues)
        motif_mask = torch.zeros((n_motif, 37), dtype=torch.bool)
        x_motif = torch.zeros((n_motif, 37, 3), dtype=torch.float32)
        residue_type = torch.full((n_motif,), rc.restype_num, dtype=torch.int64)
        for i, (c, r) in enumerate(unique_residues):
            wanted = [name for cc, rr, names in parsed if cc == c and rr == r for name in names]
            res_atoms = [a for a in atoms if a['chain_id'] == c and a['res_id'] == r]
            if not res_atoms:
                continue
            res_1 = rc.restype_3to1.get(res_atoms[0]['res_name'], 'UNK')
            residue_type[i] = rc.restype_order.get(res_1, rc.restype_num)
            for a in res_atoms:
                if a['atom_name'] in wanted and a['atom_name'] in rc.atom_order:
                    idx = rc.atom_order[a['atom_name']]
                    motif_mask[i, idx] = True
                    coord = torch.tensor(a['coord'], dtype=torch.float32)
                    x_motif[i, idx] = coord / 10.0 if coors_to_nm else coord
    else:
        assert contig_string is not None, 'contig_string or motif_atom_spec required'
        selected_atoms = []
        seen_chains = set()
        for part in contig_string.split('/'):
            if part[0] not in alphabet:
                continue
            chain_id = part[0]
            if motif_only:
                if chain_id in seen_chains:
                    continue
                seen_chains.add(chain_id)
                seg_atoms = [a for a in atoms if a['chain_id'] == chain_id]
            else:
                body = part[1:]
                if '-' in body:
                    start, end = map(int, body.split('-'))
                else:
                    start = end = int(body)
                seg_atoms = [a for a in atoms if a['chain_id'] == chain_id and start <= a['res_id'] <= end]
            selected_atoms.extend(seg_atoms)

        unique_residues = []
        seen = set()
        for a in selected_atoms:
            key = (a['chain_id'], a['res_id'])
            if key not in seen:
                seen.add(key)
                unique_residues.append(key)
        n_motif = len(unique_residues)
        motif_mask = torch.zeros((n_motif, 37), dtype=torch.bool)
        x_motif = torch.zeros((n_motif, 37, 3), dtype=torch.float32)
        residue_type = torch.full((n_motif,), rc.restype_num, dtype=torch.int64)
        for i, (c, r) in enumerate(unique_residues):
            res_atoms = [a for a in selected_atoms if a['chain_id'] == c and a['res_id'] == r]
            res_name = res_atoms[0]['res_name']
            res_1 = rc.restype_3to1.get(res_name, 'UNK')
            residue_type[i] = rc.restype_order.get(res_1, rc.restype_num)
            avail = [rc.atom_order[a['atom_name']] for a in res_atoms if a['atom_name'] in rc.atom_order]
            chosen = select_motif_atoms(avail, atom_selection_mode, res_name)
            for a in res_atoms:
                if a['atom_name'] in rc.atom_order:
                    idx = rc.atom_order[a['atom_name']]
                    if idx in chosen:
                        motif_mask[i, idx] = True
                        coord = torch.tensor(a['coord'], dtype=torch.float32)
                        x_motif[i, idx] = coord / 10.0 if coors_to_nm else coord

    if center_motif:
        motif_center = mean_w_mask(x_motif.flatten(0, 1), motif_mask.flatten(0, 1)).unsqueeze(0)
        x_motif = (x_motif - motif_center) * motif_mask[..., None]

    if contig_string is None:
        full_length = n_motif
        motif_indices_0 = list(range(n_motif))
        resolved_contig = None
    else:
        pos = 0
        motif_indices_0 = []
        resolved_parts = []
        for part in contig_string.split('/'):
            if part[0] in alphabet:
                body = part[1:]
                start, end = map(int, body.split('-')) if '-' in body else (int(body), int(body))
                seg_len = end - start + 1
                motif_indices_0.extend(range(pos, pos + seg_len))
                new_part = f'{part[0]}{pos + 1}' + (f'-{pos + seg_len}' if seg_len > 1 else '')
                resolved_parts.append(new_part)
            else:
                if '-' in part:
                    raise ValueError(f'Oracle contig_string must use deterministic scaffold lengths, got {part}')
                seg_len = int(part)
                resolved_parts.append(str(seg_len))
            pos += seg_len
        full_length = pos
        resolved_contig = '/'.join(resolved_parts)
        assert len(motif_indices_0) == n_motif, f'Contig motif count {len(motif_indices_0)} != extracted {n_motif}'

    idx_tensor = torch.tensor(motif_indices_0, dtype=torch.int64)
    full_mask = torch.zeros((full_length, 37), dtype=torch.bool)
    full_x = torch.zeros((full_length, 37, 3), dtype=torch.float32)
    full_seq = torch.full((full_length,), rc.restype_num, dtype=torch.int64)
    full_mask[idx_tensor] = motif_mask
    full_x[idx_tensor] = x_motif
    full_seq[idx_tensor] = residue_type

    if center_motif:
        motif_center_full = mean_w_mask(full_x.flatten(0, 1), full_mask.flatten(0, 1)).unsqueeze(0)
        full_x = (full_x - motif_center_full) * full_mask[..., None]

    seq_motif_mask = full_mask.sum(dim=-1).bool()
    return {
        'length': full_length,
        'n_motif_res': int(seq_motif_mask.sum().item()),
        'motif_indices': motif_indices_0,
        'resolved_contig': resolved_contig,
        'motif_mask_raw': motif_mask,
        'x_motif_raw': x_motif,
        'residue_type_raw': residue_type,
        'motif_mask': full_mask.unsqueeze(0),
        'x_motif': full_x.unsqueeze(0),
        'seq_motif_mask': seq_motif_mask.unsqueeze(0),
        'seq_motif': full_seq.unsqueeze(0),
        'mask': torch.ones((1, full_length), dtype=torch.bool),
    }


def compute_motif_rmsd_angstrom(
    gen_coords_nm: torch.Tensor,
    gen_atom_mask: torch.Tensor,
    gen_aatype: torch.Tensor,
    motif_mask_raw: torch.Tensor,
    x_motif_raw_nm: torch.Tensor,
    motif_aatype_raw: torch.Tensor,
    motif_indices: list[int],
    unindexed: bool = False,
) -> dict:
    """Compute motif preservation metrics (Kabsch-aligned & unaligned RMSD in Å, plus motif seq recovery)."""
    gen_ang = gen_coords_nm.squeeze(0).float() * 10.0  # [L, 37, 3]
    gen_mask = gen_atom_mask.squeeze(0).bool()  # [L, 37]
    gen_aa = gen_aatype.squeeze(0).long()  # [L]
    motif_ang = x_motif_raw_nm.float() * 10.0  # [M, 37, 3]
    L = gen_ang.shape[0]
    M = motif_ang.shape[0]

    if not unindexed:
        matched_idx = list(motif_indices)
    else:
        # Exact greedy matching from motif_utils.pad_motif_to_full_length_unindexed
        matched_idx = []
        for match_aa in (True, False):
            matched_idx = []
            for i in range(M):
                m_mask_i = motif_mask_raw[i]
                if m_mask_i.sum() == 0:
                    matched_idx.append(0)
                    continue
                best_j = None
                best_rmsd = float('inf')
                for j in range(L):
                    if j in matched_idx:
                        continue
                    if match_aa and gen_aa[j].item() != motif_aatype_raw[i].item():
                        continue
                    overlap = m_mask_i & gen_mask[j]
                    if overlap.sum() == 0:
                        continue
                    d = (motif_ang[i][overlap] - gen_ang[j][overlap]).square().sum(-1).mean().sqrt().item()
                    if d < best_rmsd:
                        best_rmsd = d
                        best_j = j
                matched_idx.append(best_j)
            if None not in matched_idx:
                break
        if None in matched_idx:
            matched_idx = list(range(M))

    idx_t = torch.tensor(matched_idx, dtype=torch.int64)
    full_m_mask = torch.zeros((L, 37), dtype=torch.bool)
    full_m_ang = torch.zeros((L, 37, 3), dtype=torch.float32)
    full_m_aa = torch.full((L,), rc.restype_num, dtype=torch.int64)
    full_m_mask[idx_t] = motif_mask_raw
    full_m_ang[idx_t] = motif_ang
    full_m_aa[idx_t] = motif_aatype_raw

    overlap_mask = gen_mask & full_m_mask
    res_active = full_m_mask.any(dim=-1)
    seq_rec = (gen_aa[res_active] == full_m_aa[res_active]).float().mean().item() if res_active.any() else 1.0

    pts_gen = gen_ang[overlap_mask]
    pts_ref = full_m_ang[overlap_mask]
    unaligned_rmsd = (pts_gen - pts_ref).square().sum(-1).mean().sqrt().item()
    if pts_gen.shape[0] >= 3:
        pts_aligned, pts_ref_aligned = kabsch_align_ind(pts_gen, pts_ref, ret_both=True)
        aligned_rmsd = (pts_aligned - pts_ref_aligned).square().sum(-1).mean().sqrt().item()
    else:
        aligned_rmsd = unaligned_rmsd
    return {
        'matched_indices': matched_idx,
        'active_motif_atoms': int(overlap_mask.sum().item()),
        'motif_rmsd_aligned_ang': aligned_rmsd,
        'motif_rmsd_unaligned_ang': unaligned_rmsd,
        'motif_seq_recovery': seq_rec,
    }
