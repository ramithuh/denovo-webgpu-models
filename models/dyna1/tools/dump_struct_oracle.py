#!/usr/bin/env python3
"""Generate upstream PyTorch FP32 oracle fixtures for Dyna-1 seq+struct and struct-only modes."""
import json, math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.cache/esm_upstream'))
sys.path.insert(0, str(ROOT / '.cache/dyna1_upstream'))
sys.path.insert(0, str(ROOT / 'tools'))

from esm.models.esm3 import ESM3
from esm.models.vqvae import StructureTokenEncoder
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.utils import encoding
from model.transformer import Transformer
from export_dyna1 import load_bundle
from export_structure_encoder import load_structure_bundle

VOCAB = ['<cls>','<pad>','<eos>','<unk>',*'LAGVSERTIDPKQNFYMHWCXBUZO','.', '-', '|', '<mask>']


def setup():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)


def prob_adj(logits):
    return torch.sigmoid(logits - math.log(0.05 / 0.95))


def stats(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return dict(
        rel_rms=float(((a - b).square().sum() / b.square().sum().clamp_min(1e-30)).sqrt()),
        max_abs=float((a - b).abs().max()),
        correlation=float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
    )


def load_models(cache: Path):
    setup()
    enc_sd = torch.load(cache / 'esm3_structure_encoder_v0.pth', map_location='cpu', weights_only=True)
    with torch.device('meta'):
        struct_enc = StructureTokenEncoder(1024, 1, 128, 2, 128, 4096)
        tokenizers = SimpleNamespace(
            sequence=SimpleNamespace(mask_token_id=32, bos_token_id=0, eos_token_id=2, pad_token_id=1),
            structure=StructureTokenizer(),
        )
        backbone = ESM3(1536, 24, 256, 48, lambda dev: struct_enc, None, None, tokenizers)
        head = Transformer(1536)
    struct_enc.load_state_dict(enc_sd, strict=True, assign=True)
    struct_enc = struct_enc.float().eval()
    base = torch.load(cache / 'esm3_sm_open_v1.pth', map_location='cpu', weights_only=True, mmap=True)
    backbone.load_state_dict(base, strict=True, assign=True)
    backbone = backbone.float().eval()
    for block in backbone.transformer.blocks:
        block.attn.rotary.device = torch.device('cpu')
        block.attn.rotary.reset_parameters()
    ckpt = torch.load(cache / 'dyna1.pt', map_location='cpu', weights_only=True, mmap=True)
    head.load_state_dict({k.removeprefix('classifier.'): v for k, v in ckpt.items()}, strict=True, assign=True)
    head = head.float().eval()
    return struct_enc, backbone, head, tokenizers


def load_pdb_case(cache: Path, pdb_id: str):
    p = cache / 'pdb' / f'{pdb_id}.npz'
    if p.exists():
        d = np.load(p)
        return str(d['sequence']), d['bb3'].astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    chain = ProteinChain.from_rcsb(pdb_id, chain_id='A')
    protein = ESMProtein.from_protein_chain(chain)
    bb3 = protein.coordinates[:, :3, :].float().numpy()
    np.savez(p, sequence=protein.sequence, bb3=bb3, residue_index=chain.residue_index)
    return protein.sequence, bb3


@torch.inference_mode()
def main():
    cache = ROOT / '.cache'
    struct_enc, backbone, head, _ = load_models(cache)
    h22 = {}
    emb_hook = {}
    backbone.encoder.register_forward_hook(lambda m, a, o: emb_hook.update(x=o.detach()))
    backbone.transformer.blocks[22].register_forward_hook(lambda m, a, o: h22.update(x=o[0].detach() if isinstance(o, tuple) else o.detach()))

    cases = {}
    for pdb_id in ['1UBQ', '1OCA', '1RYK']:
        seq, bb3 = load_pdb_case(cache, pdb_id)
        cases[pdb_id] = (seq, bb3)
    # Also test missing backbone atoms (N/CA/C set to NaN at residues 0, 10, 75 of 1UBQ)
    ubq_seq, ubq_bb3 = cases['1UBQ']
    ubq_missing = ubq_bb3.copy()
    ubq_missing[0, :, :] = np.nan
    ubq_missing[10, 1, :] = np.nan
    ubq_missing[75, 2, :] = np.nan
    cases['1UBQ_missing'] = (ubq_seq, ubq_missing)

    dest = ROOT / 'fixtures'
    dest.mkdir(exist_ok=True)
    meta = dict(
        format='dyna1-struct-oracle-v1',
        upstream_commit='814a453fd85305536b919901e41062b91d5c9d86',
        torch_version=torch.__version__,
        device='cpu',
        dtype='float32',
        proteins={},
        tensors={},
    )
    refs = {}
    offset = 0
    with (dest / 'oracle_dyna1_struct.bin').open('wb') as f:
        def write_tensor(key, arr, dtype='F32'):
            nonlocal offset
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().contiguous().numpy()
            arr = arr.astype('<i4' if dtype == 'I32' else '<f4')
            raw = arr.tobytes()
            meta['tensors'][key] = dict(dtype=dtype, shape=list(arr.shape), byteOffset=offset, byteLength=len(raw))
            f.write(raw)
            offset += len(raw)

        for name, (seq, bb3) in cases.items():
            L = len(seq)
            meta['proteins'][name] = dict(sequence=seq, length=L)
            coords_t = torch.from_numpy(bb3).float().unsqueeze(0)
            # Capture pre_vq output too
            pre_vq_box = {}
            h_pvq = struct_enc.pre_vq_proj.register_forward_hook(lambda m, a, o: pre_vq_box.update(x=o.detach()))
            try:
                _, struct_toks = struct_enc.encode(coords_t)
            finally:
                h_pvq.remove()
            struct_toks = struct_toks.long()  # [1, L]
            seq_toks = torch.tensor([[VOCAB.index(c) for c in seq]], dtype=torch.long)
            mask = torch.ones((1, L), dtype=torch.bool)

            # 1) seq-only
            backbone(sequence_tokens=seq_toks, structure_tokens=None, sequence_id=mask)
            l_seq = head(h22['x'], torch.logical_not(mask)).reshape(-1)
            p_seq = prob_adj(l_seq)

            # 2) seq+struct (--pdb ... --use_pdb_seq)
            backbone(sequence_tokens=seq_toks, structure_tokens=struct_toks, sequence_id=mask)
            emb_both = emb_hook['x'][0]
            h22_both = h22['x'][0]
            l_both = head(h22['x'], torch.logical_not(mask)).reshape(-1)
            p_both = prob_adj(l_both)

            # 3) struct-only (--pdb ... without --use_pdb_seq)
            backbone(sequence_tokens=None, structure_tokens=struct_toks, sequence_id=mask)
            emb_str = emb_hook['x'][0]
            l_str = head(h22['x'], torch.logical_not(mask)).reshape(-1)
            p_str = prob_adj(l_str)

            write_tensor(f'{name}.coords', bb3, 'F32')
            write_tensor(f'{name}.structure_tokens', struct_toks[0], 'I32')
            write_tensor(f'{name}.pre_vq', pre_vq_box['x'][0], 'F32')
            write_tensor(f'{name}.seq_only.logits', l_seq, 'F32')
            write_tensor(f'{name}.seq_only.probs', p_seq, 'F32')
            write_tensor(f'{name}.seq_struct.logits', l_both, 'F32')
            write_tensor(f'{name}.seq_struct.probs', p_both, 'F32')
            write_tensor(f'{name}.struct_only.logits', l_str, 'F32')
            write_tensor(f'{name}.struct_only.probs', p_str, 'F32')
            if name == '1UBQ':
                write_tensor(f'{name}.seq_struct.embedding', emb_both, 'F32')
                write_tensor(f'{name}.seq_struct.final_embeddings', h22_both, 'F32')
                write_tensor(f'{name}.struct_only.embedding', emb_str, 'F32')

            refs[name] = dict(
                seq_toks=seq_toks, coords_t=coords_t, struct_toks=struct_toks,
                l_both=l_both, p_both=p_both, l_str=l_str, p_str=p_str,
                l_seq=l_seq, p_seq=p_seq,
            )

    (dest / 'oracle_dyna1_struct.json').write_text(json.dumps(meta, indent=2) + '\n')

    # Verify round-trip of exported structure_encoder bundle + existing fp32/fp16/int8 bundles!
    se_sd = load_structure_bundle(ROOT / 'weights/structure_encoder')
    delta = se_sd.pop('structure_tokens_embed_delta')
    enc_weights = {k.removeprefix('encoder.'): v for k, v in se_sd.items() if k.startswith('encoder.')}
    with torch.device('meta'):
        struct_enc_rt = StructureTokenEncoder(1024, 1, 128, 2, 128, 4096)
    enc_weights['codebook.N'] = torch.zeros(4096)
    enc_weights['codebook.z_avg'] = enc_weights['codebook.embeddings'].clone()
    struct_enc_rt.load_state_dict(enc_weights, strict=True, assign=True)
    struct_enc_rt = struct_enc_rt.float().eval()

    class AdditiveEncoder(torch.nn.Module):
        def __init__(self, emb_w, delta_w):
            super().__init__()
            self.register_buffer('emb_w', emb_w)
            self.register_buffer('delta_w', delta_w)
        def forward(self, seq_t, struct_t, *args):
            return torch.nn.functional.embedding(seq_t, self.emb_w) + torch.nn.functional.embedding(struct_t, self.delta_w)

    token_metrics = {}
    rt_toks = {}
    for name, ref in refs.items():
        _, toks = struct_enc_rt.encode(ref['coords_t'])
        rt_toks[name] = toks.long()
        exact = int((toks == ref['struct_toks']).sum())
        total = int(ref['struct_toks'].numel())
        token_metrics[name] = dict(exact=exact, total=total, match_rate=exact / total)
        assert exact == total, f'Token mismatch on {name}: {exact}/{total}'

    metrics = dict(structure_encoder_tokens=token_metrics, precisions={})
    for prec in ['fp32', 'fp16', 'int8']:
        sd = load_bundle(ROOT / 'weights' / prec)
        emb_w = sd.pop('embedding.weight')
        backbone.encoder = AdditiveEncoder(emb_w, delta)
        for i, block in enumerate(backbone.transformer.blocks[:23]):
            prefix = f'backbone.{i}.'
            block.load_state_dict({k.removeprefix(prefix): v for k, v in sd.items() if k.startswith(prefix)}, strict=False)
        head.load_state_dict({k.removeprefix('head.'): v for k, v in sd.items() if k.startswith('head.')}, strict=True)
        metrics['precisions'][prec] = {}
        for name, ref in refs.items():
            L = ref['seq_toks'].shape[1]
            mask = torch.ones((1, L), dtype=torch.bool)
            backbone(sequence_tokens=ref['seq_toks'], structure_tokens=rt_toks[name], sequence_id=mask)
            lb = head(h22['x'], torch.logical_not(mask)).reshape(-1)
            pb = prob_adj(lb)
            backbone(sequence_tokens=None, structure_tokens=rt_toks[name], sequence_id=mask)
            ls = head(h22['x'], torch.logical_not(mask)).reshape(-1)
            ps = prob_adj(ls)
            metrics['precisions'][prec][name] = dict(
                seq_struct=dict(logits=stats(lb, ref['l_both']), probs=stats(pb, ref['p_both'])),
                struct_only=dict(logits=stats(ls, ref['l_str']), probs=stats(ps, ref['p_str'])),
            )
    (dest / 'struct_precision_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
