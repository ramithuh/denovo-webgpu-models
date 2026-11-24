#!/usr/bin/env python3
"""Deterministic eager PyTorch FP32 oracle. Offline fixture generation only.

Uses the official EMA checkpoint and pinned source, never a traced/captured graph.
FP32 mode explicitly disables the upstream pairformer's forced BF16 autocast.
"""
import argparse
import importlib.metadata
import copy
import json
import os
import random
import time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['RFD3_DENSE_SDPA_ATTENTION'] = '0'
import numpy as np
import torch
from checkpoint import ROOT, DEFAULT_CHECKPOINT, load_checkpoint, COMMIT
from export_weights import atomic_json, digest_file


def tensor_json(t):
    return dict(shape=list(t.shape), dtype=str(t.dtype), data=t.detach().cpu().reshape(-1).tolist())


def pipeline_example(checkpoint, specification, seed):
    import hydra
    from omegaconf import OmegaConf
    from rfd3.inference.input_parsing import DesignInputSpecification
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    cfg = OmegaConf.to_container(checkpoint['train_cfg'].datasets.val.unconditional.dataset.transform, resolve=True)
    cfg.update(diffusion_batch_size=1, residue_cache_dir=None)
    pipeline = hydra.utils.instantiate(OmegaConf.create(cfg))
    spec = DesignInputSpecification.safe_init(**specification)
    return pipeline(spec.to_pipeline_input(example_id='oracle'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, default=ROOT / 'fixtures/oracle')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seed', type=int, default=20261123)
    parser.add_argument('--specifications', type=Path, help='JSON map of names to upstream design specifications')
    parser.add_argument('--linear-only', action='store_true')
    parser.add_argument('--trajectory', action='store_true', help='Capture a four-point monomer trajectory with shared normal draws')
    parser.add_argument('--recycles', type=int, default=2)
    args = parser.parse_args()
    if args.recycles < 1:
        raise ValueError('Recycles must be positive')
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    random.seed(args.seed); np.random.seed(args.seed % 2**32); torch.manual_seed(args.seed)
    checkpoint, net, config = load_checkpoint(args.checkpoint)
    for module in net.modules():
        if hasattr(module, 'force_bfloat16'):
            module.force_bfloat16 = False
        if hasattr(module, 'use_deepspeed_evo'):
            module.use_deepspeed_evo = False
    net = net.to(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(upstream_commit=COMMIT, checkpoint_sha256=digest_file(args.checkpoint), seed=args.seed,
                  device=args.device, torch_version=torch.__version__, state_branch='shadow',
                  precision='FP32; TF32 disabled; forced BF16 disabled; eager torch kernels',
                  recycles=args.recycles, case_seed_offset=100, trajectory_seed_offset=10000,
                  atomworks_version=importlib.metadata.version('atomworks'), biotite_version=importlib.metadata.version('biotite'),
                  linear_cases=[], model_cases=[])
    modules = dict(net.named_modules())
    selected = [
        'diffusion_module.process_r',
        'token_initializer.atom_1d_embedder_1.embedders.ref_atom_name_chars',
        'token_initializer.transition_post_token.linear_1',
        'diffusion_module.diffusion_transformer.transformer_blocks.0.transition_block.linear_1',
        'diffusion_module.sequence_head.linear',
    ]
    # Resolve the transformer path from the actual module inventory, not a guessed checkpoint name.
    selected[3] = next(n for n, m in modules.items() if n.startswith('diffusion_module.diffusion_transformer.')
                       and n.endswith('transition_block.linear_1'))
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    with torch.inference_mode():
        for index, name in enumerate(selected):
            module = modules[name]
            rows = [7, 3, 5, 9, 2][index]
            x = torch.randn(rows, module.in_features, generator=generator).to(args.device)
            y = module(x)
            from rfd3.model.layers.layer_utils import EmbeddingLayer
            outputs = {'fp32': tensor_json(y)}
            for precision in ('fp16', 'int8'):
                w = module.weight
                if not isinstance(module, EmbeddingLayer):
                    if precision == 'fp16':
                        w = w.half().float()
                    else:
                        scales = w.abs().amax(dim=1) / 127
                        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
                        w = torch.round(w / scales[:, None]).clamp(-127, 127) * scales[:, None]
                outputs[precision] = tensor_json(torch.nn.functional.linear(x, w, module.bias))
            fixture = dict(name=name, input=tensor_json(x), output=tensor_json(y), output_by_precision=outputs,
                           bias=tensor_json(module.bias) if module.bias is not None else None)
            filename = f'linear-{index}.json'
            atomic_json(args.output / filename, fixture)
            record['linear_cases'].append(filename)
        if not args.linear_only:
            specs = json.loads(args.specifications.read_text()) if args.specifications else {
                'monomer': {'length': '4'}, 'cyclic': {'length': '4', 'cyclic_chains': ['A']},
                'atomic-motif': {'input': str(ROOT / 'fixtures/conditioning.pdb'), 'contig': '1,A1,2',
                    'select_fixed_atoms': {'A1': 'N,CA'}, 'select_hbond_donor': {'A1': 'N'}, 'select_buried': {'A1': 'CA'}}}
            for case_index, (name, spec) in enumerate(specs.items()):
                case_generator = torch.Generator().manual_seed(args.seed + 100 + case_index)
                if not name.replace('-', '').replace('_', '').isalnum():
                    raise ValueError('Invalid fixture name')
                example = pipeline_example(checkpoint, spec, args.seed)
                f = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in example['feats'].items()}
                feature_json = {k: tensor_json(v) for k, v in f.items() if isinstance(v, torch.Tensor)}
                l = f['atom_to_token_map'].shape[0]
                x = torch.randn(1, l, 3, generator=case_generator).to(args.device) * 16
                fixed = f['is_motif_atom_with_fixed_coord']
                x[:, fixed] = f['motif_pos'][fixed]
                t = torch.tensor([16.0], device=args.device)
                init = net.token_initializer(copy.deepcopy(f))
                if args.device == 'cuda': torch.cuda.synchronize()
                start = time.perf_counter()
                output = net.diffusion_module(x, t, copy.deepcopy(f), **init, n_recycle=args.recycles)
                if args.device == 'cuda': torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                repeated = net.diffusion_module(x, t, copy.deepcopy(f), **init, n_recycle=args.recycles)
                for key in output:
                    if not torch.equal(output[key], repeated[key]):
                        raise AssertionError('Non-deterministic oracle: ' + key)
                fixture = dict(specification=spec, features=feature_json, X_noisy_L=tensor_json(x), t=tensor_json(t),
                    initializer={k: tensor_json(v) for k, v in init.items() if isinstance(v, torch.Tensor)},
                    output={k: tensor_json(v) for k, v in output.items()}, pytorch_step_seconds=elapsed)
                if args.trajectory and name == 'monomer':
                    from rfd3.model.inference_sampler import SampleDiffusionWithMotif
                    from unittest.mock import patch
                    sampler = SampleDiffusionWithMotif(num_timesteps=4, gamma_0=0.6,
                        n_recycle=args.recycles, allow_realignment=False)
                    draws = []
                    trajectory_generator = torch.Generator().manual_seed(args.seed + 10000 + case_index)
                    def normal(mean, std, size, device, **kwargs):
                        value = (torch.randn(size, generator=trajectory_generator) * std + mean).to(device)
                        draws.append(tensor_json(value))
                        return value
                    base = torch.zeros_like(x)
                    with patch('torch.normal', normal):
                        trajectory = sampler.sample_diffusion_like_af3(f=copy.deepcopy(f), diffusion_module=net.diffusion_module,
                            diffusion_batch_size=1, coord_atom_lvl_to_be_noised=base, initializer_outputs=init,
                            ref_initializer_outputs=None, f_ref=None)
                    fixture['trajectory'] = dict(schedule=tensor_json(sampler._construct_inference_noise_schedule(torch.device(args.device))),
                        coordinates=tensor_json(base), random=draws,
                        output={k: [tensor_json(t) for t in v] if isinstance(v, list) else tensor_json(v)
                                for k, v in trajectory.items() if v is not None})
                filename = name + '.json'
                atomic_json(args.output / filename, fixture)
                record['model_cases'].append(filename)
                print(json.dumps(dict(case=name, tokens=len(f['restype']), atoms=l, pytorch_step_seconds=elapsed)), flush=True)
    atomic_json(args.output / 'manifest.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
