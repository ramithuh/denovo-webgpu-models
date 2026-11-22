"""CPU FP32 intermediate traces from unmodified upstream CoordinateDenoiser.

These are explicitly denoiser oracles. Sampler trajectories have their own gate.
"""
import argparse
import copy
import json
from environment import ROOT, COMMIT
from export_protpardelle import checkpoint_state, config_for, reload_bundle
import numpy as np
import torch
from protpardelle.core.models import CoordinateDenoiser
from protpardelle.utils import dict_to_namespace

torch.set_num_threads(8)
torch.set_grad_enabled(False)


def inputs(config, mode, length):
    # Save actual draws; consumers never reproduce these from the NumPy seed.
    rng = np.random.default_rng(517 + length)
    coords = (rng.normal(size=(1, length, 37, 3)) * 12).astype(np.float32)
    mask = np.ones((1, length), np.float32)
    residues = np.arange(1, length + 1, dtype=np.int64)[None]
    chains = np.zeros((1, length), np.int64)
    crop = np.zeros_like(coords)
    if mode != 'unconditional':
        crop[:, 2:5] = rng.normal(size=(1, 3, 37, 3)).astype(np.float32) * 4
    if mode == 'backbone_motif':
        crop[:, :, [i for i in range(37) if i not in (0, 1, 2, 4)]] = 0
    if mode == 'sidechain_motif':
        crop[:, :, [0, 1, 2, 4]] = 0
    if mode == 'tip_motif':
        crop[:, :, :32] = 0
    if mode == 'multichain_hotspot':
        chains[:, length // 2:] = 1
        residues[:, length // 2:] += config['data']['chain_residx_gap'] - 1
        if config['data']['chain_residx_gap'] == 0:
            residues[:, length // 2:] = np.arange(1, length - length // 2 + 1)
    if mode == 'masked_partial':
        mask[:, -2:] = 0
    self_cond = None if mode == 'unconditional' else rng.normal(size=coords.shape).astype(np.float32) * .1
    hotspot = np.zeros((1, length), np.int64)
    if mode == 'multichain_hotspot':
        hotspot[:, 2:4] = 1
    noise = np.full((1, length), .75 if mode == 'masked_partial' else 12., np.float32)
    return {'noisy_coords': torch.from_numpy(coords), 'noise_level': torch.from_numpy(noise),
            'seq_mask': torch.from_numpy(mask), 'residue_index': torch.from_numpy(residues),
            'chain_index': torch.from_numpy(chains), 'hotspot_mask': torch.from_numpy(hotspot),
            'struct_self_cond': None if self_cond is None else torch.from_numpy(self_cond),
            'struct_crop_cond': torch.from_numpy(crop)}


def run(model, data):
    traces, handles = {}, []
    def capture(name):
        return lambda module, args, out: traces.__setitem__(name, out.detach().clone())
    handles.append(model.noise_block.block[0].register_forward_hook(capture('noise_embedding')))
    handles.append(model.noise_block.register_forward_hook(capture('noise_conditioning')))
    handles.append(model.net.register_forward_pre_hook(lambda module, args: traces.__setitem__('input_features', args[0].permute(0, 1, 3, 2).flatten(2).clone())))
    handles.append(model.net.to_patch_embedding.register_forward_hook(lambda module, args, out: traces.__setitem__('patch_embedding', (out * data['seq_mask'][..., None]).clone())))
    for i, (attn, ff) in enumerate(model.net.transformer.layers):
        handles.append(attn.register_forward_hook(capture(f'block.{i}.attention')))
        handles.append(ff.register_forward_hook(capture(f'block.{i}.ff')))
        handles.append(ff.register_forward_hook(lambda module, args, out, i=i: traces.__setitem__(f'block.{i}.hidden', ((args[0] + out) * data['seq_mask'][..., None]).clone())))
    handles.append(model.net.from_patch[1].register_forward_hook(capture('output_patch')))
    try:
        output = model(**data)
        traces['denoised_coords'] = output
        return traces
    finally:
        for handle in handles:
            handle.remove()


def save_fixture(name, config, cases):
    metadata = {'format': 'protpardelle-oracle-v1', 'scope': 'denoiser', 'variant': name, 'upstream_commit': COMMIT, 'config': config, 'cases': []}
    blob = bytearray()
    def array(value):
        if value is None:
            return None
        raw = value.detach().cpu().numpy()
        raw = raw.astype('<i4' if raw.dtype.kind in 'iu' else '<f4')
        entry = {'offset': len(blob), 'length': raw.size, 'shape': list(raw.shape), 'dtype': 'I32' if raw.dtype.kind == 'i' else 'F32'}
        blob.extend(raw.tobytes())
        return entry
    for case in cases:
        metadata['cases'].append({'mode': case['mode'], 'precision': case['precision'], 'length': case['length'],
                                  'inputs': {k: array(v) for k, v in case['inputs'].items()},
                                  'traces': {k: array(v) for k, v in case['traces'].items()}})
    if len(blob)>10_000_000:
        metadata['binary_parts']=[]
        for i,start in enumerate(range(0,len(blob),10_000_000)):
            file=f'oracle_{name}.part-{i}.bin';metadata['binary_parts'].append(file)
            (ROOT/'fixtures'/file).write_bytes(blob[start:start+10_000_000])
        (ROOT/'fixtures'/f'oracle_{name}.bin').unlink(missing_ok=True)
    else:
        (ROOT / 'fixtures' / f'oracle_{name}.bin').write_bytes(blob)
    (ROOT / 'fixtures' / f'oracle_{name}.json').write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', action='append')
    args = parser.parse_args()
    metrics = []
    for path in sorted((ROOT / '.cache/model_params/weights').glob('*.pth')):
        name = path.stem
        if args.variant and name not in args.variant:
            continue
        if 'minimpnn' in name:
            continue  # Separate graph/sequence oracle, never count it as denoiser coverage.
        config = config_for(name)
        model = CoordinateDenoiser(dict_to_namespace(config)).cpu().eval()
        original = checkpoint_state(path)
        state = {k.removeprefix('struct_model.'): v for k, v in original.items() if k.startswith('struct_model.')}
        # Old checkpoints contain an unused metadata buffer added by old rotary code.
        expected = set(model.state_dict())
        extras = set(state) - expected
        allowed = {k for k in extras if k.endswith('.dummy')}
        if extras - allowed:
            raise ValueError(f'Unexpected tensors {extras - allowed}')
        missing = expected - set(state)
        if any(not k.startswith('net.cond_to_patch_embedding.') for k in missing) or ('noise_residual' in config['model']['conditioning_style'] and missing):
            raise ValueError(f'Missing live weights: {missing}')
        unused_defaults = {k: model.state_dict()[k] for k in missing}
        model.load_state_dict(unused_defaults | {k: v for k, v in state.items() if k in expected}, strict=True)
        modes = ['unconditional', 'masked_partial']
        if config['model']['crop_conditional']:
            modes += ['backbone_motif', 'allatom_motif', 'sidechain_motif', 'tip_motif', 'multichain_hotspot']
        cases, reference = [], {}
        for mode in modes:
            length=32 if mode=='unconditional' else 8
            reference[mode]=run(model,inputs(config,mode,length))['denoised_coords'].clone()
        for precision in ('fp32', 'fp16', 'int8'):
            bundle = reload_bundle(ROOT / 'weights' / name / precision)
            model.load_state_dict(unused_defaults | {k.removeprefix('struct_model.'): v for k, v in bundle.items() if k.startswith('struct_model.') and k.removeprefix('struct_model.') in expected}, strict=True)
            for mode in modes:
                length = 32 if mode == 'unconditional' else 8
                data = inputs(config, mode, length)
                traces = run(model, data)
                target, actual = reference[mode].double(), traces['denoised_coords'].double()
                error = ((actual - target).square().sum() / target.square().sum().clamp(min=1e-30)).sqrt().item()
                metrics.append({'variant': name, 'mode': mode, 'precision': precision, 'output_relRMS': error})
                cases.append({'mode': mode, 'precision': precision, 'length': length, 'inputs': data, 'traces': traces})
        save_fixture(name, config, cases)
        print(f'{name}: {len(cases)} denoiser traces', flush=True)
    if not metrics:
        raise RuntimeError('No upstream traces generated')
    (ROOT / 'fixtures/denoiser_export_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    main()
