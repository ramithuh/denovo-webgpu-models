"""Verify local checkpoint provenance and architecture compatibility."""
from environment import ROOT, CHECKPOINTS, COMMIT, REVISION, checkpoint
import hashlib
import json
from collections import Counter
import torch
from omegaconf import OmegaConf

EXPECTED = {
    'diverse': '360af8bd6e59527ff6ec25dd81253967f3bd3567d200053b10680634751f8e3c',
    'adherence': 'ac7078b3dc13064c68e0c3fd542e5bc538c33558bf6607f65e499eb336ca5e5d',
}

def plain(x):
    if OmegaConf.is_config(x):
        return plain(OmegaConf.to_container(x, resolve=False))
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [plain(v) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)

def main():
    records = {}
    schemas = []
    for variant, sha in EXPECTED.items():
        path = CHECKPOINTS / f'boltzgen1_{variant}.ckpt'
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        assert actual == sha, (variant, actual)
        ckpt = checkpoint(variant)
        state = ckpt['state_dict']
        schema = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in state.items()}
        schemas.append(schema)
        groups = Counter()
        for k, v in state.items():
            groups[k.split('.')[0]] += v.numel()
        records[variant] = {'sha256': actual, 'byteLength': path.stat().st_size,
            'num_tensors': len(state), 'total_params': sum(v.numel() for v in state.values()),
            'groups': dict(groups), 'hyper_parameters': plain(ckpt['hyper_parameters'])}
        print(variant, records[variant]['total_params'], len(state), flush=True)
    assert schemas[0] == schemas[1], 'Checkpoint tensor schemas differ'
    out = {'upstream_commit': COMMIT, 'checkpoint_revision': REVISION,
        'schema_compatible': True, 'checkpoints': records, 'tensors': schemas[0]}
    (ROOT / 'fixtures/architecture.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(records['diverse']['hyper_parameters'], indent=2))

if __name__ == '__main__':
    main()
