#!/usr/bin/env python3
"""Compare quantized upstream trajectories to the original FP32 weight bundle."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def trajectory(precision, mode):
    meta = json.loads((ROOT / 'fixtures' / f'trajectory_{precision}_{mode}_L24.json').read_text())
    binary = (ROOT / 'fixtures' / meta['binary']).read_bytes()
    assert len(binary) == meta['binary_byteLength']
    assert hashlib.sha256(binary).hexdigest() == meta['binary_sha256']
    def array(entry):
        return np.frombuffer(binary, dtype='<f4', offset=entry['offset'], count=entry['length']).astype(np.float64)
    return meta, array

rows = []
for mode in ['unconditional', 'motif', 'symmetric']:
    reference, ref_array = trajectory('fp32', mode)
    for precision in ['fp32', 'fp16', 'int8']:
        meta, array = trajectory(precision, mode)
        assert np.array_equal(array(meta['initial']), ref_array(reference['initial']), equal_nan=True)
        for index, (actual, expected) in enumerate(zip(meta['steps'], reference['steps'])):
            for key in ['translation', 'rotationNoise']:
                assert np.array_equal(array(actual[key]), ref_array(expected[key]))
            x = array(actual['px0']).reshape(-1, 4, 3)
            y = ref_array(expected['px0']).reshape(-1, 4, 3)
            delta = x - y
            rows.append(dict(mode=mode, precision=precision, step=index + 1, t=actual['t'],
                relRMS=float(np.sqrt(np.sum(delta ** 2) / np.sum(y ** 2))),
                caRmsdAngstrom=float(np.sqrt(np.mean(np.sum(delta[:, 1] ** 2, axis=-1))))))

(ROOT / 'fixtures/trajectory_quantization_metrics.json').write_text(json.dumps(dict(
    reference='Pinned upstream FP32 trajectory; identical initial coordinates and Gaussian noise verified',
    alignment='none', coordinates=rows), indent=2) + '\n')
for row in rows:
    if row['step'] == 3:
        print(row)
