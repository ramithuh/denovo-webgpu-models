"""Compile every runtime WGSL variant through wgpu-native/Naga on hardware."""
import json
import os
from pathlib import Path
import subprocess
import wgpu

ROOT = Path(__file__).resolve().parents[1]
os.environ['XDG_CACHE_HOME'] = str(ROOT / '.cache/wgpu')
subprocess.run(['node', '--input-type=module', '-e', '''
import {writeFile,mkdir} from 'node:fs/promises';
import * as s from './webgpu/shaders.js';
await mkdir('.cache/wgsl',{recursive:true});
const variants={embedding:s.embedding,norm:s.norm,residual:s.residual,fp32:s.convolution('F32'),fp16:s.convolution('F16'),int8:s.convolution('I8'),native_f16:s.convolution('F16',true)};
for(const [name,code] of Object.entries(variants))await writeFile(`.cache/wgsl/${name}.wgsl`,code);
'''], cwd=ROOT, check=True)
adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
assert adapter.info['adapter_type'] != 'CPU', 'Software adapter is not acceptable'
device = adapter.request_device_sync(required_features=['shader-f16'])
validated = []
try:
    for path in sorted((ROOT / '.cache/wgsl').glob('*.wgsl')):
        module = device.create_shader_module(code=path.read_text())
        device.create_compute_pipeline(layout='auto', compute={'module': module, 'entry_point': 'main'})
        validated.append(path.name)
finally:
    device.destroy()
assert len(validated) == 7
result = {'wgpu_py': wgpu.__version__, 'adapter': dict(adapter.info), 'validated': validated, 'skipped': 0}
(ROOT / 'fixtures/wgsl_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
