"""Validate the exact modular shaders exercised in hardware tests using Naga."""
import hashlib
import json
from pathlib import Path
import wgpu
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'.cache/shaders'
files=json.loads((folder/'manifest.json').read_text())
adapter=wgpu.gpu.request_adapter_sync(power_preference='high-performance')
assert adapter is not None,'Hardware wgpu adapter required'
assert not any(x in str(adapter.info).lower() for x in ['swiftshader','llvmpipe','lavapipe','cpu']),adapter.info
device=adapter.request_device_sync(required_features=['shader-f16'])
try:
    for file in files:
        code=(folder/file).read_text()
        assert hashlib.sha256(code.encode()).hexdigest()+'.wgsl'==file
        device.create_shader_module(code=code)
    result={'frontend':'wgpu-native/Naga','wgpu_version':wgpu.__version__,'adapter':dict(adapter.info),'passed':len(files),'failed':0,'skipped':0}
    (ROOT/'fixtures/wgsl_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
finally:device.destroy()
