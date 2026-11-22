"""Naga/wgpu-native validation of shaders captured from actual dynamic execution."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['XDG_CACHE_HOME']=str(ROOT/'.cache/wgpu')
import wgpu
adapter=wgpu.gpu.request_adapter_sync(power_preference='high-performance')
assert adapter.info['adapter_type']!='CPU','Hardware is required'
device=adapter.request_device_sync(required_features=['shader-f16'],required_limits={'max-storage-buffers-per-shader-stage':8})
validated=[]
try:
 for path in sorted((ROOT/'.cache/wgsl').glob('*.wgsl')):
  module=device.create_shader_module(code=path.read_text())
  device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'main'})
  validated.append(path.name)
finally:device.destroy()
assert len(validated)>=20,'Incomplete shader capture'
result={'wgpu':wgpu.__version__,'adapter':dict(adapter.info),'validated':validated,'skipped':0}
(ROOT/'fixtures/wgsl_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
