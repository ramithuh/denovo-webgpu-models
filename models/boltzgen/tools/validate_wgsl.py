"""Compile emitted WGSL with wgpu-native/Naga on the hardware Vulkan adapter."""
from environment import ROOT
import os,json,time
os.environ['WGPU_BACKEND_TYPE']='Vulkan'
os.environ['VK_ICD_FILENAMES']='/usr/share/vulkan/icd.d/nvidia_icd.json'
import wgpu
adapter=wgpu.gpu.request_adapter_sync(power_preference='high-performance')
assert 'cpu' not in str(adapter.info).lower() and 'nvidia' in str(adapter.info).lower(),adapter.info
device=adapter.request_device_sync()
codes=json.loads((ROOT/'.cache/shaders.json').read_text());start=time.perf_counter()
for i,code in enumerate(codes):
    try:
        module=device.create_shader_module(code=code)
        pipeline=device.create_compute_pipeline(layout='auto',compute={'module':module,'entry_point':'main'})
    except Exception:
        (ROOT/'.cache/naga-failed.wgsl').write_text(code)
        print('FAILED',i,flush=True)
        raise
    if i%200==0:print('NAGA',i,len(codes),flush=True)
record={'validator':'wgpu-native/Naga','wgpu_python':wgpu.__version__,'adapter':dict(adapter.info),'shaders':len(codes),'failures':0,'seconds':time.perf_counter()-start}
(ROOT/'fixtures/naga_validation.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
device.destroy()
