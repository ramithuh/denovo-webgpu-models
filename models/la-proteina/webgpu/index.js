import { GPUAlgebra } from './algebra.js';
import { makeEngine } from '../shared/engine.js';
import { acquireAdapter, adapterMetadata } from './device.js';
export async function createLaProteinaEngine(options={}) {
  const owned=!options.device;let device=options.device;
  let metadata=options.adapter?adapterMetadata(options.adapter):{adapterInfo:options.adapterInfo??null,isFallbackAdapter:options.isFallbackAdapter??null};
  if(owned){
    let lastErr;
    for(let attempt=0;attempt<4;attempt++){
      try{
        const acquired=await acquireAdapter(options);metadata={adapterInfo:acquired.adapterInfo,isFallbackAdapter:acquired.isFallbackAdapter};
        const adapter=acquired.adapter;
        device=await adapter.requestDevice({requiredLimits:{maxBufferSize:adapter.limits.maxBufferSize,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize},
          requiredFeatures:options.useF16&&adapter.features.has('shader-f16')?['shader-f16']:[]});
        await device.queue.onSubmittedWorkDone();
        lastErr=null;break;
      }catch(e){
        if(device){try{device.destroy();}catch{}device=null;}
        lastErr=e;
        if(options.signal?.aborted||/software\/fallback/i.test(String(e)))throw e;
        await new Promise(r=>setTimeout(r,100));
      }
    }
    if(lastErr)throw lastErr;
  }else if(options.allowSoftwareFallback===false&&(metadata.isFallbackAdapter!==false||!metadata.adapterInfo)){
    throw Error('Strict external-device mode requires verified hardware adapter metadata');
  }
  try{
    const engine=await makeEngine(new GPUAlgebra(device,{useF16:options.useF16}),options),destroy=engine.destroy;
    let released=false;
    engine.destroy=()=>{if(released)return;released=true;destroy();if(owned)device.destroy();};
    return Object.assign(engine,metadata,{usesShaderF16:Boolean(options.precision==='fp16'&&options.useF16&&device.features.has('shader-f16'))});
  }catch(e){if(owned)device.destroy();throw e;}
}
