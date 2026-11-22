import { abort } from '../shared/loader.js';
export function adapterMetadata(adapter) {
  const info=adapter.info??{};
  const adapterInfo={vendor:info.vendor??'',architecture:info.architecture??'',device:info.device??'',description:info.description??''};
  const isFallbackAdapter=Boolean(info.isFallbackAdapter??adapter.isFallbackAdapter)||/swiftshader|llvmpipe|lavapipe|software|\bWARP\b|microsoft basic render/i.test(Object.values(adapterInfo).join(' '));
  return {adapterInfo,isFallbackAdapter};
}
export async function acquireAdapter({allowSoftwareFallback=true,signal}={}) {
  if(!globalThis.navigator?.gpu)throw Error('WebGPU unavailable');
  for(let attempt=0;attempt<3;attempt++){
    abort(signal);
    const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'})??await navigator.gpu.requestAdapter();
    if(adapter){
      const metadata=adapterMetadata(adapter);
      if(!allowSoftwareFallback&&metadata.isFallbackAdapter)throw Error('Hardware WebGPU required: software/fallback adapter rejected');
      return {adapter,...metadata};
    }
    await new Promise(r=>setTimeout(r,100));
  }
  throw Error('WebGPU adapter acquisition failed after retries');
}
