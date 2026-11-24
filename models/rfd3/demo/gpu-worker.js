import { createPackedLinearRuntime } from '../webgpu/linear.js';
import { iterateWeights } from '../shared/weights.js';
self.onmessage=async({data:{precision}})=>{
  let runtime;
  try{
    if(!['fp32','fp16','int8'].includes(precision))throw new Error('Unknown precision');
    const baseURL=new URL('../weights/'+precision+'/',import.meta.url);
    const response=await fetch(new URL('manifest.json',baseURL));
    if(!response.ok)throw new Error('Weight bundle is unavailable; export weights before running this check.');
    const manifest=await response.json();runtime=await createPackedLinearRuntime();
    let weight;
    for await(const tensor of iterateWeights(manifest,baseURL,{names:['diffusion_module.process_r.weight']}))weight=runtime.uploadWeight(tensor);
    const input=Float32Array.from({length:96},(_,i)=>Math.sin(i));
    await runtime.linear(input,weight);
    const start=performance.now(),output=await runtime.linear(input,weight),milliseconds=performance.now()-start;
    postMessage({adapter:runtime.resources.adapterInfo.description,precision,milliseconds,
      gpuBufferBytes:runtime.resources.peakMemoryUsage(),outputCount:output.length,finite:output.every(Number.isFinite)});
  }catch(error){postMessage({error:error.message});}
  finally{runtime?.destroy();}
};
