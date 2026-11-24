import { createGpuResources } from './resources.js';
import { iterateWeights } from '../shared/weights.js';

// Model-wide packed residency, without expanded FP32 shadow copies.
export async function loadGPUWeights({ manifest, baseURL, resources, ...options }) {
  const owner=resources??await createGpuResources(options),{device}=owner,tensors=new Map(),buffers=[];
  let bytes=0,closed=false;
  function upload(data,label){
    const buffer=owner.allocate(data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label);
    buffers.push(buffer);bytes+=buffer.size;
    const padded=new Uint8Array(buffer.size);padded.set(data);device.queue.writeBuffer(buffer,0,padded);
    return buffer;
  }
  device.pushErrorScope('validation');
  let popped=false;
  try{
    for await(const tensor of iterateWeights(manifest,baseURL,options)){
      options.signal?.throwIfAborted();
      tensors.set(tensor.name,{shape:tensor.shape,dtype:tensor.dtype,buffer:upload(tensor.bytes,tensor.name),
        scales:tensor.scales?upload(tensor.scales,tensor.name+' scales'):undefined});
    }
    await device.queue.onSubmittedWorkDone();options.signal?.throwIfAborted();
    const error=await device.popErrorScope();popped=true;if(error)throw new Error(error.message);
    return {resources:owner,tensors,byteLength:bytes,
      destroy(){if(closed)return;closed=true;for(const b of buffers)owner.release(b);tensors.clear();if(!resources)owner.destroy();}};
  }catch(error){
    if(!popped)await device.popErrorScope();
    for(const b of buffers)owner.release(b);if(!resources)owner.destroy();throw error;
  }
}
