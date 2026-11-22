import {createLaProteinaEngine} from '../webgpu/index.js';
import {assert,fixture,sampleOptions,checkOutput,compare} from './parity.js';
import {storageTests} from './storage.js';
import {createLaProteinaWorker} from '../webgpu/worker-client.js';
import {acquireAdapter} from '../webgpu/device.js';
import {demoTest} from './demo.js';
import {GPUAlgebra} from '../webgpu/algebra.js';
import {timeEmbeddingTest,centeringTest} from './time-embedding.js';
const read=async p=>{const r=await fetch('/'+p);if(!r.ok)throw Error(`Missing fixture ${p}`);return r.arrayBuffer();};
export async function run(){
  assert(navigator.gpu,'Hardware WebGPU is required');
  const shaders=new Set(),createShader=GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule=function(descriptor){shaders.add(descriptor.code);return createShader.call(this,descriptor);};
  const response=await fetch('/fixtures/oracles.json');assert(response.ok,'Oracle manifest is required');
  const manifest=await response.json(),results=[],memory={},lifecycle={},timings={};let adapterInfo,timeEmbedding,linear,centering;
  const extraResponse=await fetch('/fixtures/extended_oracles.json');assert(extraResponse.ok,'Extended dynamic-shape fixtures are required');
  manifest.cases.push(...(await extraResponse.json()).cases);
  for(const precision of ['fp32','fp16','int8']){
    const loadStart=performance.now();
    const engine=await createLaProteinaEngine({baseUrl:location.origin+'/weights',precision,allowSoftwareFallback:false});
    adapterInfo=engine.adapterInfo;assert(!engine.isFallbackAdapter,'Fallback adapter forbidden');
    memory[precision]={loaded:engine.memoryUsage(),loadMs:performance.now()-loadStart};
    try{
      for(const c of manifest.cases.filter(c=>c.precision===precision)){
        await fetch('/log',{method:'POST',body:`starting ${c.name}`});
        const ref=await fixture(c,read),out=await engine.sample(sampleOptions(c,ref));
        results.push({name:c.name,timingMs:out.timingMs,parity:checkOutput(out,ref,c)});
        assert(engine.memoryUsage().transientBuffers===0,'Transient buffer leak after inference');
        await fetch('/log',{method:'POST',body:`PASS ${c.name} ${out.timingMs.toFixed(1)} ms`});
        if(c===manifest.cases.filter(x=>x.precision===precision)[0]){
          const warm=await engine.sample(sampleOptions(c,ref));checkOutput(warm,ref,c);
          timings[precision]={coldMs:out.timingMs,warmMs:warm.timingMs};
        }
      }
      let errors=0;
      try{await engine.sample({length:0,steps:1});}catch{errors++;}
      assert(errors===1,'Invalid input must fail');assert(engine.memoryUsage().transientBuffers===0,'Input-error leak');
      try{await engine.sample({length:32,steps:1,onProgress:p=>{if(p.phase==='ld1')throw Error('injected callback failure');}});}catch(e){assert(e.message.includes('injected'),'Unexpected error');errors++;}
      assert(errors===2,'Injected failure must propagate');assert(engine.memoryUsage().transientBuffers===0,'Mid-forward error leaked buffers');
      const controller=new AbortController();
      try{await engine.sample({length:32,steps:3,signal:controller.signal,onProgress:p=>{if(p.phase==='ld1')controller.abort();}});}catch(e){assert(e.name==='AbortError','Unexpected abort result');errors++;}
      assert(errors===3,'Mid-forward abort must fail');assert(engine.memoryUsage().transientBuffers===0,'Abort leaked buffers');
      const c=manifest.cases.find(c=>c.precision===precision),ref=await fixture(c,read);
      const recovered=await engine.sample(sampleOptions(c,ref));checkOutput(recovered,ref,c);
      assert(engine.memoryUsage().reusedBuffers>0,'Scratch buffers were not reused');
      lifecycle[precision]={normal:true,inputError:true,midForwardError:true,midForwardAbort:true,recovery:true,...engine.memoryUsage()};
    }finally{engine.destroy();engine.destroy();}
    assert(engine.memoryUsage().bytes===0,'GPU destroy leaked bytes');
  }
  assert(memory.fp16.loaded.bytes/memory.fp32.loaded.bytes<.51,'FP16 not packed');
  assert(memory.int8.loaded.bytes/memory.fp32.loaded.bytes<.26,'INT8 not packed');
  const storage=await storageTests();
  const worker=await createLaProteinaWorker({baseUrl:location.origin+'/weights',precision:'int8',allowSoftwareFallback:false});
  let workerResult;
  try{
    const c=manifest.cases.find(c=>c.precision==='int8'&&c.length===32&&c.steps===3&&c.mode==='sde'),ref=await fixture(c,read);
    const out=await worker.sample(sampleOptions(c,ref));
    workerResult={adapterInfo:worker.adapterInfo,parity:checkOutput(out,ref,c),memory:await worker.memoryUsage()};
    assert(workerResult.memory.transientBuffers===0,'Worker transient leak');
  }finally{const m=await worker.destroy();assert(m.bytes===0,'Worker destroy leak');await worker.destroy();}
  const {adapter}=await acquireAdapter({allowSoftwareFallback:false});
  assert(adapter.features.has('shader-f16'),'This hardware verification requires shader-f16');
  const device=await adapter.requestDevice({requiredFeatures:['shader-f16']});
  try{
    const timeOps=new GPUAlgebra(device);
    try{
      timeEmbedding=await timeEmbeddingTest(timeOps,JSON.parse(new TextDecoder().decode(await read('fixtures/time_embedding.json'))),read);
      centering=await centeringTest(timeOps,JSON.parse(new TextDecoder().decode(await read('fixtures/centering.json'))),read);
      const metadata=JSON.parse(new TextDecoder().decode(await read('fixtures/linear.json'))),ref=await fixture(metadata,read);
      const tensor=name=>timeOps.tensor(metadata.tensors[name].shape,ref[name]);
      linear=compare(await timeOps.read(timeOps.linear(tensor('x'),tensor('w'),tensor('b'))),ref.out,'bias accumulation regression',1e-7);
      lifecycle.normalization768=compare(await timeOps.read(timeOps.norm(tensor('x'))),ref.norm,'768-channel LayerNorm regression',1e-8);
    }
    finally{timeOps.destroy();}
    const external=await createLaProteinaEngine({baseUrl:location.origin+'/weights',precision:'int8',device,adapter,allowSoftwareFallback:false});
    external.destroy();external.destroy();assert(external.memoryUsage().bytes===0,'External engine buffers leaked');
    const probe=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST});device.queue.writeBuffer(probe,0,new Uint32Array([42]));await device.queue.onSubmittedWorkDone();probe.destroy();
    lifecycle.externalDeviceRetained=true;
    const ops=new GPUAlgebra(device),tensor=ops.upload([1],new Float32Array([1]));
    const submit=device.queue.submit;let failed=false;
    try{device.queue.submit=()=>{throw Error('readback injection');};await ops.read(tensor);}
    catch(e){assert(e.message==='readback injection','Unexpected readback error');failed=true;}
    finally{device.queue.submit=submit;ops.clear();}
    assert(failed&&ops.memoryUsage().bytes===0,'Readback error cleanup failed');ops.destroy();lifecycle.readbackError=true;
    const uploadOps=new GPUAlgebra(device),writeBuffer=device.queue.writeBuffer;failed=false;
    try{device.queue.writeBuffer=()=>{throw Error('weight upload injection');};uploadOps.uploadWeight({dtype:'F16',shape:[1,2],data:new Uint16Array([0,0])});}
    catch(e){assert(e.message==='weight upload injection','Unexpected upload error');failed=true;}
    finally{device.queue.writeBuffer=writeBuffer;uploadOps.destroy();}
    assert(failed&&uploadOps.memoryUsage().bytes===0,'Weight upload error cleanup failed');lifecycle.weightUploadError=true;
    const native=await createLaProteinaEngine({baseUrl:location.origin+'/weights',precision:'fp16',device,adapter,useF16:true,allowSoftwareFallback:false});
    try{
      const c=manifest.cases.find(c=>c.precision==='fp16'&&c.steps===3),ref=await fixture(c,read);
      const out=await native.sample(sampleOptions(c,ref));
      lifecycle.nativeF16={parity:checkOutput(out,ref,c),packedBytes:native.memoryUsage().bytes};
      assert(native.usesShaderF16,'Native FP16 path was not used');
    }finally{native.destroy();}
  }finally{device.destroy();}
  const saved=await fetch('/shaders',{method:'POST',body:JSON.stringify([...shaders])});assert(saved.ok,'Shader audit save failed');
  GPUDevice.prototype.createShaderModule=createShader;
  const demo=await demoTest();
  return {adapterInfo,cases:results,memory,lifecycle,timings,storage,worker:workerResult,demo,timeEmbedding,linear,centering,shaderVariants:shaders.size,skipped:0};
}
