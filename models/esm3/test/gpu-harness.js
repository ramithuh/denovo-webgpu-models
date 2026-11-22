import {createESM3Engine,requestWebGpuDevice} from '../webgpu/index.js';
import {checkCases,compare,oracleTensor,verifyOracle} from './oracle.js';
export async function run(){const params=new URL(import.meta.url).searchParams,precision=params.get('precision')??'fp32',native=params.get('native')==='1';const log=async text=>fetch('/log',{method:'POST',body:text});
 const {device,adapter}=await requestWebGpuDevice({useF16:native});const info=adapter.info;const adapterInfo={vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,isFallbackAdapter:info.isFallbackAdapter};
 if(!/nvidia/i.test(JSON.stringify(adapterInfo)))throw new Error('NVIDIA hardware required by this validation suite: '+JSON.stringify(adapterInfo));
 if(native&&!device.features.has('shader-f16'))throw new Error('Native F16 required but unavailable');
 const errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));const t=performance.now();
 const engine=await createESM3Engine({baseUrl:'/weights',precision,device,cache:'none',useF16:native,onProgress:p=>{globalThis.loadProgress=p;if(p.phase==='weights'&&p.loaded===p.total)log('loaded '+precision);}});const coldLoadMs=performance.now()-t,baseline=engine.memoryUsage();
 try{const meta=await (await fetch(`/fixtures/oracle-${precision}.json`)).json(),bin=await (await fetch(`/weights/oracle-${precision}.bin`)).arrayBuffer();
 const metrics=await checkCases(engine,meta,bin,{log,selected:params.get('cases')?.split(',')});
 if(engine.memoryUsage().totalBytes!==baseline.totalBytes||engine.memoryUsage().transientBuffers!==0)throw new Error('Transient buffer leak');
 const generation={};if(!params.has('cases')){
 const gmeta=await (await fetch(`/fixtures/generation-${precision}.json`)).json(),gbin=await (await fetch(`/weights/oracle-generation-${precision}.bin`)).arrayBuffer();
 await verifyOracle(gmeta,gbin);
 for(const [name,c] of Object.entries(gmeta.cases)){
  await log('generate '+name);let step=0;const metrics=[];const original=engine.forward.bind(engine);
  engine.forward=async(...args)=>{const r=await original(...args);const s=compare(r[c.track+'_logits'],oracleTensor(gmeta,gbin,`${name}/${step}/logits`));if(s.relRms>=1e-4)throw new Error(`${name} step ${step} logits: ${s.relRms}`);metrics.push(s);step++;return r;};
  let out;try{out=await engine.generate(c.input,{track:c.track,steps:c.steps,temperature:0});}finally{engine.forward=original;}
  if(JSON.stringify(out.history)!==JSON.stringify(c.history))throw new Error('Generation trajectory mismatch: '+name);
  if(c.track==='structure'){const s=compare(out.coordinates,oracleTensor(gmeta,gbin,name+'/coordinates'));if(s.relRms>=1e-4)throw new Error('Generated coordinates: '+s.relRms);generation[name]={steps:metrics,coordinates:s};}else generation[name]={steps:metrics};
 }
 }
 const bench={};if(!params.has('cases'))for(const length of [20,64,128]){const sequence='ACDEFGHIKLMNPQRSTVWY'.repeat(8).slice(0,length);await engine.forward(sequence);const times=[];for(let i=0;i<3;i++){const r=await engine.forward(sequence);times.push(r.timingMs);}bench[length]=times;}
 const controller=new AbortController();controller.abort();let aborted=false;try{await engine.forward('ACDE',{signal:controller.signal});}catch(e){aborted=e.name==='AbortError';}if(!aborted)throw new Error('Abort did not reject');
 const checkMemory=()=>{const now=engine.memoryUsage();if(now.totalBytes!==baseline.totalBytes||now.transientBuffers!==0)throw new Error('Transient leak after interrupted run');};
 const originalLinear=engine.b.linear.bind(engine.b);let calls=0;engine.b.linear=async(...args)=>{const r=await originalLinear(...args);if(++calls===7)throw new Error('Injected operation failure');return r;};
 let failed=false;try{await engine.forward('ACDEFG');}catch(e){failed=e.message.includes('Injected operation failure');}finally{engine.b.linear=originalLinear;}if(!failed)throw new Error('Error injection did not fail');checkMemory();
 const mid=new AbortController();calls=0;engine.b.linear=async(...args)=>{const r=await originalLinear(...args);if(++calls===7)mid.abort();return r;};
 let cancelled=false;try{await engine.forward('ACDEFG',{signal:mid.signal});}catch(e){cancelled=e.name==='AbortError';}finally{engine.b.linear=originalLinear;}if(!cancelled)throw new Error('Mid-run abort did not reject');checkMemory();
 await engine.forward('ACDEFG');checkMemory();
 const originalCreate=device.createBuffer.bind(device);let created=0,destroyed=0;device.createBuffer=(descriptor)=>{const buffer=originalCreate(descriptor);created++;const release=buffer.destroy.bind(buffer);let done=false;buffer.destroy=()=>{if(!done){destroyed++;done=true;}release();};return buffer;};
 let reads=0,loadFailed=false;try{await createESM3Engine({baseUrl:'/weights',precision,device,cache:'none',read:async url=>{if(++reads===3)throw new Error('Injected shard failure');const r=await fetch(url);return r.arrayBuffer();}});}catch(e){loadFailed=e.message.includes('Injected shard failure');}finally{device.createBuffer=originalCreate;}
 if(!loadFailed||created<2||created!==destroyed)throw new Error(`Load cleanup failed: ${created} created, ${destroyed} destroyed`);
 engine.destroy();engine.destroy();if(engine.memoryUsage().totalBytes!==0)throw new Error('Destroy leak');const probe=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST});probe.destroy();if(errors.length)throw new Error(errors.join('\n'));
 return {precision,nativeF16:native,adapterInfo,coldLoadMs,weightBytes:baseline.totalBytes,metrics,generation,bench,validationErrors:errors};
 }finally{engine.destroy();device.destroy();}
}
