import {createGVPTransformerEngine,requestWebGPU} from '../webgpu/index.js';
import {diffStats} from '../shared/index.js';
import {oracleTensor} from './parity.js';
const assert=(ok,message)=>{if(!ok)throw new Error(message);};
async function rejects(fn,pattern){try{await fn();}catch(e){if(!pattern||pattern.test(String(e)))return;throw e;}throw new Error('Expected rejection');}
function trackedDevice(device){const live=new Set();return {live,proxy:new Proxy(device,{get(target,key){if(key==='createBuffer')return desc=>{const b=target.createBuffer(desc),destroy=b.destroy.bind(b);live.add(b);Object.defineProperty(b,'destroy',{value:()=>{live.delete(b);destroy();}});return b;};const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;}})};}
async function worker(data){const w=new Worker('/test/runtime-worker.js',{type:'module'});try{return await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('Worker timed out')),180000);w.onmessage=({data})=>{clearTimeout(timer);data.ok?resolve(data):reject(new Error(data.error));};w.onerror=e=>{clearTimeout(timer);reject(new Error(e.message));};w.postMessage(data);});}finally{w.terminate();}}
export async function run(){
 const cases=await(await fetch('/fixtures/cases.json')).json(),input=cases.find(c=>c.name==='nanobody'),meta=await(await fetch('/fixtures/oracle_nanobody_int8.json')).json(),bytes=await(await fetch('/fixtures/oracle_nanobody_int8.bin')).arrayBuffer();
 const {device,adapter}=await requestWebGPU({useF16:true}),tracked=trackedDevice(device),checks={};let engine;
 try{
  let fault=null,controller;
  engine=await createGVPTransformerEngine({baseUrl:'/weights',variant:'antifold',precision:'int8',device:tracked.proxy,onProgress:p=>{if(fault==='throw'&&p.phase==='inference')throw new Error('Injected inference failure');if(fault==='abort'&&p.phase==='inference')controller.abort();if(fault==='submitted'&&p.phase==='submitted')controller.abort();}});
  const assertClean=()=>assert(engine.memoryUsage().transientBuffers===0,'Transient GPU allocation leak');
  await engine.predict(input);assertClean();checks.normal=true;
  await rejects(()=>engine.predict(input,{sequence:'AA'}),/lengths/);assertClean();checks.invalid=true;
  fault='throw';await rejects(()=>engine.predict(input),/Injected/);assertClean();checks.thrown=true;
  for(const stage of ['abort','submitted']){fault=stage;controller=new AbortController();await rejects(()=>engine.predict(input,{signal:controller.signal}),/Abort/);assertClean();checks[stage]=true;}
  fault=null;const running=engine.predict(input);await rejects(()=>engine.predict(input),/already running/);await running;assertClean();checks.overlap=true;
  engine.destroy();engine.destroy();assert(engine.memoryUsage().totalBytes===0&&tracked.live.size===0,'Destroy leaked buffers');await rejects(()=>engine.predict(input),/destroyed/);checks.destroy=true;
  const probe=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST});probe.destroy();checks.externalDevice=true;
  controller=new AbortController();await rejects(()=>createGVPTransformerEngine({baseUrl:'/weights',variant:'antifold',precision:'int8',device:tracked.proxy,signal:controller.signal,onProgress:p=>{if(p.phase==='weights')controller.abort();}}),/Abort/);assert(tracked.live.size===0,'Aborted weight load leaked buffers');checks.loadAbort=true;
  const admissionAdapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});const limited=await admissionAdapter.requestDevice();const limitedTracked=trackedDevice(limited);
  let limitedEngine;
  try{limitedEngine=await createGVPTransformerEngine({baseUrl:'/weights',variant:'antifold',precision:'int8',device:limitedTracked.proxy});const coords=new Float32Array(2048*9);for(let i=0;i<2048;i++)coords.set(input.coords[i%input.coords.length].flat(),i*9);await rejects(()=>limitedEngine.predict({coords,sequence:'A'.repeat(2048)}),/maxBufferSize|maxStorageBufferBindingSize/);assert(limitedEngine.memoryUsage().transientBuffers===0,'Admission dispatched inference');checks.admission=true;}finally{limitedEngine?.destroy();assert(limitedTracked.live.size===0,'Admission failure leaked buffers');limited.destroy();}
  // Exercise shader-f16 storage loads separately from the portable unpack path.
  const fp16=await createGVPTransformerEngine({baseUrl:'/weights',variant:'antifold',precision:'fp16',device,useF16:true});
  try{const r=await fp16.predict(input),m=await(await fetch('/fixtures/oracle_nanobody_fp16.json')).json(),b=await(await fetch('/fixtures/oracle_nanobody_fp16.bin')).arrayBuffer();assert(diffStats(r.logits,oracleTensor(m,b,'logits')).relRMS<1e-4,'shader-f16 parity');checks.shaderF16=true;}finally{fp16.destroy();}
 }finally{engine?.destroy();device.destroy();}
 for(const cache of ['cache','opfs']){
  const prime=await worker({cache,input}),offline=await worker({cache,input,offline:true});assert(diffStats(offline.logits,oracleTensor(meta,bytes,'logits')).relRMS<1e-4,'Worker offline parity');checks[cache]={primeRequests:prime.networkCalls,offlineRequests:offline.networkCalls};
  for(const failure of ['SecurityError','QuotaExceededError']){const result=await worker({cache,input,failure});assert(diffStats(result.logits,oracleTensor(meta,bytes,'logits')).relRMS<1e-4,'Unavailable-storage parity');checks[cache][failure]=result.warnings;}
 }
 return {checks};
}
