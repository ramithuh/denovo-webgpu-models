import {createDyna1Engine,requestWebGpuDevice} from '../webgpu/index.js';
import {diffStats} from '../shared/dyna1/index.js';
const log=msg=>fetch('/log',{method:'POST',body:msg});
export async function run() {
 const params=new URL(import.meta.url).searchParams;
 const selectedPrecision=params.get('precision')??'fp32',selectedF16=params.get('native')==='1';
 const {adapter,device,hasF16}=await requestWebGpuDevice({useF16:selectedF16});const info=adapter.info;
 const adapterInfo={vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,isFallbackAdapter:info.isFallbackAdapter};
 if(info.isFallbackAdapter||/swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(adapterInfo))) throw new Error('Hardware GPU required: '+JSON.stringify(adapterInfo));
 await log(JSON.stringify(adapterInfo));
 const errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
 const meta=await (await fetch('/fixtures/oracle_dyna1.json')).json(),buf=await (await fetch('/fixtures/oracle_dyna1.bin')).arrayBuffer();
 const ref=(name)=>{const r=meta.tensors[name];const b=buf.slice(r.byteOffset,r.byteOffset+r.byteLength);return r.dtype==='I32'?new Int32Array(b):new Float32Array(b);};
 const metrics={},enabledFeatures={};
 for(const [precision,useF16] of [[selectedPrecision,selectedF16]]) {
  const mode=precision+(useF16?'-native':'-core');await log('Loading '+mode);let last=0;
  if(useF16&&!hasF16)throw new Error('shader-f16 unavailable for native-path test');
  const modeDevice=device;enabledFeatures[mode]=[...device.features];
  const engine=await createDyna1Engine({baseUrl:'/weights',precision,device:modeDevice,useF16,cache:'none',onProgress:p=>{if(p.loaded-last>500e6){last=p.loaded;log(mode+' '+Math.round(p.loaded/1e6)+' MB');}}});
  await log('Loaded '+mode);metrics[mode]={};
  try {
   const memoryBefore=engine.memoryUsage();
   const aborted=new AbortController();aborted.abort();
   try{await engine.predict('ACD',{signal:aborted.signal});throw new Error('Abort was ignored');}catch(e){if(e.name!=='AbortError')throw e;}
   for(const [name,sequence] of Object.entries(meta.sequences)) {
    const out=await engine.predict(sequence,{returnIntermediates:true});const tensors={};
    for(const [key,values] of Object.entries(out.intermediates)) {
     const s=diffStats(values,ref(name+'.'+key));tensors[key]=s;
     if(precision==='fp32'&&s.relRms>=1e-4) throw new Error(`${mode} ${name}.${key}: ${JSON.stringify(s)}`);
    }
    const threshold=precision==='fp16'?5e-4:precision==='fp32'?1e-4:.01;
    for(const key of ['logits','probs']) {
     const s=tensors[key];if(s.relRms>=threshold) throw new Error(`${mode} ${name}.${key}: ${JSON.stringify(s)}`);
     if(precision==='int8'&&s.correlation<=.99) throw new Error(`${mode} ${name}.${key} correlation: ${JSON.stringify(s)}`);
    }
    const times=[];for(let i=0;i<3;i++) times.push((await engine.predict(sequence)).timingMs);
    if(engine.memoryUsage().totalBytes!==memoryBefore.totalBytes) throw new Error('Activation buffers leaked');
    metrics[mode][name]={tensors,weightBytes:memoryBefore.totalBytes,coldMs:out.timingMs,warmMs:times,medianMs:[...times].sort((a,b)=>a-b)[1]};
    await log(mode+' '+name+' '+JSON.stringify({logits:tensors.logits,probs:tensors.probs,warmMs:times}));
   }
  } finally {engine.destroy();engine.destroy();if(engine.memoryUsage().bufferCount!==0)throw new Error('destroy leaked GPU buffers');}
  try{await engine.predict('ACD');throw new Error('Destroyed engine ran');}catch(e){if(e.message!=='Engine destroyed')throw e;}
  modeDevice.destroy();
 }
 device.destroy();if(errors.length) throw new Error(errors.join('\n'));
 return {adapterInfo,hasF16,enabledFeatures,metrics,validationErrors:errors};
}
