import {createGVPTransformerEngine} from '../webgpu/index.js';
import {parseStructure} from '../shared/index.js';
self.onmessage=async({data})=>{
 let engine;
 try{
  let networkCalls=0;const nativeFetch=globalThis.fetch.bind(globalThis);
  globalThis.fetch=(...args)=>{networkCalls++;if(data.offline)throw new Error('Network blocked for offline test');return nativeFetch(...args);};
  if(data.failure){
   const fail=()=>{throw new DOMException('Simulated unavailable storage',data.failure);};
   if(data.cache==='cache')Object.defineProperty(globalThis,'caches',{value:{open:data.failure==='SecurityError'?fail:async()=>({match:async()=>undefined,put:fail})},configurable:true});
   else Object.defineProperty(navigator.storage,'getDirectory',{value:data.failure==='SecurityError'?fail:async()=>({getDirectoryHandle:async()=>({getFileHandle:async(key,options)=>{if(!options?.create)throw new DOMException('Missing cached object','NotFoundError');return {createWritable:fail};}})}),configurable:true});
  }
  let warnings=0;
  engine=await createGVPTransformerEngine({baseUrl:'/weights',variant:'antifold',precision:'int8',cache:data.cache,onProgress:p=>{if(p.phase==='cache-warning')warnings++;}});
  const result=await engine.predict(data.input);const normal=engine.memoryUsage();engine.destroy();const after=engine.memoryUsage();
  if(data.offline&&networkCalls!==0)throw new Error('Offline cache made network requests');if(data.failure&&warnings===0)throw new Error('Storage failure not exercised');
  if(normal.transientBuffers!==0||after.totalBytes!==0)throw new Error('Worker memory leak');
  self.postMessage({ok:true,logits:result.logits,networkCalls,warnings,normal,after},[result.logits.buffer]);
 }catch(e){engine?.destroy();self.postMessage({ok:false,error:e.stack||String(e)});}
};
