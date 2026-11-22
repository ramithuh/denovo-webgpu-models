import {loadWeights,decodeTensor,tokenize} from '../shared/dyna1/index.js';
function check(value,message){if(!value)throw new Error(message);}
export async function run(){
 const modes=[];
 for(const cache of ['cache','opfs']){
  if(cache==='opfs')check(!!navigator.storage.getDirectory,'OPFS unavailable');
  const baseUrl=new URL('/test/cache-fixture',location.href).href;
  const a=await loadWeights(baseUrl,{cache});check(Array.from(decodeTensor(a.weights.w)).join(',')==='0.5,-1,6,-8','INT8 decode failed');
  const original=globalThis.fetch;globalThis.fetch=()=>{throw new Error('Cache miss: attempted network');};
  try{const b=await loadWeights(baseUrl,{cache});check(Array.from(decodeTensor(b.weights.w)).join(',')==='0.5,-1,6,-8','Cached data changed');}finally{globalThis.fetch=original;}
  const ctrl=new AbortController();ctrl.abort();try{await loadWeights(baseUrl,{cache,signal:ctrl.signal});throw new Error('Abort ignored');}catch(e){check(e.name==='AbortError',e.message);}
  modes.push(cache);
 }
 const code=`import {tokenize} from '${new URL('/shared/dyna1/index.js',location.href)}';import {createDyna1Engine} from '${new URL('/webgpu/index.js',location.href)}';postMessage({tokens:Array.from(tokenize('ACD').tokens),engine:typeof createDyna1Engine});`;
 const url=URL.createObjectURL(new Blob([code],{type:'text/javascript'})),worker=new Worker(url,{type:'module'});
 try{const result=await new Promise((resolve,reject)=>{worker.onmessage=e=>resolve(e.data);worker.onerror=e=>reject(new Error(e.message));});check(result.tokens.join(',')==='5,23,13'&&result.engine==='function','Worker import failed');}finally{worker.terminate();URL.revokeObjectURL(url);}
 await caches.delete('dyna1-v1');await (await navigator.storage.getDirectory()).removeEntry('dyna1-v1',{recursive:true});
 return {modes,workerImports:true};
}
