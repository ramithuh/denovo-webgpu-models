import {createLaProteinaEngine} from './index.js';
let engine;const controllers=new Map();
self.addEventListener('message',async({data})=>{
  const {id,method,args}=data;
  if(method==='abort'){controllers.get(args.id)?.abort();return;}
  const controller=new AbortController();controllers.set(id,controller);
  const onProgress=progress=>self.postMessage({id,progress});
  try{
    let result;
    if(method==='init'){
      if(engine)throw Error('Worker already initialized');
      engine=await createLaProteinaEngine({...args,signal:controller.signal,onProgress});
      result={precision:engine.precision,variant:engine.variant,variants:engine.variants,adapterInfo:engine.adapterInfo,isFallbackAdapter:engine.isFallbackAdapter,memory:engine.memoryUsage()};
    }else{
      if(!engine)throw Error('Worker is not initialized');
      if(method==='sample')result=await engine.sample({...args,signal:controller.signal,onProgress});
      else if(['decode','encode','encodeDecode','denoiseStep'].includes(method))result=await engine[method](args,{signal:controller.signal,onProgress});
      else if(method==='memoryUsage')result=engine.memoryUsage();
      else if(method==='destroy'){for(const c of controllers.values())c.abort();engine.destroy();result=engine.memoryUsage();engine=null;}
      else throw Error(`Unknown worker method ${method}`);
    }
    const buffers=new Set();
    const collect=v=>{if(ArrayBuffer.isView(v))buffers.add(v.buffer);else if(v&&typeof v==='object')for(const value of Object.values(v))collect(value);};
    collect(result);self.postMessage({id,result},[...buffers]);
  }catch(e){self.postMessage({id,error:{name:e.name,message:e.message,stack:e.stack}});}
  finally{controllers.delete(id);}
});
