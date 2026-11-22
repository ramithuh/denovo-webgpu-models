export async function createLaProteinaWorker(options={}) {
  if(options.device||options.read)throw Error('Worker accepts fetch URLs, not GPUDevice or read callbacks');
  const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
  const pending=new Map();let nextId=1,closed=false;
  const failAll=error=>{for(const p of pending.values()){p.cleanup();p.reject(error);}pending.clear();};
  worker.addEventListener('error',event=>failAll(Error(event.message)));
  worker.addEventListener('message',({data})=>{
    const p=pending.get(data.id);if(!p)return;
    if(data.progress){p.onProgress?.(data.progress);return;}
    pending.delete(data.id);p.cleanup();
    if(data.error){const e=Error(data.error.message);e.name=data.error.name;p.reject(e);}else p.resolve(data.result);
  });
  const call=(method,args={},control={})=>new Promise((resolve,reject)=>{
    if(closed){reject(Error('Worker destroyed'));return;}
    const {signal,onProgress}=control;if(signal?.aborted){reject(signal.reason??new DOMException('Aborted','AbortError'));return;}
    const id=nextId++,cancel=()=>worker.postMessage({method:'abort',args:{id}});
    signal?.addEventListener('abort',cancel,{once:true});
    pending.set(id,{resolve,reject,onProgress,cleanup:()=>signal?.removeEventListener('abort',cancel)});
    worker.postMessage({id,method,args});
  });
  const {signal,onProgress,...init}=options;
  if(init.baseUrl)init.baseUrl=new URL(String(init.baseUrl),globalThis.location?.href??import.meta.url).href;
  let metadata;
  try{metadata=await call('init',init,{signal,onProgress});}
  catch(e){worker.terminate();throw e;}
  return {...metadata,
    sample(args={}){const {signal,onProgress,...input}=args;return call('sample',input,{signal,onProgress});},
    decode:(args,control)=>call('decode',args,control),
    denoiseStep:(args,control)=>call('denoiseStep',args,control),
    memoryUsage:()=>closed?Promise.resolve({bytes:0,transientBuffers:0}):call('memoryUsage'),
    async destroy(){if(closed)return;try{return await call('destroy');}finally{closed=true;worker.terminate();failAll(Error('Worker destroyed'));}},
  };
}
