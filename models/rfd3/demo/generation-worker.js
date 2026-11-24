import { createRFD3Engine } from '../webgpu/index.js';
let engine,controller;
self.onmessage=async({data})=>{
 if(data.cancel){controller?.abort();return;}
 if(controller){postMessage({error:'A generation is already running'});return;}
 controller=new AbortController();
 try{
  const {precision='fp16',options}=data;
  if(!['fp32','fp16','int8'].includes(precision))throw new Error('Unknown precision');
  if(engine?.precision!==precision){engine?.destroy();engine=null;}
  if(!engine){
   postMessage({status:'Loading and verifying '+precision.toUpperCase()+' weights…'});
   const baseURL=new URL('../weights/'+precision+'/',import.meta.url);
   const [manifestResponse,architectureResponse]=await Promise.all([fetch(new URL('manifest.json',baseURL),{signal:controller.signal}),fetch(new URL('../fixtures/architecture.json',import.meta.url),{signal:controller.signal})]);
   if(!manifestResponse.ok||!architectureResponse.ok)throw new Error('Model assets are unavailable. Serve the rfd3 directory with its exported weights.');
   engine=await createRFD3Engine({manifest:await manifestResponse.json(),architecture:await architectureResponse.json(),baseURL,signal:controller.signal});
  }
  postMessage({status:'Preparing model…'});const started=performance.now();
  const result=await engine.generate({...options,signal:controller.signal,retainTrajectory:false,onStep:({step,total})=>postMessage({step:step+1,total,status:`Diffusion ${step+1} / ${total}`})});
  postMessage({done:true,pdb:result.pdb,mmcif:result.mmcif,seed:result.seed,seconds:(performance.now()-started)/1000,peakBytes:engine.peakMemoryUsage(),adapter:engine.resources.adapterInfo.description});
  engine.trim();
 }catch(error){postMessage({error:error.name==='AbortError'?'Generation cancelled.':error.message});engine?.destroy();engine=null;}
 finally{controller=null;}
};
