import {createESM3Engine} from './index.js';
let engine,controller,busy=false;
self.onmessage=async({data})=>{
 const {id,method,input,options={}}=data;
 if(method==='cancel'){controller?.abort();return;}
 if(method==='destroy'){controller?.abort();engine?.destroy();engine=null;self.postMessage({id,result:{released:true}});return;}
 if(busy){self.postMessage({id,error:'Worker is busy'});return;}
 busy=true;controller=new AbortController();
 try{
  let result;
  if(method==='load'){engine?.destroy();engine=await createESM3Engine({...options,signal:controller.signal,onProgress:progress=>self.postMessage({id,progress})});result={memory:engine.memoryUsage()};}
  else {if(!engine)throw new Error('Load the model first');if(!['forward','generate','encodeStructure','decodeStructure'].includes(method))throw new Error('Unknown worker method');result=await engine[method](input,{...options,signal:controller.signal});}
  self.postMessage({id,result});
 }catch(e){self.postMessage({id,error:e.message,name:e.name});}finally{busy=false;controller=null;}
};
