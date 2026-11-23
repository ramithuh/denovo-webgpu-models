import {createGninaEngine} from "./index.js";
let engine,initializing=false;
const pending=new Map();
self.onmessage=async ({data:{id,method,args=[]}})=>{
  if(method==="abort"){pending.get(id)?.abort();return;}
  const controller=new AbortController();
  pending.set(id,controller);
  try {
    let value;
    if(method==="init"){
      if(engine||initializing) throw Error("Engine is already initialized");
      initializing=true;
      try {
        engine=await createGninaEngine({...args[0],signal:controller.signal});
        value={adapterInfo:engine.adapterInfo,memory:engine.memoryUsage()};
      } finally {initializing=false;}
    } else if(method==="destroy"){
      for(const [other,c] of pending) if(other!==id) c.abort();
      engine?.destroy();value=engine?.memoryUsage();engine=null;
    } else if(["dockPose","scorePose","scorePoses","minimizePose","maskContributions"].includes(method)){
      if(!engine) throw Error("Initialize engine first");
      value=await engine[method](args[0],args[1],{...args[2],signal:controller.signal});
    } else throw Error("Unknown worker method");
    self.postMessage({id,value});
  } catch(error){self.postMessage({id,error:String(error)});}
  finally {pending.delete(id);}
};
