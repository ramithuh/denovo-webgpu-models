import {createGninaEngine,parseReceptor,parseLigands} from "../webgpu/index.js";
import {verify} from "./parity.js";
const root=new URL("../",import.meta.url),baseUrl=new URL("weights/default",root);
async function response(path){const r=await fetch(new URL(path,root));if(!r.ok) throw Error("Missing "+path);return r;}
const text=async p=>(await response(p)).text();
const floats=async p=>new Float32Array(await (await response(p)).arrayBuffer());
function assert(ok,msg){if(!ok) throw Error(msg);}
async function rejects(fn){try{await fn();}catch{return;}throw Error("Expected rejection");}
export async function suite(precision) {
  const start=performance.now(),engine=await createGninaEngine({baseUrl,precision,allowSoftwareFallback:false});
  const coldMs=performance.now()-start;
  try {
    const parity=await verify(engine,precision,text,floats);
    const rec=parseReceptor(await text("fixtures/184l_rec.pdb")), [lig]=parseLigands(await text("fixtures/184l_lig.sdf"));
    const warmMs=[];
    for(let i=0;i<3;i++){const t=performance.now();await engine.scorePose(rec,lig);warmMs.push(performance.now()-t);}
    const mem=engine.memoryUsage();
    const fail=async (fn)=>{await rejects(fn);assert(!engine.memoryUsage().transientBytes,"Leak on failure");};
    await fail(()=>engine.scorePose(rec,lig,{trace(){throw Error("injected trace failure");}}));
    const controller=new AbortController();
    await fail(()=>engine.scorePose(rec,lig,{signal:controller.signal,trace(){controller.abort();}}));
    const poses=await engine.scorePoses(rec,[lig,lig],{contributions:true});
    assert(poses.length===2 && poses[1].cnnScore===poses[0].cnnScore,"Batch mismatch");
    assert(Math.abs(poses[0].contributions.perAtom.reduce((s,n)=>s+n,0)-poses[0].vinaScore)<1e-6,"Vina breakdown mismatch");
    if(precision==="int8"){
      const key=rec.atoms.find(a=>a.x>lig.atoms[0].x-2 && Math.hypot(a.x-lig.atoms[0].x,a.y-lig.atoms[0].y,a.z-lig.atoms[0].z)<6).residue;
      const masked=await engine.maskContributions(rec,lig,{atomIndices:[lig.atoms[0].index],residues:[key]});
      assert(Math.abs(masked.atoms[0].cnnScore)>1e-7&&Math.abs(masked.residues[0].cnnScore)>1e-7,"Missing CNN ablation delta");
      const result=await engine.minimizePose(rec,lig,{iterations:1});
      assert(result.objective<=poses[0].vinaScore-Math.log(poses[0].cnnScore)+1e-7,"Minimizer increased objective");
    }
    engine.destroy();engine.destroy();
    assert(engine.memoryUsage().totalBytes===0,"Leak on destroy");
    await rejects(()=>engine.scorePose(rec,lig));
    return {adapterInfo:engine.adapterInfo,coldMs,warmMs,mem,afterDestroy:engine.memoryUsage(),parity};
  } finally{engine.destroy();}
}
export async function offline(mode) {
  let engine=await createGninaEngine({baseUrl,precision:"int8",allowSoftwareFallback:false,cache:mode});
  engine.destroy();
  // Worker gets its own device, no active engine or in-memory weights are shared.
  const worker=new Worker(new URL("webgpu/worker.js",root),{type:"module"});
  let seq=0;
  function call(method,...args) {
    const id=++seq;
    return new Promise((resolve,reject)=>{
      worker.onmessage=({data})=>data.error?reject(Error(data.error)):resolve(data.value);
      worker.onerror=e=>reject(Error(e.message));worker.postMessage({id,method,args});
    });
  }
  return {async run(rec,lig){
    try {
      await call("init",{baseUrl:baseUrl.href,precision:"int8",allowSoftwareFallback:false,cache:mode});
      const scores=await call("scorePose",rec,lig);
      const mem=await call("destroy");
      assert(mem.totalBytes===0,"Worker GPU leak");
      return scores.cnnScore;
    } finally{worker.terminate();}
  }};
}

export async function checkSharedDevice() {
  const device=await (await navigator.gpu.requestAdapter({powerPreference:"high-performance"})).requestDevice();
  const descriptor=Object.getOwnPropertyDescriptor(globalThis,"caches");
  const errors=[];
  device.addEventListener("uncapturederror",event=>errors.push(event.error.message));
  try {
    Object.defineProperty(globalThis,"caches",{configurable:true,get(){throw new DOMException("Private mode","SecurityError");}});
    const engine=await createGninaEngine({device,baseUrl,precision:"fp16",ensemble:["crossdock_default2018_KD_4"],cache:"cache",allowSoftwareFallback:false});
    const r=await text("fixtures/184l_rec.pdb"),l=await text("fixtures/184l_lig.sdf");
    const pending=engine.scorePose(r,l);engine.destroy();
    await rejects(()=>pending);
    assert(engine.memoryUsage().totalBytes===0,"Concurrent destruction leaked a buffer");
    const second=await createGninaEngine({device,baseUrl,precision:"fp16",ensemble:["crossdock_default2018_KD_4"],allowSoftwareFallback:false});
    const s=await second.scorePose(r,l);second.destroy();
    assert(Object.keys(s.models).length===1&&Number.isFinite(s.cnnScore),"External device reuse failed");
    assert(!errors.length,errors.join("\n"));
    return {privateStorage:true,externalDevice:true,destroyInFlight:true};
  }finally{
    if(descriptor) Object.defineProperty(globalThis,"caches",descriptor);else delete globalThis.caches;
    device.destroy();
  }
}
