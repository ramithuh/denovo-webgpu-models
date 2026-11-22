import {createLaProteinaEngine} from '../webgpu/index.js';
import {assert,fixture,sampleOptions,checkOutput} from './parity.js';

export async function run(){
  const r=await fetch('/fixtures/long_oracles.json');assert(r.ok,'400-step fixtures required');
  const manifest=await r.json(),results=[];
  const read=async p=>{const r=await fetch('/'+p);assert(r.ok,p);return r.arrayBuffer();};
  assert(manifest.cases.length===4,'All four long cases are required');
  for(const precision of ['int8','fp32']){
    const engine=await createLaProteinaEngine({baseUrl:location.origin+'/weights',precision,allowSoftwareFallback:false});
    try{
      for(const c of manifest.cases.filter(c=>c.precision===precision)){
        const ref=await fixture(c,read);
        const out=await engine.sample({...sampleOptions(c,ref),onProgress:p=>{
          if(p.phase==='sampling'&&p.step%100===0)fetch('/log',{method:'POST',body:`${c.name} step ${p.step}`});
        }});
        results.push({name:c.name,timingMs:out.timingMs,parity:checkOutput(out,ref,c),sequence:out.sequence});
        assert(engine.memoryUsage().transientBuffers===0,'Long-run buffer leak');
        await fetch('/log',{method:'POST',body:`PASS ${c.name}`});
      }
    }finally{engine.destroy();}
    assert(engine.memoryUsage().bytes===0,'Long-run destroy leak');
  }
  return {cases:results,fullTrajectory:true,skipped:0};
}
