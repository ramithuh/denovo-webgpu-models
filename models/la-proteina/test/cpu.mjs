import {readFile,writeFile} from 'node:fs/promises';
import {createLaProteinaCPU} from '../cpu/index.js';
import {fixture,sampleOptions,checkOutput,assert} from './parity.js';
const root=new URL('../',import.meta.url);
const read=async p=>{const b=await readFile(p instanceof URL?p:new URL(p,root));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const manifest=JSON.parse(await readFile(new URL('fixtures/oracles.json',root),'utf8'));
const engine=await createLaProteinaCPU({baseUrl:new URL('weights',root).href,read});
const results=[];
try{
  let cases=manifest.cases.filter(c=>c.precision==='fp32');
  assert(cases.length>0,'No CPU cases');
  for(const c of cases){
    const ref=await fixture(c,read),out=await engine.sample(sampleOptions(c,ref));
    const result={name:c.name,timingMs:out.timingMs,parity:checkOutput(out,ref,c)};
    results.push(result);console.log(c.name,out.timingMs,'ms PASS');
  }
}finally{engine.destroy();}
assert(engine.memoryUsage().bytes===0,'CPU destroy leak');
await writeFile(new URL('fixtures/cpu_metrics.json',root),JSON.stringify({cases:results,skipped:0},null,2)+'\n');
