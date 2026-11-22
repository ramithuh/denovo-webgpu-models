import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createGVPTransformerCPU} from '../cpu/index.js';
import {compareOutput,checkReplay,oracleTensor} from './parity.js';
const read=async path=>{const b=await readFile(path);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
test('CPU FP32: every mode, intermediate, output, and exact autoregressive replay',{timeout:1800000},async()=>{
 const cases=JSON.parse(await readFile('fixtures/cases.json','utf8')),metrics={};
 for(const variant of ['antifold','esmif1']){
  const engine=await createGVPTransformerCPU({baseUrl:'weights',variant,read});
  try{for(const c of cases.filter(c=>c.variant===variant)){
   const meta=JSON.parse(await readFile(`fixtures/oracle_${c.name}_fp32.json`,'utf8')),bytes=await read(`fixtures/oracle_${c.name}_fp32.bin`),input=c.sourceFile?await readFile(c.sourceFile,'utf8'):c.nativeSequence?{...c,sequence:c.nativeSequence}:c,r=await engine[c.nativeSequence?'score':'predict'](input,{returnIntermediates:true,...(c.nativeSequence?{sequence:c.sequence}:{})});
   metrics[c.name]={traces:compareOutput(r,c,meta,bytes),timingMs:r.timingMs};
   const sample=await engine.sample(input,{...meta.replay,uniforms:oracleTensor(meta,bytes,'replay.uniforms'),returnIntermediates:true});metrics[c.name].replay=checkReplay(sample,meta,bytes);
   if(meta.independent){const independent=await engine.sampleIndependent(c,{...meta.independent,uniforms:oracleTensor(meta,bytes,'independent.uniforms')});assert.deepEqual(independent.tokens,oracleTensor(meta,bytes,'independent.tokens'));assert.deepEqual(independent.selected,oracleTensor(meta,bytes,'independent.mask'));metrics[c.name].independentReplay=true;}
   console.log(c.name,'passed',r.timingMs.toFixed(1),'ms');
  }
  await assert.rejects(()=>engine.predict({coords:[]}));await assert.rejects(()=>engine.predict(cases[0],{sequence:'AA'}));await assert.rejects(()=>engine.sample(cases[0],{uniforms:[2]}));
  const abort=new AbortController();abort.abort();await assert.rejects(()=>engine.predict(cases[0],{signal:abort.signal}),{name:'AbortError'});
  }finally{engine.destroy();}
  await assert.rejects(()=>engine.predict(cases[0]),/destroyed/);
 }
 await writeFile('fixtures/cpu_metrics.json',JSON.stringify(metrics,null,2)+'\n');
});
