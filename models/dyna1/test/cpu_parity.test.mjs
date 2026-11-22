import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createDyna1CPU} from '../cpu/dyna1/index.js';
import {diffStats,tokenize} from '../shared/dyna1/index.js';
const root=new URL('../',import.meta.url);
const read=async url=>{const b=await readFile(new URL(url,root));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
test('CPU FP32: all 35 transformer blocks and outputs match upstream on both proteins',async()=>{
 const meta=JSON.parse(new TextDecoder().decode(await read('fixtures/oracle_dyna1.json'))),buf=await read('fixtures/oracle_dyna1.bin');
 const engine=await createDyna1CPU({baseUrl:'weights/fp32',read});const metrics={};
 for(const [name,seq] of Object.entries(meta.sequences)) {
  const out=engine.predict(seq,{returnIntermediates:true});metrics[name]={timingMs:out.timingMs,tensors:{}};
  for(const [key,v] of Object.entries(out.intermediates)) {
   const rec=meta.tensors[name+'.'+key];assert.ok(rec,`Missing oracle ${name}.${key}`);
   const raw=buf.slice(rec.byteOffset,rec.byteOffset+rec.byteLength),ref=rec.dtype==='I32'?new Int32Array(raw):new Float32Array(raw);
   if(key==='tokens') assert.deepEqual(v,ref);else {const s=diffStats(v,ref);metrics[name].tensors[key]=s;console.log(name,key,s.relRms);assert.ok(s.relRms<1e-4,`${name}.${key}: ${JSON.stringify(s)}`);}
  }
 }
 engine.destroy();await writeFile(new URL('../fixtures/cpu_metrics.json',import.meta.url),JSON.stringify(metrics,null,2)+'\n');
});
test('canonical sequence validation and no BOS/EOS',()=>{
 assert.deepEqual(Array.from(tokenize(' A C \nD ').tokens),[5,23,13]);
 for(const seq of ['', 'AX','A>A','123']) assert.throws(()=>tokenize(seq));
});
