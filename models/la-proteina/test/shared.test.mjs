import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,readdir} from 'node:fs/promises';
import {parseSafetensors,loadBundle,toFloat} from '../shared/loader.js';
import {tinyBundle} from './storage.js';
import {schedule,noiseSchedule} from '../shared/sampling.js';
import {acquireAdapter} from '../webgpu/device.js';
import {structure} from '../shared/pdb.js';
import {CPUAlgebra} from '../cpu/algebra.js';
import {timeEmbeddingTest,centeringTest} from './time-embedding.js';
import {fixture,compare} from './parity.js';

test('PyTorch time embeddings, centering and 768-channel normalization',async()=>{
  const read=async path=>{const b=await readFile(new URL('../'+path,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
  const ops=new CPUAlgebra();
  try{
    await timeEmbeddingTest(ops,JSON.parse(new TextDecoder().decode(await read('fixtures/time_embedding.json'))),read);
    await centeringTest(ops,JSON.parse(new TextDecoder().decode(await read('fixtures/centering.json'))),read);
    const metadata=JSON.parse(new TextDecoder().decode(await read('fixtures/linear.json'))),ref=await fixture(metadata,read);
    compare(await ops.read(ops.norm(ops.tensor(metadata.tensors.x.shape,ref.x))),ref.norm,'768-channel LayerNorm regression',1e-8);
  }
  finally{ops.destroy();}
});

test('strict safetensors, integrity, packed FP16 and INT8',async()=>{
  const {bytes,manifest}=await tinyBundle();
  assert.deepEqual([...parseSafetensors(bytes).w.data],[1,2,3,4]);
  const read=async url=>url.endsWith('manifest.json')?new TextEncoder().encode(JSON.stringify(manifest)).buffer:bytes.slice(0);
  assert.deepEqual([...(await loadBundle({baseUrl:'https://test',read})).weights.w.data],[1,2,3,4]);
  manifest.shards[0].sha256='0'.repeat(64);await assert.rejects(loadBundle({baseUrl:'https://test',read}),/integrity/);
  assert.throws(()=>parseSafetensors(new ArrayBuffer(7)),/Invalid/);
  assert.deepEqual([...toFloat({dtype:'F16',shape:[1,2],data:new Uint16Array([0x3c00,0xc000])})],[1,-2]);
  assert.deepEqual([...toFloat({dtype:'I8',shape:[1,2],data:new Int8Array([-127,127]),scale:new Float32Array([.5])})],[-63.5,63.5]);
});
test('exact schedules and pinned stochastic noise scale semantics',()=>{
  assert.deepEqual([...schedule(3,'log')],[0,0.7924813032150269,0.9632163047790527,1]);
  assert.deepEqual([...schedule(3,'power')],[0,0.111111119389534,0.4444443881511688,1]);
  assert.equal(schedule(400,'power')[203],0.2575562298297882);
  assert.equal(schedule(400,'power')[210],0.2756250500679016);
  assert.equal(noiseSchedule(0,'1/t'),100);assert(Math.abs(noiseSchedule(0,'tan')-157.080322265625)<1e-5);
  assert.throws(()=>schedule(0,'log'));assert.throws(()=>schedule(1,'other'));
});
test('PDB preserves fixed-width atom records for extreme short-step samples',()=>{
  const coords=new Float32Array(111).fill(-200),logits=new Float32Array(20);logits[0]=1;
  const out=structure({coords,logits});
  assert.deepEqual(out.pdbTranslationAngstrom,[2000,2000,2000]);
  assert(out.pdb.includes('COORDINATE OFFSET ANGSTROM'));
  for(const line of out.pdb.split('\n').filter(x=>x.startsWith('ATOM'))){assert.equal(line.length,80);assert.equal(Number(line.slice(30,38)),0);}
});
test('production adapter null retry, plain fallback, and strict software rejection',async()=>{
  const original=Object.getOwnPropertyDescriptor(globalThis,'navigator');
  try{
    let calls=0;const software={info:{vendor:'SwiftShader',isFallbackAdapter:true}};
    Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:{requestAdapter:async opts=>{calls++;return opts?null:software;}}}});
    const a=await acquireAdapter();assert.equal(calls,2);assert.equal(a.isFallbackAdapter,true);
    await assert.rejects(acquireAdapter({allowSoftwareFallback:false}),/software/);
    navigator.gpu.requestAdapter=async()=>null;await assert.rejects(acquireAdapter(),/retries/);
    const ctrl=new AbortController();ctrl.abort();await assert.rejects(acquireAdapter({signal:ctrl.signal}),/abort/i);
  }finally{if(original)Object.defineProperty(globalThis,'navigator',original);else delete globalThis.navigator;}
});
test('runtime modules use only relative ES imports and contain no Node APIs',async()=>{
  for(const dir of ['shared','cpu','webgpu'])for(const file of await readdir(new URL('../'+dir+'/',import.meta.url))){
    if(!file.endsWith('.js'))continue;
    const text=await readFile(new URL('../'+dir+'/'+file,import.meta.url),'utf8');
    assert(!/\b(?:require\s*\(|process\.|Buffer\.|node:)/.test(text),file);
    for(const m of text.matchAll(/(?:from\s*|import\s*)['"]([^'"]+)/g))assert(m[1].startsWith('.'),`${file}: ${m[1]}`);
  }
});
test('both complete exports and every oracle are required',async()=>{
  const exports=JSON.parse(await readFile(new URL('../fixtures/export_metrics.json',import.meta.url)));
  for(const model of ['LD1','AE1'])for(const precision of ['fp32','fp16','int8']){
    const m=exports.models[model][precision];assert(m.strict_load);assert.equal(m.roundtrip_maxAbs,0);assert(m.max_shard_bytes<=25000000);
  }
  const oracles=JSON.parse(await readFile(new URL('../fixtures/oracles.json',import.meta.url)));
  assert.equal(new Set(oracles.cases.map(c=>c.length)).size,2);
  assert(new Set(oracles.cases.map(c=>c.steps)).size>=2);
  for(const c of oracles.cases){const b=await readFile(new URL('../'+c.file,import.meta.url));assert.equal(b.byteLength,c.byteLength);}
});
