import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createMonomerFeatures, validateFeatures } from '../shared/features.js';
for(const name of ['monomer','cyclic'])test('all '+name+' featurizer tensors exactly match upstream',async()=>{
  const oracle=JSON.parse(await readFile(new URL('../fixtures/oracle/'+name+'.json',import.meta.url)));
  const actual=createMonomerFeatures({length:4,cyclic:name==='cyclic'});
  assert.deepEqual(Object.keys(actual).sort(),Object.keys(oracle.features).sort());
  for(const [key,value]of Object.entries(oracle.features))assert.deepEqual(actual[key],new Float32Array(value.data),key);
});
test('monomer featurizer builds dynamic token and atom counts',()=>{
  for(const length of [1,3,7,12]){
    const f=createMonomerFeatures({length});
    assert.equal(f.atom_to_token_map.length,length*14);
    assert.equal(f.is_ca.reduce((a,b)=>a+b,0),length);
  }
  assert.throws(()=>createMonomerFeatures({length:2,cyclic:true}),/three/);
});
test('unsupported short multichain inputs fail before GPU neighbor dispatch',async()=>{
  const {config}=JSON.parse(await readFile(new URL('../fixtures/architecture.json',import.meta.url)));
  const short=createMonomerFeatures({length:12});
  short.asym_id=Float32Array.from({length:12},(_,i)=>Math.floor(i/3));
  assert.throws(()=>validateFeatures(short,config),/at least 32 tokens/);
  const supported=createMonomerFeatures({length:32});
  supported.asym_id=Float32Array.from({length:32},(_,i)=>Math.floor(i/8));
  assert.deepEqual(validateFeatures(supported,config),{tokens:32,atoms:448});
});
