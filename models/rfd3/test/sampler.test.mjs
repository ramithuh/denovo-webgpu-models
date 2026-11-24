import test from 'node:test';
import assert from 'node:assert/strict';
import { noiseSchedule, resolveRandom, sequenceEntropy, sampleCPU } from '../shared/sampler.js';
import { createMonomerFeatures } from '../shared/features.js';
test('partial schedule rejects fewer than two points and invalid controls',()=>{
  assert.throws(()=>noiseSchedule({steps:1}),/schedule/);
  assert.throws(()=>noiseSchedule({partialT:0.006}),/two/);
  assert.throws(()=>noiseSchedule({partialT:NaN}),/positive/);
  const schedule=noiseSchedule({steps:20,partialT:10});
  assert.ok(schedule.length>=2&&schedule.every(x=>x<=10));
});
test('random draws persist resolved seeds, including zero',()=>{
  for(const seed of [0,1,123,0xffffffff]){
    const a=resolveRandom(seed),b=resolveRandom(seed);
    assert.equal(a.seed,seed);assert.deepEqual(a.normal(99),b.normal(99));
  }
  const auto=resolveRandom();assert.ok(Number.isInteger(auto.seed));
});
test('entropy includes all 32 classes, independent of sequence decoding mask',()=>{
  const entropy=sequenceEntropy(new Float32Array(32));
  assert.ok(Math.abs(entropy[0]-Math.log(32))<1e-6);
});
test('CFG applies for scales below and above one; fixed atoms remain fixed',()=>{
  const f=createMonomerFeatures({length:1}),r={...f,reference:true};
  f.is_motif_atom_with_fixed_coord[0]=1;f.motif_pos[0]=4;
  const engine={initialize:()=>({}),run:({features,X_noisy_L})=>({X_L:{shape:[1,14,3],data:Float32Array.from(X_noisy_L,(x,i)=>i<3?x:features.reference?0:1)},
    sequence_logits_I:{data:new Float32Array(32),shape:[1,1,32]}})};
  const random={initial:new Float32Array(42),steps:[new Float32Array(42)]};
  for(const scale of [0,0.5,1,2]){
    const out=sampleCPU(engine,{features:f,referenceFeatures:r,cfgScale:scale,schedule:[2,1],gamma0:0,stepScale:1,random,seed:0});
    assert.equal(out.X_L.data[0],4);
    assert.equal(out.X_L.data[3],0.5*scale);
  }
  assert.throws(()=>sampleCPU(engine,{features:f,symmetry:{},random}),/symmetry/);

});
