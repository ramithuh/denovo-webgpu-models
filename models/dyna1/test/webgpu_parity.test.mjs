import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {runBrowser} from './browser.mjs';
test('Real WebGPU full model parity and latency, all precisions',async()=>{
 const combined={ok:true,metrics:{},enabledFeatures:{},validationErrors:[]};
 // Isolate Dawn device lifetimes across precision modes, including feature-disabled devices.
 for(const [precision,native] of [['fp32',0],['fp16',0],['fp16',1],['int8',0]]){
  const r=await runBrowser(`/test/gpu-harness.js?precision=${precision}&native=${native}`);
  assert.ok(r.ok,r.error);combined.adapterInfo=r.adapterInfo;combined.hasF16=r.hasF16;
  Object.assign(combined.metrics,r.metrics);Object.assign(combined.enabledFeatures,r.enabledFeatures);combined.validationErrors.push(...r.validationErrors);
 }
 await writeFile(new URL('../fixtures/gpu_metrics.json',import.meta.url),JSON.stringify(combined,null,2)+'\n');
 console.log(JSON.stringify(combined.adapterInfo));
});
