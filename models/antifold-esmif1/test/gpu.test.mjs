import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {runBrowser} from './browser.mjs';
test('Real hardware WebGPU: both models, all precisions, all modes, traces, sampling and memory',{timeout:1800000},async()=>{
 const result=await runBrowser('/test/gpu-harness.js');assert.equal(result.ok,true,result.error);await writeFile('fixtures/gpu_metrics.json',JSON.stringify(result,null,2)+'\n');
});
test('Browser module workers, offline Cache/OPFS, private storage errors, cancellation, ownership and admission',{timeout:1800000},async()=>{
 const result=await runBrowser('/test/lifecycle-harness.js');assert.equal(result.ok,true,result.error);await writeFile('fixtures/browser_metrics.json',JSON.stringify(result,null,2)+'\n');
});
test('Static demo: load structure, run ESM-IF1 on WebGPU, render results, release GPU',{timeout:300000},async()=>{
 const result=await runBrowser('/test/demo-harness.js',{timeoutMs:300000});assert.equal(result.ok,true,result.error);
});
