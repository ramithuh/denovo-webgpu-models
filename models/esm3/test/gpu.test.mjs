import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {runBrowser} from './browser.mjs';
test('complete ESM3 multimodal hardware WebGPU parity, all storage precisions',async()=>{const result={};for(const [precision,native] of [['fp32',0],['fp16',0],['fp16',1],['int8',0]]){const r=await runBrowser(`/test/gpu-harness.js?precision=${precision}&native=${native}`,{timeoutMs:1200000});assert.ok(r.ok,r.error);result[precision+(native?'-native':'')]=r;await writeFile(new URL('../fixtures/gpu.json',import.meta.url),JSON.stringify(result,null,2)+'\n');}});
