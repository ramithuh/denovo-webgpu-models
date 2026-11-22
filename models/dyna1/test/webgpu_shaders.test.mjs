import test from 'node:test';import assert from 'node:assert/strict';import {runBrowser} from './browser.mjs';
test('Dawn compiles all WGSL storage variants',async()=>{const r=await runBrowser('/test/shader-harness.js');assert.ok(r.ok,r.error);console.log(r);});
