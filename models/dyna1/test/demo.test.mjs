import test from 'node:test';import assert from 'node:assert/strict';import {runBrowser} from './browser.mjs';
test('Standalone demo loads INT8, predicts, renders residues, and releases model',async()=>{const r=await runBrowser('/test/demo-harness.js');assert.ok(r.ok,r.error);console.log(r);});
