import test from 'node:test';import assert from 'node:assert/strict';import {runBrowser} from './browser.mjs';
test('Cache API, OPFS, abort propagation, and browser worker imports',async()=>{const r=await runBrowser('/test/cache-harness.js');assert.ok(r.ok,r.error);console.log(r);});
