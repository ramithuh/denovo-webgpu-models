import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
import {runBrowser} from './browser.mjs';
test('hardware shaders compile',async()=>{const r=await runBrowser('/test/shaders-harness.js');assert.ok(r.ok,r.error);});
test('Cache API, OPFS offline, private-mode fallback, real module Worker and folding demo',async()=>{const r=await runBrowser('/test/browser-features.js');assert.ok(r.ok,r.error);await writeFile(new URL('../fixtures/browser.json',import.meta.url),JSON.stringify(r,null,2)+'\n');});
