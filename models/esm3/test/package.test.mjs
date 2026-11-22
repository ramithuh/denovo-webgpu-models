import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,readdir,rm} from 'node:fs/promises';
import {tokenize,prepareInput,parsePDB} from '../shared/input.js';
import {parseSafetensors,loadWeights} from '../shared/weights.js';
import {vendorSync} from '../tools/vendor-sync.mjs';
const root=new URL('../',import.meta.url);
test('tokenization, coordinate gaps, tracks, errors, and standalone runtime imports',async()=>{
 assert.deepEqual([...tokenize('AC_D:EF')],[0,5,23,32,13,31,9,18,2]);
 const pdb=parsePDB(await readFile(new URL('fixtures/1l2y.pdb',root),'utf8'));assert.equal(pdb.sequence,'NLYIQWLKDGGPSSGRPPPS');const p=prepareInput(pdb);assert.equal(p.L,22);assert.equal(p.structure_coords.length,198);assert.ok(Number.isNaN(p.structure_coords[0]));assert.equal(p.encodeStructure,true);
 const multi=prepareInput({sequence:'AC',ss8:'GH',sasa:[.8,151.5]});assert.deepEqual([...multi.ss8_tokens],[0,3,4,0]);assert.deepEqual([...multi.sasa_tokens],[0,3,18,0]);
 assert.throws(()=>prepareInput({sequence:'ABC!',coordinates:[]}),/Invalid/);assert.throws(()=>prepareInput({sequence:'AC',ss8_tokens:[0]}),/expected/);assert.throws(()=>parseSafetensors(new ArrayBuffer(8)),/Invalid/);
 for(const dir of ['shared','cpu','webgpu'])for(const file of await readdir(new URL(dir+'/',root))){if(!file.endsWith('.js'))continue;const s=await readFile(new URL(dir+'/'+file,root),'utf8');assert.doesNotMatch(s,/from\s+['"](?:node:|https?:|@|[a-z][\w-]*['"])/);assert.doesNotMatch(s,/require\s*\(/);}
});
test('missing weights and corrupt shards reject without a successful result',async()=>{
 await assert.rejects(loadWeights('/missing',{read:async()=>{throw new Error('missing');}}),/missing/);
 const manifest=JSON.parse(await readFile(new URL('weights/int8/manifest.json',root)));const raw=new TextEncoder().encode(JSON.stringify(manifest)).buffer;
 await assert.rejects(loadWeights('/test',{read:async u=>u.endsWith('manifest.json')?raw:new ArrayBuffer(12)}),/size mismatch/);
});
test('vendor sync includes only self-contained runtime modules',async()=>{
 const dest=new URL('.cache/vendor-smoke',root).pathname;await vendorSync(dest);const gpu=await import(dest+'/webgpu/index.js'),cpu=await import(dest+'/cpu/index.js');assert.equal(typeof gpu.createESM3Engine,'function');assert.equal(typeof cpu.createESM3CPU,'function');assert.deepEqual((await readdir(dest)).sort(),['LICENSE.md','README_WEBGPU.md','cpu','package.json','shared','webgpu'].sort());await rm(dest,{recursive:true,force:true});
});
