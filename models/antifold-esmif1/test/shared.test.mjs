import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,readdir} from 'node:fs/promises';
import {parseStructure,prepareInput,combineChains,regionMask,tokenize,parseSafetensors,loadWeights,sha256} from '../shared/index.js';
const read=async p=>{const b=await readFile(p);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
test('PDB/mmCIF parsing exactly matches upstream: models, altLoc, gaps, insertion, modified residues and ignored context',async()=>{
 const meta=JSON.parse(await readFile('fixtures/parser_oracle.json','utf8')),bytes=await read('fixtures/parser_oracle.bin');
 for(const ext of ['pdb','cif']){
  const parsed=parseStructure(await readFile(`fixtures/parser.${ext}`,'utf8'));
  assert.deepEqual(parsed.chains.map(c=>c.id),Object.keys(meta.cases[ext]));
  for(const c of parsed.chains){const expected=meta.cases[ext][c.id],t=meta.tensors[ext+'.'+c.id],coords=new Float32Array(bytes,t.offset,t.length);assert.equal(c.sequence,expected.sequence);assert.deepEqual(c.coords,coords);assert.deepEqual(c.residues.map(r=>r.number),expected.positions);assert.deepEqual(c.residues.map(r=>`${r.number}${r.insertion}${r.chain}`),expected.ids);}
  const combined=combineChains(parsed,{targetChain:parsed.chains[1].id});assert.equal(combined.residues[0].chain,parsed.chains[1].id);assert.equal(combined.residues.filter(x=>x===null).length,10);
 }
});
test('Explicit IMGT/Kabat masks preserve chain roles and insertion variants',()=>{
 const residues=[{chain:'H',number:109,insertion:''},{chain:'H',number:109,insertion:'A'},{chain:'L',number:109,insertion:''},{chain:'H',number:-2,insertion:''}];
 assert.deepEqual([...regionMask(residues,{heavyChain:'H',lightChain:'L',regions:['H:109']})],[1,1,0,0]);
 assert.deepEqual([...regionMask(residues,{heavyChain:'H',lightChain:'L',regions:['H:109A']})],[0,1,0,0]);
 assert.deepEqual([...regionMask([{chain:'HH',number:100,insertion:'A'},{chain:'LL',number:100,insertion:''}],{numbering:'kabat',heavyChain:'HH',lightChain:'LL',regions:['CDR3']})],[1,0]);
 assert.throws(()=>regionMask(residues,{regions:['CDR3']}),/heavy chain/);assert.throws(()=>regionMask(residues,{heavyChain:'H',regions:['H:10-2']}),/Reversed/);
});
test('Invalid input and corrupt/incomplete bundles fail',async()=>{
 assert.throws(()=>tokenize('A?'));assert.throws(()=>parseStructure(''));assert.throws(()=>prepareInput({coords:[null]}));assert.throws(()=>parseSafetensors(new ArrayBuffer(2)));
 await assert.rejects(()=>loadWeights('missing',{read}));
 const manifest=await read('weights/antifold/fp32/manifest.json');
 await assert.rejects(()=>loadWeights('weights/antifold/fp32',{read:async p=>p.endsWith('manifest.json')?manifest:new ArrayBuffer(4)}),/byte length/);
});
test('All six bundles have checked shards under the strict cap and colocated INT8 scales',async()=>{
 for(const v of ['antifold','esmif1'])for(const p of ['fp32','fp16','int8']){
  const base=`weights/${v}/${p}`,m=JSON.parse(await readFile(`${base}/manifest.json`,'utf8'));assert.equal(m.total_params,141662150);let total=0;
  for(const s of m.shards){const b=await read(`${base}/${s.file}`);assert.equal(b.byteLength,s.byteLength);assert.ok(b.byteLength<=25000000);assert.equal(await sha256(b),s.sha256);const tensors=parseSafetensors(b);for(const t of Object.values(m.tensors))for(const part of t.parts??[t])if(part.shard===s.file&&part.dtype==='I8')assert.ok(tensors[part.quant.scale_tensor]);total+=b.byteLength;}
  assert.equal(total,m.total_bytes);
 }
});
test('Runtime modules contain no external imports or Node APIs',async()=>{
 for(const dir of ['shared','cpu','webgpu'])for(const f of await readdir(dir)){if(!f.endsWith('.js'))continue;const code=await readFile(`${dir}/${f}`,'utf8');assert.doesNotMatch(code,/from\s*['"](?:node:|[^.'"])/);assert.doesNotMatch(code,/\brequire\s*\(|\bprocess\./);}
});
test('Known modified peptide residues reject consistently; unknown molecular context is ignored',async()=>{
 const source=await readFile('fixtures/parser.pdb','utf8');assert.throws(()=>parseStructure(source.replaceAll('MSE','SEP')),/Unsupported residue SEP/);
 const parsed=parseStructure(source.replaceAll('MSE','LIG'));assert.equal(parsed.chains[0].sequence,'CX');
});
test('Every committed oracle has matching byte length and SHA-256',async()=>{
 for(const file of await readdir('fixtures'))if(file.endsWith('.json')&&(file.startsWith('oracle_')||file==='parser_oracle.json')){const meta=JSON.parse(await readFile(`fixtures/${file}`,'utf8')),bytes=await read(`fixtures/${file.replace('.json','.bin')}`);assert.equal(bytes.byteLength,meta.byteLength);assert.equal(await sha256(bytes),meta.sha256);}
});
test('Loader rejects a same-length corrupted shard and overlapping multipart rows',async()=>{
 const base='weights/antifold/fp32';
 await assert.rejects(()=>loadWeights(base,{read:async path=>{const data=await read(path);if(path.endsWith('.safetensors'))new Uint8Array(data)[data.byteLength-1]^=1;return data;}}),/SHA-256/);
 const manifest=JSON.parse(await readFile(`${base}/manifest.json`,'utf8'));Object.values(manifest.tensors)[0].parts[0].row_start=1;
 await assert.rejects(()=>loadWeights(base,{read:async()=>new TextEncoder().encode(JSON.stringify(manifest)).buffer}),/multipart/);
});
test('Combined chain metadata is accepted as raw input and heavy chain ordering is explicit',async()=>{
 const parsed=parseStructure(await readFile('fixtures/parser.pdb','utf8')),input=combineChains(parsed,{chains:['B','A'],heavyChain:'A',lightChain:'B'});
 assert.equal(input.residues[0].chain,'A');assert.equal(prepareInput({...input,chains:['A','B']}).L,input.sequence.length);
});
