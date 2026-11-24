import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { iterateWeights } from '../shared/weights.js';
import { RFD3CPUModel } from '../cpu/model.js';
import { metrics } from '../shared/metrics.js';
test('complete CPU token initializer matches real checkpoint FP32 oracle', async () => {
  const architecture=JSON.parse(await readFile(new URL('../fixtures/architecture.json',import.meta.url)));
  const base=new URL('../weights/fp32/',import.meta.url);
  const manifest=JSON.parse(await readFile(new URL('manifest.json',base)));
  const weights=new Map(),names=Object.keys(manifest.tensors).filter(n=>n.startsWith('token_initializer.'));
  for await(const t of iterateWeights(manifest,base,{names,cache:'none',fetcher:async url=>new Response(await readFile(new URL(url)))}))
    weights.set(t.name,{...t,bytes:t.bytes.slice(),scales:t.scales?.slice()});
  const model=new RFD3CPUModel(weights,architecture.config),results=[];
  for(const name of ['monomer','cyclic','atomic-motif']) {
    const fixture=JSON.parse(await readFile(new URL('../fixtures/oracle/'+name+'.json',import.meta.url)));
    const features=Object.fromEntries(Object.entries(fixture.features).map(([k,v])=>[k,new Float32Array(v.data)]));
    const start=performance.now(),actual=model.initialize(features),elapsed=performance.now()-start;
    for(const [key,v]of Object.entries(actual)) {
      const stats=metrics(v.data,fixture.initializer[key].data);
      results.push({case:name,tensor:key,...stats,cpuInitializerMs:elapsed});
      assert.ok(stats.maxAbsDiff<0.002,JSON.stringify(results.at(-1)));
    }
  }
  await writeFile(new URL('../fixtures/cpu-initializer-parity.json',import.meta.url),JSON.stringify(results,null,2)+'\n');
  console.log(JSON.stringify({initializer:results}));
});
