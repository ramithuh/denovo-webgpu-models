import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { parseSafetensors } from '../shared/shard.js';
import { validateManifest, iterateWeights, decodeWeight } from '../shared/weights.js';
import { linear } from '../cpu/ops.js';
import { metrics } from '../shared/metrics.js';
const root = new URL('../', import.meta.url);
const fileFetcher = async url => new Response(await readFile(new URL(url)));

for (const precision of ['fp32', 'fp16', 'int8']) test('full shard verification and schema: ' + precision, async () => {
  const base = new URL('weights/' + precision + '/', root);
  const manifest = validateManifest(JSON.parse(await readFile(new URL('manifest.json', base))));
  let bytes = 0, tensorCount = 0;
  for (const shard of manifest.shards) {
    const b = await readFile(new URL(shard.file, base));
    assert.equal(b.length, shard.byteLength);
    assert.ok(b.length < 25_000_000);
    assert.equal(createHash('sha256').update(b).digest('hex'), shard.sha256);
    const tensors = parseSafetensors(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    for (const [name, entry] of Object.entries(manifest.tensors)) {
      if (entry.shard !== shard.file) continue;
      const t = tensors.get(name);
      assert.equal(t.dtype, entry.dtype); assert.deepEqual(t.shape, entry.shape);
      if (entry.role !== 'linear') assert.equal(t.dtype, 'F32');
      if (entry.scale_key) {
        const s = tensors.get(entry.scale_key);
        assert.equal(s.dtype, 'F32'); assert.deepEqual(s.shape, [t.shape[0]]);
        for (const scale of decodeWeight(s)) assert.ok(Number.isFinite(scale) && scale > 0);
      }
      tensorCount++;
    }
    bytes += b.length;
  }
  assert.equal(bytes, manifest.total_bytes);
  assert.equal(tensorCount, Object.keys(manifest.tensors).length);
});

test('real checkpoint linear CPU FP32 oracle parity', async () => {
  const oracle = JSON.parse(await readFile(new URL('fixtures/oracle/manifest.json', root)));
  const base = new URL('weights/fp32/', root);
  const manifest = JSON.parse(await readFile(new URL('manifest.json', base)));
  const results = [];
  for (const filename of oracle.linear_cases) {
    const f = JSON.parse(await readFile(new URL('fixtures/oracle/' + filename, root)));
    const key = f.name + '.weight';
    let weight;
    for await (const w of iterateWeights(manifest, base, { names: [key], cache: 'none', fetcher: fileFetcher })) weight = w;
    const y = linear(new Float32Array(f.input.data), weight, f.bias ? new Float32Array(f.bias.data) : undefined);
    const result = metrics(y, f.output.data);
    assert.ok(result.maxAbsDiff < 1e-4, JSON.stringify({ layer: f.name, ...result }));
    results.push({ layer: f.name, ...result });
  }
  console.log(JSON.stringify({ cpu_linear: results }));
});

test('P2-6: iterateWeights normalizes baseURL without trailing slash, calls onProgress, and supports custom read(url)', async () => {
  const baseWithSlash = new URL('weights/fp16/', root).href;
  const baseNoSlash = baseWithSlash.replace(/\/$/, '');
  const manifest = JSON.parse(await readFile(new URL('weights/fp16/manifest.json', root)));
  const firstName = Object.keys(manifest.tensors)[0];
  const progress = [];
  const urls = [];
  let loadedTensor;
  for await (const w of iterateWeights(manifest, baseNoSlash, {
    names: [firstName],
    cache: 'none',
    read: async url => {
      urls.push(url);
      const buf = await readFile(new URL(url));
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    },
    onProgress: ev => progress.push(ev),
  })) loadedTensor = w;
  assert.ok(loadedTensor && loadedTensor.name === firstName);
  assert.equal(urls.length, 1);
  assert.ok(urls[0].includes('/weights/fp16/'), `expected /weights/fp16/ in ${urls[0]}`);
  assert.equal(progress.length, 2);
  assert.equal(progress[0].phase, 'weights');
  assert.equal(progress[0].loaded, 0);
  assert.equal(progress[1].loaded, progress[1].total);
  assert.ok(progress[1].total > 0);
});
