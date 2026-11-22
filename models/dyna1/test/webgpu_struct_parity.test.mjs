import test from 'node:test';
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { runBrowser } from './browser.mjs';

test('Real WebGPU structure encoder + seq+struct + struct-only parity across precisions', async () => {
  const combined = { ok: true, modes: {}, validationErrors: [] };
  for (const [precision, native] of [['fp32', 0], ['fp16', 0], ['fp16', 1], ['int8', 0]]) {
    const r = await runBrowser(`/test/gpu-struct-harness.js?precision=${precision}&native=${native}`);
    assert.ok(r.ok, r.error);
    combined.adapterInfo = r.adapterInfo;
    combined.hasF16 = r.hasF16;
    combined.modes[r.mode] = r.metrics;
    combined.validationErrors.push(...r.validationErrors);
  }
  await writeFile(new URL('../fixtures/gpu_struct_metrics.json', import.meta.url), JSON.stringify(combined, null, 2) + '\n');
  console.log(JSON.stringify(combined.adapterInfo));
});
