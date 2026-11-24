import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, mkdir, writeFile, rm, symlink } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { syncUpstream } from '../sync-upstream.mjs';
test('staged vendor includes all runtimes and replaces previous generation as a directory', async () => {
  const temp = await mkdtemp(resolve('.cache/vendor-test-'));
  try {
    const target = join(temp, 'vendor');
    await mkdir(target); await writeFile(join(target, 'obsolete'), 'old generation');
    const result = await syncUpstream(target, { write: true });
    assert.ok(result.files.includes('webgpu/linear.js'));
    assert.ok(result.files.includes('cpu/ops.js'));
    assert.ok(result.files.includes('fixtures/architecture.json'));
    assert.ok(!result.files.some(f => f.startsWith('demo/') || f === 'tools/serve.mjs' || f === 'LICENSE-PYTORCH'));
    assert.equal(await readFile(join(target, 'NOTICE'), 'utf8'), await readFile('NOTICE', 'utf8'));
    await assert.rejects(readFile(join(target, 'obsolete')), { code: 'ENOENT' });
    await syncUpstream(target, { write: true });
    const link = join(temp, 'link'); await symlink(target, link);
    await assert.rejects(syncUpstream(link, { write: true }), /real directory/);
    await assert.rejects(syncUpstream(resolve('shared'), { write: true }), /overlaps/);
  } finally { await rm(temp, { recursive: true, force: true }); }
});
