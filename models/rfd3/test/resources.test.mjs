import test from 'node:test';
import assert from 'node:assert/strict';
import { createGpuResources } from '../webgpu/resources.js';
function mock(info = { vendor: 'nvidia', isFallbackAdapter: false }) {
  const lost = Promise.withResolvers();
  const device = { adapterInfo: info, lost: lost.promise,
    limits: { maxBufferSize: 1024, maxStorageBufferBindingSize: 128, minStorageBufferOffsetAlignment: 16 },
    createBuffer: o => ({ ...o, destroyed: false, destroy() { this.destroyed = true; } }),
    destroy() { this.destroyed = true; } };
  return { device, lose: lost.resolve };
}
test('device loss invalidates owner and clears accounting, preserving peak', async () => {
  const { device, lose } = mock();
  const owner = await createGpuResources({ device });
  const b = owner.allocate(256, 128);
  assert.equal(owner.memoryUsage(), 256);
  assert.equal(owner.storageBinding(b, { size: 128 }).size, 128);
  assert.throws(() => owner.storageBinding(b), /binding/);
  assert.throws(() => owner.allocate(4, 0), /usage/);
  assert.throws(() => owner.allocate(4, 2 ** 32 + 128), /usage/);
  lose({ reason: 'unknown', message: 'test loss' });
  await owner.lost;
  assert.equal(owner.memoryUsage(), 0);
  assert.equal(owner.peakMemoryUsage(), 256);
  assert.equal(b.destroyed, true);
  assert.throws(() => owner.allocate(4, 128), /lost/);
  owner.release(b); owner.destroy();
  assert.equal(device.destroyed, undefined);
});
test('software detection handles descriptions and normalized fallback flags', async () => {
  for (const info of [{ description: 'Mesa softpipe' }, { description: 'llvmpipe' },
    { description: 'SwiftShader' }, { vendor: 'nvidia', isFallbackAdapter: 'true' }]) {
    await assert.rejects(createGpuResources({ device: mock(info).device }), /hardware/);
  }
  const owner = await createGpuResources({ device: mock({ vendor: 'nvidia', isFallbackAdapter: 'false' }).device });
  assert.equal(owner.isFallbackAdapter, false);
  owner.destroy();
  // Safari/Firefox privacy-redacted adapterInfo with undefined isFallbackAdapter is accepted
  const redacted = await createGpuResources({ device: mock({ vendor: '', device: '', architecture: '', description: '' }).device });
  assert.equal(redacted.isFallbackAdapter, false);
  redacted.destroy();
  // Known software renderers are still allowed only when allowSoftwareFallback is true
  const sw = await createGpuResources({ device: mock({ description: 'SwiftShader' }).device, allowSoftwareFallback: true });
  assert.equal(sw.isFallbackAdapter, true);
  sw.destroy();
});
test('device acquisition requests full adapter buffer limits', async () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const { device } = mock();
  let requested;
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: {
    requestAdapter: async () => ({ info: { vendor: 'nvidia' }, isFallbackAdapter: false,
      limits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 30 },
      requestDevice: async options => { requested = options; return device; } }),
  } } });
  try {
    const owner = await createGpuResources();
    assert.equal(requested.requiredLimits.maxBufferSize, 2 ** 31);
    assert.equal(requested.requiredLimits.maxStorageBufferBindingSize, 2 ** 30);
    owner.destroy(); assert.equal(device.destroyed, true);
  } finally {
    if (previous) Object.defineProperty(globalThis, 'navigator', previous);
    else delete globalThis.navigator;
  }
});
