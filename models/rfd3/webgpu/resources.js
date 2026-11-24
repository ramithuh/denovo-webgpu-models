const software = /swiftshader|llvmpipe|lavapipe|softpipe|software|\bwarp\b|basic render driver/i;
const negotiatedLimits = ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize',
  'maxStorageBuffersPerShaderStage', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ', 'maxComputeWorkgroupsPerDimension'];
const flag = value => value === true || value === 1 || value === 'true';
const adapterText = info => [info?.vendor, info?.device, info?.architecture, info?.description].join(' ').trim();
function checkAdapter(info, fallback, allowed) {
  if (!allowed && (fallback || software.test(adapterText(info))))
    throw new Error('A hardware WebGPU adapter is required');
}
export function formatGpuLossDetail(loss) {
  if (!loss || typeof loss !== 'object') return '';
  const parts = [];
  if (loss.reason) parts.push('reason=' + String(loss.reason));
  if (loss.message) parts.push(String(loss.message).trim());
  return parts.length ? ' (' + parts.join(': ') + ')' : '';
}

export async function createGpuResources({ device: supplied, allowSoftwareFallback = false, signal,
  requiredLimits = {} } = {}) {
  signal?.throwIfAborted();
  let device = supplied;
  let info = device?.adapterInfo;
  let fallback = info?.isFallbackAdapter == null ? undefined : flag(info.isFallbackAdapter);
  if (!device) {
    const gpu = globalThis.navigator?.gpu;
    if (!gpu) throw new Error('WebGPU unavailable');
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' }) ?? await gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    info = adapter.info;
    fallback = info?.isFallbackAdapter == null && adapter.isFallbackAdapter == null ? undefined :
      flag(info?.isFallbackAdapter) || flag(adapter.isFallbackAdapter);
    signal?.throwIfAborted();
    checkAdapter(info, fallback, allowSoftwareFallback);
    const limits = Object.fromEntries(negotiatedLimits.filter(k => adapter.limits[k] != null)
      .map(k => [k, adapter.limits[k]]));
    Object.assign(limits, requiredLimits);
    device = await adapter.requestDevice({ requiredLimits: limits });
    if (signal?.aborted) { device.destroy(); signal.throwIfAborted(); }
  } else {
    checkAdapter(info, fallback, allowSoftwareFallback);
    for (const [key, value] of Object.entries(requiredLimits)) {
      const isMinimum = key.startsWith('min');
      if (!(key in device.limits) || (isMinimum ? device.limits[key] > value : device.limits[key] < value))
        throw new RangeError('Supplied device does not meet required limit: ' + key);
    }
  }
  const buffers = new Set();
  let resident = 0, peak = 0, closed = false, loss;
  function release(buffer) {
    if (!buffers.delete(buffer)) return;
    resident -= buffer.size;
    buffer.destroy();
  }
  function invalidate(reason) {
    closed = true;
    loss = reason;
    for (const buffer of buffers) buffer.destroy();
    buffers.clear();
    resident = 0;
  }
  const lost = device.lost?.then(reason => { invalidate(reason); return reason; });
  return {
    device, adapterInfo: info, lost,
    isFallbackAdapter: Boolean(fallback || software.test(adapterText(info))),
    get closed() { return closed; },
    get loss() { return loss; },
    memoryUsage: () => resident,
    peakMemoryUsage: () => peak,
    resetPeakMemoryUsage: () => { peak = resident; },
    allocate(size, usage, label = '') {
      if (closed) throw new Error('GPU resources destroyed or device lost' + formatGpuLossDetail(loss));
      if (!Number.isSafeInteger(size) || size <= 0) throw new RangeError('Invalid buffer size');
      if (!Number.isInteger(usage) || usage <= 0 || usage > 0x3ff) throw new RangeError('Invalid buffer usage');
      const aligned = 4 * Math.ceil(size / 4);
      // A backing buffer may exceed the binding limit; bind subranges explicitly.
      if (!Number.isSafeInteger(aligned) || aligned > device.limits.maxBufferSize)
        throw new RangeError('Buffer exceeds device backing-buffer limit');
      const buffer = device.createBuffer({ label, size: aligned, usage });
      buffers.add(buffer); resident += buffer.size; peak = Math.max(peak, resident);
      return buffer;
    },
    storageBinding(buffer, { offset = 0, size = buffer.size - offset } = {}) {
      if (closed) throw new Error('GPU resources destroyed or device lost' + formatGpuLossDetail(loss));
      if (!buffers.has(buffer)) throw new Error('Buffer is not live in this owner');
      if (!(buffer.usage & 128)) throw new Error('Buffer lacks STORAGE usage');
      if (!Number.isSafeInteger(offset) || offset < 0 || offset % device.limits.minStorageBufferOffsetAlignment ||
          !Number.isSafeInteger(size) || size <= 0 || size % 4 || offset + size > buffer.size ||
          size > device.limits.maxStorageBufferBindingSize) throw new RangeError('Invalid storage binding range');
      return { buffer, offset, size };
    },
    release,
    destroy() {
      if (closed) return;
      invalidate();
      if (!supplied) device.destroy();
    },
  };
}
