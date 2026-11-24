import { createGpuResources } from './resources.js';
const code = `
struct Dims { m: u32, n: u32, k: u32, dtype: u32, bias: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> d: Dims;
var<workgroup> a: array<f32, 128>;
var<workgroup> b: array<f32, 128>;
fn weight(index: u32, row: u32) -> f32 {
  if (d.dtype == 0u) { return bitcast<f32>(w[index]); }
  if (d.dtype == 1u) { let v = unpack2x16float(w[index / 2u]); return v[index % 2u]; }
  let byte = (w[index / 4u] >> (8u * (index % 4u))) & 255u;
  return f32(bitcast<i32>(byte << 24u) >> 24) * scales[row];
}
@compute @workgroup_size(8, 8) fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.y * 8u + local.y;
  let col = group.x * 8u + local.x;
  var sum = 0.0;
  for (var base = 0u; base < d.k; base += 16u) {
    for (var j = 0u; j < 2u; j++) {
      let ka = local.x + 8u * j;
      let kb = local.y + 8u * j;
      var av = 0.0; var bv = 0.0;
      if (row < d.m && base + ka < d.k) { av = x[row * d.k + base + ka]; }
      if (col < d.n && base + kb < d.k) { bv = weight(col * d.k + base + kb, col); }
      a[local.y * 16u + ka] = av;
      b[local.x * 16u + kb] = bv;
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) { sum += a[local.y * 16u + k] * b[local.x * 16u + k]; }
    workgroupBarrier();
  }
  if (row < d.m && col < d.n) {
    if (d.bias != 0u) { sum += bias[col]; }
    out[row * d.n + col] = sum;
  }
}`;

export async function createPackedLinearRuntime(options = {}) {
  const arena = options.resources ?? await createGpuResources(options);
  const { device } = arena, weights = new Set();
  let destroyed = false;
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code, label: 'RFD3 packed linear' });
  let pipeline;
  try {
    pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  } catch (error) {
    await device.popErrorScope(); if (!options.resources) arena.destroy(); throw error;
  }
  const error = await device.popErrorScope();
  if (error) { if (!options.resources) arena.destroy(); throw new Error(error.message); }
  function upload(bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    const buffer = arena.allocate(bytes.byteLength || 4, usage);
    // The final word is padded only on upload. Narrow weights never expand to FP32.
    const padded = new Uint8Array(buffer.size); padded.set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    device.queue.writeBuffer(buffer, 0, padded); return buffer;
  }
  const zero = upload(new Float32Array(1));
  return {
    resources: arena,
    uploadWeight(tensor) {
      if (destroyed || arena.closed) throw new Error('Runtime is closed');
      if (!['F32', 'F16', 'I8'].includes(tensor.dtype) || tensor.shape.length !== 2 ||
          !tensor.shape.every(n => Number.isSafeInteger(n) && n > 0)) throw new Error('Invalid linear weight');
      const expected = tensor.shape[0] * tensor.shape[1] * ({ F32: 4, F16: 2, I8: 1 }[tensor.dtype]);
      if (tensor.bytes.byteLength !== expected || tensor.dtype === 'I8' && tensor.scales?.byteLength !== tensor.shape[0] * 4)
        throw new Error('Invalid packed weight bytes');
      let buffer, scales;
      try {
        buffer = upload(tensor.bytes);
        scales = tensor.scales ? upload(tensor.scales) : zero;
        const weight = { shape: [...tensor.shape], dtype: { F32: 0, F16: 1, I8: 2 }[tensor.dtype], buffer, scales };
        weights.add(weight); return weight;
      } catch (error) { if (buffer) arena.release(buffer); if (scales && scales !== zero) arena.release(scales); throw error; }
    },
    releaseWeight(weight) {
      if (!weights.delete(weight)) return;
      arena.release(weight.buffer); if (weight.scales !== zero) arena.release(weight.scales);
    },
    async linear(input, weight, bias, { signal } = {}) {
      signal?.throwIfAborted();
      if (destroyed || !weights.has(weight)) throw new Error('Weight is not live in this runtime');
      const [n, k] = weight.shape, m = input.length / k;
      if (!(input instanceof Float32Array) || !Number.isSafeInteger(m) || m < 1 || bias && bias.length !== n)
        throw new Error('Invalid linear input');
      const local = [];
      const keep = b => { local.push(b); return b; };
      device.pushErrorScope('validation');
      let popped = false;
      try {
        const x = keep(upload(input)), biasBuffer = bias ? keep(upload(bias)) : zero;
        const output = keep(arena.allocate(m * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC));
        const readback = keep(arena.allocate(m * n * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ));
        const uniform = keep(upload(new Uint32Array([m, n, k, weight.dtype, bias ? 1 : 0, 0, 0, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
        const entries = [x, weight.buffer, weight.scales, biasBuffer, output, uniform].map((buffer, binding) =>
          ({ binding, resource: binding === 5 ? { buffer } : arena.storageBinding(buffer) }));
        const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(m / 8)); pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, m * n * 4);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        signal?.throwIfAborted();
        const result = new Float32Array(readback.getMappedRange().slice(0)); readback.unmap();
        const validation = await device.popErrorScope(); popped = true;
        if (validation) throw new Error(validation.message);
        return result;
      } finally {
        if (!popped) await device.popErrorScope();
        for (const buffer of local) arena.release(buffer);
      }
    },
    destroy() {
      if (destroyed) return; destroyed = true;
      for (const weight of [...weights]) this.releaseWeight(weight);
      arena.release(zero); if (!options.resources) arena.destroy();
    },
  };
}
