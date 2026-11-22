// Compare scores from the classifier's packed projection against the separate
// query/key inputs used by the backbone. V contains sentinels to catch bad strides.
import { scoresSource } from '../webgpu/dyna1/shaders.js';

export async function verifyPackedAttention(device) {
  const rows = 5, heads = 6, width = heads * 256;
  const q = new Float32Array(rows * width);
  const k = new Float32Array(rows * width);
  const packed = new Float32Array(3 * rows * width).fill(101);
  for (let i = 0; i < q.length; i++) {
    q[i] = ((i * 7) % 53 - 26) / 16;
    k[i] = ((i * 11) % 71 - 35) / 16;
  }
  for (let row = 0; row < rows; row++) {
    const start = row * width;
    packed.set(q.subarray(start, start + width), 3 * start);
    packed.set(k.subarray(start, start + width), 3 * start + width);
  }
  const buffers = [];
  function allocate(size, usage) {
    const buffer = device.createBuffer({ size, usage });
    buffers.push(buffer);
    return buffer;
  }
  function upload(data) {
    const buffer = allocate(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }
  const count = heads * rows * rows, bytes = count * 4;
  try {
    const encoder = device.createCommandEncoder();
    const readback = allocate(2 * bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const qkv = upload(packed);
    for (const packedQkv of [false, true]) {
      const module = device.createShaderModule({ code: scoresSource(rows, width, heads, { packedQkv }) });
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const out = allocate(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const inputs = packedQkv ? [qkv, qkv] : [upload(q), upload(k)];
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [...inputs, out].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(count / 64));
      pass.end();
      encoder.copyBufferToBuffer(out, 0, readback, packedQkv ? bytes : 0, bytes);
    }
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.getMappedRange());
    for (let i = 0; i < count; i++) {
      const head = Math.floor(i / (rows * rows)), row = Math.floor(i / rows) % rows, column = i % rows;
      let expected = 0;
      for (let j = 0; j < 256; j++) {
        expected += q[row * width + head * 256 + j] * k[column * width + head * 256 + j];
      }
      if (result[i] !== expected / 16 || result[count + i] !== result[i]) {
        throw new Error('Packed attention score mismatch at ' + i);
      }
    }
    readback.unmap();
    return count;
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}
