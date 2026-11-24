import { createGpuResources } from "../webgpu/resources.js";
import { readShard } from "../shared/shard.js";

async function run() {
  const expected = "browser storage integrity test\n";
  const data = new TextEncoder().encode(expected);
  const hash = await crypto.subtle.digest("SHA-256", data);
  const metadata = { byteLength: data.length,
    sha256: Array.from(new Uint8Array(hash), x => x.toString(16).padStart(2, "0")).join("") };
  let offlineReads = 0;
  function offline() {
    offlineReads++;
    throw new Error("Fetch attempted offline");
  }
  for (const cache of ["cache-storage", "opfs"]) {
    await readShard("/fixture", metadata, { cache });
    const hit = await readShard("/fixture", metadata, { cache, fetcher: offline });
    if (new TextDecoder().decode(hit) !== expected) throw new Error("Offline content mismatch");
  }
  const denied = () => { throw new DOMException("Private storage denied", "SecurityError"); };
  const bytes = await readShard("/fixture", metadata, { openStorage: denied });
  if (new TextDecoder().decode(bytes) !== expected) throw new Error("Storage fallback failed");
  const abort = new AbortController();
  abort.abort();
  try {
    await readShard("/fixture", metadata, { signal: abort.signal, fetcher: offline });
    throw new Error("Cancellation was ignored");
  } catch (error) {
    if (error.name !== "AbortError") throw error;
  }

  const active = new AbortController();
  try {
    await readShard("/partial", metadata, {
      cache: "none", signal: active.signal,
      fetcher: async (url, options) => {
        const response = await fetch(url, options);
        setTimeout(() => active.abort(), 0);
        return response;
      },
    });
    throw new Error("Active fetch cancellation was ignored");
  } catch (error) {
    if (error.name !== "AbortError") throw error;
  }

  const arena = await createGpuResources({ allowSoftwareFallback: false });
  const { device, adapterInfo } = arena;
  let compute;
  try {
    device.pushErrorScope("validation");
    const output = arena.allocate(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = arena.allocate(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const module = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage, read_write> result: array<u32>;
      @compute @workgroup_size(4) fn main(@builtin(local_invocation_index) i: u32) {
        result[i] = i * 3u + 1u;
      }` });
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto", compute: { module, entryPoint: "main" },
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: output } }] }));
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, 16);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    compute = [...new Uint32Array(readback.getMappedRange())];
    readback.unmap();
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
    arena.release(readback);
    if (arena.memoryUsage() !== 16) throw new Error("Incorrect live buffer count");
  } finally {
    arena.destroy();
    arena.destroy();
  }
  return {
    gpu: { vendor: adapterInfo.vendor, architecture: adapterInfo.architecture,
      description: adapterInfo.description, fallback: arena.isFallbackAdapter },
    compute, peak: arena.peakMemoryUsage(), resident: arena.memoryUsage(), offlineReads,
  };
}

run().then(value => postMessage(value), error => postMessage({ error: error.stack ?? error.message }));
