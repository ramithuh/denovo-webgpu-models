import { loadWeights } from "../shared/weights.js";
import { network, validateInput } from "../shared/network.js";
import { sampleWith } from "../shared/sampling.js";
import { acquire } from "./device.js";
import { GPUOps } from "./ops.js";
export { rejectSoftware } from "./device.js";
export async function createProteinaEngine(options = {}) {
  const {
    baseUrl,
    precision = "fp32",
    useF16 = false,
    signal,
    onProgress,
  } = options;
  if (
    typeof baseUrl !== "string" ||
    !["fp32", "fp16", "int8"].includes(precision)
  )
    throw TypeError("baseUrl and a supported precision are required");
  const acquired = await acquire(options),
    { device, owned } = acquired;
  const allocation = new Map(),
    pool = new Map(),
    weights = {},
    constants = {},
    pipelines = new Map();
  let destroyed = false,
    busy = false,
    lost = null;
  device.lost.then((info) => {
    lost = info.message || "WebGPU device lost";
  });
  const e = {
    device,
    weights,
    constants,
    pipelines,
    useF16,
    hasUnvalidatedPipelines: true,
    uniformCapacity: 0,
    uniformData: null,
    uniformU32: null,
    uniformF32: null,
    uniformBuffer: null,
    ensureUniformCapacity(slots) {
      if (slots <= e.uniformCapacity && e.uniformBuffer) return;
      const next = Math.max(512, 1 << Math.ceil(Math.log2(slots)));
      if (e.uniformBuffer) e.free(e.uniformBuffer);
      e.uniformData = new ArrayBuffer(next * 256);
      e.uniformU32 = new Uint32Array(e.uniformData);
      e.uniformF32 = new Float32Array(e.uniformData);
      e.uniformBuffer = e.allocate(
        next * 256,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        "scratch",
      );
      e.uniformCapacity = next;
    },
    allocate(bytes, usage, kind) {
      const size = Math.max(4, Math.ceil(bytes / 4) * 4);
      if (
        !Number.isSafeInteger(size) ||
        size > device.limits.maxBufferSize ||
        (usage & GPUBufferUsage.STORAGE &&
          size > device.limits.maxStorageBufferBindingSize)
      )
        throw RangeError(`Buffer size ${size} exceeds WebGPU limits`);
      const b = device.createBuffer({ size, usage, label: `proteina ${kind}` });
      allocation.set(b, { size, kind });
      return b;
    },
    free(b) {
      if (allocation.has(b)) {
        b.destroy();
        allocation.delete(b);
      }
    },
    scratch(bytes) {
      const size = Math.max(4, Math.ceil(bytes / 4) * 4),
        list = pool.get(size);
      if (list?.length) return list.pop();
      return e.allocate(
        size,
        GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
        "scratch",
      );
    },
    recycle(b) {
      const size = allocation.get(b)?.size;
      if (!size) return;
      if (!pool.has(size)) pool.set(size, []);
      pool.get(size).push(b);
    },
  };
  const upload = (data, kind = "weights") => {
    const b = e.allocate(
      data.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      kind,
    );
    if (data.byteLength % 4) {
      const p = new Uint8Array(Math.ceil(data.byteLength / 4) * 4);
      p.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      device.queue.writeBuffer(b, 0, p);
    } else device.queue.writeBuffer(b, 0, data);
    return b;
  };
  e.trim = () => {
    let size = 0;
    for (const list of pool.values())
      for (const b of list) size += allocation.get(b).size;
    for (const list of pool.values())
      while (list.length && size > 128 * 1024 * 1024) {
        const b = list.pop();
        size -= allocation.get(b).size;
        e.free(b);
      }
  };
  const destroy = () => {
    if (destroyed) return;
    destroyed = true;
    for (const b of allocation.keys()) e.free(b);
    e.uniformBuffer = null;
    e.uniformCapacity = 0;
    pool.clear();
    pipelines.clear();
    for (const n of Object.keys(weights)) delete weights[n];
    if (owned) device.destroy();
  };
  const memoryUsage = () => {
    const stats = {
      weightBytes: 0,
      scratchBytes: 0,
      transientBytes: 0,
      transientBuffers: 0,
      totalBytes: 0,
      buffers: allocation.size,
    };
    for (const { size, kind } of allocation.values()) {
      stats.totalBytes += size;
      if (kind === "weights") stats.weightBytes += size;
      else if (kind === "scratch") stats.scratchBytes += size;
      else {
        stats.transientBytes += size;
        stats.transientBuffers++;
      }
    }
    return stats;
  };
  let manifest;
  try {
    const result = await loadWeights(
      `${baseUrl.replace(/\/$/, "")}/${precision}`,
      {
        ...options,
        expectedPrecision: precision,
        onTensor: (name, t) => {
          signal?.throwIfAborted();
          const buffer = upload(t.bytes);
          weights[name] = {
            buffer,
            shape: t.shape,
            dtype: t.dtype,
            bytes: t.bytes.byteLength,
          };
          if (t.scales) weights[name].scale = upload(t.scales);
        },
      },
    );
    manifest = result.manifest;
    constants.timeFreq = upload(Float32Array.from(manifest.constants.timeFreq));
    constants.indexDivisor = upload(
      Float32Array.from(manifest.constants.indexDivisor),
    );
    constants.bins = upload(
      Float32Array.from([
        ...manifest.constants.scBins,
        ...manifest.constants.xtBins,
      ]),
    );
    e.zero = upload(new Float32Array(4));
    signal?.throwIfAborted();
  } catch (error) {
    destroy();
    throw error;
  }
  async function execute(coordinates, options = {}) {
    if (destroyed) throw Error("Engine destroyed");
    if (lost) throw Error(lost);
    const input = validateInput(coordinates, options),
      N = input.length + 10;
    if (
      N * N * 512 * 4 >
      Math.min(
        device.limits.maxBufferSize,
        device.limits.maxStorageBufferBindingSize,
      )
    )
      throw RangeError("Length exceeds pair buffer device limit");
    const started = performance.now();
    let ops;
    const checkValidation = e.hasUnvalidatedPipelines || Boolean(options.trace);
    if (checkValidation) device.pushErrorScope("validation");
    try {
      ops = new GPUOps(e, input);
      ops.init();
      const y = network(ops, input, {
        trace: options.trace ? (n, t) => ops.trace(n, t) : undefined,
      });
      const result = await ops.finish(y, options.signal);
      if (checkValidation) e.hasUnvalidatedPipelines = false;
      return {
        ...result,
        length: input.length,
        units: "nm",
        timingMs: performance.now() - started,
      };
    } finally {
      ops?.cleanup();
      e.trim();
      if (checkValidation) {
        const error = await device.popErrorScope();
        if (error) throw Error(error.message);
      }
    }
  }
  async function locked(fn) {
    if (busy) throw Error("Engine is busy");
    if (destroyed) throw Error("Engine destroyed");
    busy = true;
    try {
      return await fn();
    } finally {
      busy = false;
    }
  }
  return {
    manifest,
    precision,
    adapterInfo: acquired.adapterInfo,
    isFallbackAdapter: acquired.isFallbackAdapter,
    forward: (x, o) => locked(() => execute(x, o)),
    sample: (o) => locked(() => sampleWith(execute, o)),
    memoryUsage,
    destroy,
  };
}
