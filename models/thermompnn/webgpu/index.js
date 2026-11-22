import { loadWeights, checkAbort } from "../shared/loader.js";
import { parseStructure } from "../shared/structure.js";
import { prepareInput, resultFrom } from "../shared/results.js";
import { executeGraph } from "../shared/graph.js";
import { GPUOps } from "./ops.js";
export function assertHardware(adapter) {
  if (!adapter) throw new Error("Hardware WebGPU adapter is required");
  const info = adapter.info ?? adapter.adapterInfo ?? {};
  if (
    adapter.isFallbackAdapter ||
    info.isFallbackAdapter ||
    /swiftshader|llvmpipe|lavapipe|software|\bwarp\b|basic render/i.test(
      [info.vendor, info.architecture, info.device, info.description].join(" "),
    )
  )
    throw new Error("Software WebGPU adapters are not supported");
  return info;
}
export async function createThermoMPNNEngine(options = {}) {
  const {
    baseUrl,
    precision = "fp32",
    cache = "none",
    signal,
    onProgress,
    read,
  } = options;
  checkAbort(signal);
  let device = options.device,
    ownsDevice = !device;
  if (!device) {
    if (!globalThis.navigator?.gpu)
      throw new Error("Hardware WebGPU is unavailable");
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    assertHardware(adapter);
    const features =
      options.useF16 && adapter.features.has("shader-f16")
        ? ["shader-f16"]
        : [];
    device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
  } else assertHardware(device);
  const owned = new Map(),
    pipelines = new Map(),
    zeros = new Map();
  let destroyed = false,
    busy = false,
    lost = null;
  device.lost.then((info) => {
    lost = info.message || info.reason;
  });
  const engine = {
    device,
    w: {},
    useF16: !!options.useF16 && device.features.has("shader-f16"),
    ensureAlive() {
      if (destroyed) throw new Error("Engine is destroyed");
      if (lost) throw new Error(`WebGPU device lost: ${lost}`);
    },
    admit(size, storage = true) {
      if (
        !Number.isSafeInteger(size) ||
        size < 4 ||
        size > device.limits.maxBufferSize ||
        (storage && size > device.limits.maxStorageBufferBindingSize)
      )
        throw new Error(
          `WebGPU buffer admission failed: ${size} bytes exceeds ${storage ? "maxStorageBufferBindingSize / " : ""}maxBufferSize (${device.limits.maxStorageBufferBindingSize} / ${device.limits.maxBufferSize})`,
        );
    },
    buffer(size, usage, type) {
      engine.admit(size, !!(usage & GPUBufferUsage.STORAGE));
      const b = device.createBuffer({ size: Math.ceil(size / 4) * 4, usage });
      owned.set(b, { size: b.size, type });
      return b;
    },
    release(b) {
      if (owned.has(b)) {
        b.destroy();
        owned.delete(b);
      }
    },
    zero(n) {
      if (!zeros.has(n)) {
        const buffer = engine.buffer(
          n * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          "resident",
        );
        zeros.set(n, { buffer, size: n * 4 });
      }
      return zeros.get(n);
    },
    pipeline(code) {
      if (!pipelines.has(code)) {
        const module = device.createShaderModule({ code });
        const entries = [
          ...code.matchAll(/@binding\((\d+)\) var<storage,(read_write|read)>/g),
        ].map((m) => ({
          binding: Number(m[1]),
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: m[2] === "read" ? "read-only-storage" : "storage" },
        }));
        const layout = device.createPipelineLayout({
          bindGroupLayouts: [device.createBindGroupLayout({ entries })],
        });
        pipelines.set(
          code,
          device.createComputePipeline({
            layout,
            compute: { module, entryPoint: "main" },
          }),
        );
      }
      return pipelines.get(code);
    },
  };
  function cleanup() {
    for (const b of owned.keys()) engine.release(b);
    pipelines.clear();
    zeros.clear();
    engine.w = {};
    if (ownsDevice) device.destroy();
  }
  try {
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    let weights;
    try {
      weights = await loadWeights({
        baseUrl,
        precision,
        cache,
        signal,
        onProgress,
        read,
      });
      checkAbort(signal);
      for (const [name, t] of Object.entries(weights.tensors)) {
        const size = Math.ceil(t.bytes.byteLength / 4) * 4,
          buffer = engine.buffer(
            size,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "resident",
          ),
          bytes = new Uint8Array(size);
        bytes.set(t.bytes);
        device.queue.writeBuffer(buffer, 0, bytes);
        const record = { buffer, size, shape: t.shape, dtype: t.dtype };
        if (t.scale) {
          const sb = engine.buffer(
            t.scale.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "resident",
          );
          device.queue.writeBuffer(sb, 0, t.scale);
          record.scale = { buffer: sb, size: t.scale.byteLength };
        }
        engine.w[name] = record;
      }
    } finally {
      const validation = await device.popErrorScope(),
        memory = await device.popErrorScope();
      if (validation || memory) throw new Error((validation ?? memory).message);
    }
    const manifest = weights.manifest;
    return {
      manifest,
      precision,
      useF16: engine.useF16,
      adapterInfo: device.adapterInfo,
      async predict(text, options = {}) {
        engine.ensureAlive();
        if (busy) throw new Error("Inference already in progress");
        checkAbort(options.signal);
        // Admission happens before any inference buffers or dispatches exist.
        const { structure, mutations } = prepareInput(
          text,
          { ...options, maxLength: Math.min(options.maxLength ?? 2048, 2048) },
          parseStructure,
        );
        const L = structure.length,
          K = Math.min(48, L);
        engine.admit(L * K * 512 * 4);
        engine.admit(L * L * 4);
        busy = true;
        const start = performance.now();
        let ops, result, failure;
        device.pushErrorScope("out-of-memory");
        device.pushErrorScope("validation");
        try {
          ops = new GPUOps(engine, structure, options);
          const tensor = await executeGraph(ops, structure);
          checkAbort(options.signal);
          const values = await ops.readAll(tensor);
          let traces;
          if (options.returnIntermediates) {
            traces = { ...values };
            delete traces.output;
            for (const name of ["X", "S", "mask", "chain", "residue_idx"])
              traces[name] = structure[name].slice();
            traces.targeted = Float32Array.from(
              { length: L },
              (_, i) => values.output[i * 20],
            );
          }
          result = resultFrom(
            structure,
            values.output,
            mutations,
            performance.now() - start,
            traces,
          );
        } catch (error) {
          failure = error;
        } finally {
          ops?.dispose();
          try {
            const validation = await device.popErrorScope(),
              memory = await device.popErrorScope();
            if (!failure && (validation || memory))
              failure = new Error((validation ?? memory).message);
          } finally {
            busy = false;
          }
        }
        if (failure) throw failure;
        return result;
      },
      memoryUsage() {
        let residentBytes = 0,
          transientBytes = 0,
          residentBuffers = 0,
          transientBuffers = 0;
        for (const { size, type } of owned.values()) {
          if (type === "resident") {
            residentBytes += size;
            residentBuffers++;
          } else {
            transientBytes += size;
            transientBuffers++;
          }
        }
        return {
          residentBytes,
          residentBuffers,
          transientBytes,
          transientBuffers,
          totalBytes: residentBytes + transientBytes,
        };
      },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        cleanup();
      },
    };
  } catch (error) {
    destroyed = true;
    cleanup();
    throw error;
  }
}
