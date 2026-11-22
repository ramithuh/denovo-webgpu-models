import { loadWeights, decodeTensor } from "../shared/weights.js";
import { normalizeTokens } from "../shared/tokenizer.js";
import { generate, validLength } from "../shared/sampling.js";
import * as shaders from "./shaders.js";

export async function createEvoDiffEngine({
  baseUrl,
  precision = "int8",
  device,
  cache = "cache",
  useF16 = false,
  read,
  signal,
  onProgress,
} = {}) {
  if (!["fp32", "fp16", "int8"].includes(precision))
    throw new RangeError("Invalid precision");
  signal?.throwIfAborted();
  const owned = !device;
  if (!device) {
    if (!globalThis.navigator?.gpu)
      throw new Error("Hardware WebGPU is required");
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("Hardware WebGPU adapter missing");
    rejectSoftware(adapter.info);
    if (useF16 && !adapter.features.has("shader-f16"))
      throw new Error("shader-f16 unavailable");
    device = await adapter.requestDevice({
      requiredFeatures: useF16 ? ["shader-f16"] : [],
    });
  }
  rejectSoftware(device.adapterInfo);
  if (useF16 && !device.features.has("shader-f16"))
    throw new Error("Device needs shader-f16");
  const allocations = new Map(),
    weights = {},
    pipes = {},
    scratch = {};
  let destroyed = false,
    busy = false,
    lost = null,
    capacity = 0;
  device.lost.then((info) => {
    lost = info.message || "GPU device lost";
  });
  function alloc(size, usage, kind, label) {
    size = Math.max(4, Math.ceil(size / 4) * 4);
    if (
      size > device.limits.maxBufferSize ||
      (usage & GPUBufferUsage.STORAGE &&
        size > device.limits.maxStorageBufferBindingSize)
    )
      throw new RangeError(`GPU buffer limit: ${label} (${size})`);
    const b = device.createBuffer({ size, usage, label });
    allocations.set(b, { size, kind });
    return b;
  }
  function free(b) {
    if (b) {
      b.destroy();
      allocations.delete(b);
    }
  }
  function upload(data, label) {
    const b = alloc(
      data.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      "weights",
      label,
    );
    device.queue.writeBuffer(b, 0, data);
    return b;
  }
  function destroy() {
    if (destroyed) return;
    destroyed = true;
    for (const b of allocations.keys()) b.destroy();
    allocations.clear();
    if (owned) device.destroy();
  }
  function memoryUsage() {
    const result = {
      weightBytes: 0,
      scratchBytes: 0,
      transientBytes: 0,
      transientBuffers: 0,
      totalBytes: 0,
      buffers: allocations.size,
    };
    for (const { size, kind } of allocations.values()) {
      result.totalBytes += size;
      result[
        kind === "weights"
          ? "weightBytes"
          : kind === "scratch"
            ? "scratchBytes"
            : "transientBytes"
      ] += size;
      if (kind === "transient") result.transientBuffers++;
    }
    return result;
  }
  try {
    await loadWeights(`${baseUrl.replace(/\/$/, "")}/${precision}`, {
      expectedPrecision: precision,
      read,
      cache,
      signal,
      onProgress,
      onTensor(name, t) {
        let data;
        if (t.shape.length === 3) {
          // Transpose storage O,I,K -> (I*K),O without changing quantized values.
          const O = t.shape[0],
            K = t.shape[1] * t.shape[2],
            width = { F32: 4, F16: 2, I8: 1 }[t.dtype];
          data = new Uint8Array(Math.ceil(t.bytes.length / 4) * 4);
          for (let o = 0; o < O; o++)
            for (let k = 0; k < K; k++)
              for (let b = 0; b < width; b++)
                data[(k * O + o) * width + b] =
                  t.bytes[(o * K + k) * width + b];
        } else data = decodeTensor(t);
        weights[name] = {
          buffer: upload(data, name),
          dtype: t.dtype,
          shape: t.shape,
        };
        if (t.scales) weights[name].scale = upload(t.scales, name + ".__scale");
      },
    });
    weights.dummy = { buffer: upload(new Float32Array([1]), "unit scale") };
    const codes = {
      embedding: shaders.embedding,
      norm: shaders.norm,
      residual: shaders.residual,
      F32: shaders.convolution("F32"),
      F16: shaders.convolution("F16", useF16),
      I8: shaders.convolution("I8"),
    };
    for (const [name, code] of Object.entries(codes)) {
      const module = device.createShaderModule({ code, label: name });
      const info = await module.getCompilationInfo();
      if (info.messages.some((m) => m.type === "error"))
        throw new Error(
          info.messages.map((m) => `${m.lineNum}: ${m.message}`).join("\n"),
        );
      pipes[name] = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module, entryPoint: "main" },
        label: name,
      });
    }
    signal?.throwIfAborted();
  } catch (e) {
    destroy();
    throw e;
  }

  function ensure(L) {
    if (L <= capacity) return;
    const max = Math.min(
      device.limits.maxBufferSize,
      device.limits.maxStorageBufferBindingSize,
    );
    if (
      L * 1024 * 4 > max ||
      Math.ceil((L * 1024) / 256) >
        device.limits.maxComputeWorkgroupsPerDimension
    )
      throw new RangeError("Sequence exceeds GPU limits");
    for (const b of Object.values(scratch)) free(b);
    for (const k of Object.keys(scratch)) delete scratch[k];
    capacity = 0;
    const storage =
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST;
    try {
      for (const name of ["x", "a", "b", "c"])
        scratch[name] = alloc(L * 1024 * 4, storage, "scratch", name);
      scratch.tokens = alloc(L * 4, storage, "scratch", "tokens");
      scratch.mask = alloc(L * 4, storage, "scratch", "mask");
      scratch.params = alloc(
        160 * 256,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        "scratch",
        "params",
      );
      capacity = L;
    } catch (e) {
      for (const b of Object.values(scratch)) free(b);
      for (const k of Object.keys(scratch)) delete scratch[k];
      throw e;
    }
  }
  async function forward(
    input,
    { returnIntermediates = false, signal, inputMask } = {},
  ) {
    if (destroyed || lost) throw new Error(lost || "Engine destroyed");
    signal?.throwIfAborted();
    const start = performance.now(),
      tokens = normalizeTokens(input),
      L = validLength(tokens.length);
    if (
      inputMask &&
      (inputMask.length !== L ||
        Array.from(inputMask).some((x) => x !== 0 && x !== 1))
    )
      throw new RangeError("inputMask must contain one 0/1 per position");
    ensure(L);
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");
    let staging,
      scopeDone = false;
    try {
      device.queue.writeBuffer(scratch.tokens, 0, tokens);
      device.queue.writeBuffer(
        scratch.mask,
        0,
        inputMask ? Float32Array.from(inputMask) : new Float32Array(L).fill(1),
      );
      const encoder = device.createCommandEncoder(),
        params = new Uint32Array(160 * 64),
        copies = [];
      let op = 0,
        offset = 0;
      const traceSize = returnIntermediates
        ? (8 + 1024 + 16 * 1024 + 1024 + 31) * L * 4
        : L * 31 * 4;
      staging = alloc(
        traceSize,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        "transient",
        "readback",
      );
      const record = (name, buffer, width) => {
        if (returnIntermediates || name === "logits") {
          const bytes = L * width * 4;
          encoder.copyBufferToBuffer(buffer, 0, staging, offset, bytes);
          copies.push({ name, offset, bytes });
          offset += bytes;
        }
      };
      const dispatch = (kind, buffers, values, groups) => {
        if (op >= 160) throw new Error("Too many dispatches");
        params.set([L, ...values], op * 64);
        const pipeline = pipes[kind],
          entries = buffers
            .map((buffer, binding) => ({ binding, resource: { buffer } }))
            .filter((e) => !(["F32", "F16"].includes(kind) && e.binding === 3));
        entries.push({
          binding: buffers.length,
          resource: { buffer: scratch.params, offset: op * 256, size: 32 },
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(
          0,
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries,
          }),
        );
        pass.dispatchWorkgroups(...groups);
        pass.end();
        op++;
      };
      const weight = (n) => {
        if (!weights[n]) throw new Error(`Missing ${n}`);
        return weights[n].buffer;
      };
      const norm = (x, y, prefix, C, act = 1) =>
        dispatch(
          "norm",
          [x, weight(prefix + ".weight"), weight(prefix + ".bias"), y],
          [C, 0, 0, 0, act, 0],
          [L],
        );
      const conv = (x, y, prefix, CI, CO, KW = 1, D = 1) => {
        const w = weights[prefix + ".weight"];
        dispatch(
          w.dtype,
          [
            x,
            w.buffer,
            weight(prefix + ".bias"),
            w.scale ?? weights.dummy.buffer,
            scratch.mask,
            y,
          ],
          [CI, CO, KW, D, 0, KW === 5 && inputMask ? 1 : 0],
          [Math.ceil(CO / 32), Math.ceil(L / 8)],
        );
      };
      dispatch(
        "embedding",
        [scratch.tokens, weight("embedder.embedder.weight"), scratch.a],
        [0, 0, 0, 0, 0, 0],
        [Math.ceil((L * 8) / 256)],
      );
      record("embedding", scratch.a, 8);
      conv(scratch.a, scratch.x, "embedder.up_embedder.conv", 8, 1024);
      record("projection", scratch.x, 1024);
      for (let i = 0; i < 16; i++) {
        signal?.throwIfAborted();
        const p = `embedder.layers.${i}`;
        norm(scratch.x, scratch.a, p + ".sequence1.0", 1024);
        conv(scratch.a, scratch.b, p + ".sequence1.2.conv", 1024, 512);
        norm(scratch.b, scratch.a, p + ".sequence1.3", 512);
        conv(scratch.a, scratch.b, p + ".conv", 512, 512, 5, 2 ** (i % 8));
        norm(scratch.b, scratch.a, p + ".sequence2.0", 512);
        conv(scratch.a, scratch.c, p + ".sequence2.2.conv", 512, 1024);
        dispatch(
          "residual",
          [scratch.c, scratch.x],
          [0, 0, 0, 0, 0, 0],
          [Math.ceil((L * 1024) / 256)],
        );
        record(`block.${i}`, scratch.x, 1024);
      }
      norm(scratch.x, scratch.a, "last_norm", 1024, 0);
      record("final_norm", scratch.a, 1024);
      conv(scratch.a, scratch.b, "decoder.conv", 1024, 31);
      record("logits", scratch.b, 31);
      device.queue.writeBuffer(scratch.params, 0, params);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const oom = await device.popErrorScope(),
        validation = await device.popErrorScope();
      scopeDone = true;
      if (oom || validation) throw new Error((oom || validation).message);
      if (lost) throw new Error(lost);
      signal?.throwIfAborted();
      await staging.mapAsync(GPUMapMode.READ);
      const mapped = staging.getMappedRange(),
        trace = {};
      for (const c of copies)
        trace[c.name] = new Float32Array(mapped, c.offset, c.bytes / 4).slice();
      staging.unmap();
      signal?.throwIfAborted();
      return {
        logits: trace.logits,
        tokens,
        timingMs: performance.now() - start,
        ...(returnIntermediates ? { intermediates: trace } : {}),
      };
    } finally {
      free(staging);
      if (!scopeDone) {
        await device.popErrorScope();
        await device.popErrorScope();
      }
    }
  }
  async function exclusive(fn) {
    if (busy) throw new Error("Engine busy");
    if (destroyed) throw new Error("Engine destroyed");
    busy = true;
    try {
      return await fn();
    } finally {
      busy = false;
    }
  }
  return {
    forward: (input, options) => exclusive(() => forward(input, options)),
    generate: (options) => exclusive(() => generate(forward, options)),
    memoryUsage,
    destroy,
  };
}
export function rejectSoftware(info) {
  if (
    info?.isFallbackAdapter ||
    /swiftshader|llvmpipe|lavapipe|software|fallback/i.test(
      [info?.vendor, info?.architecture, info?.device, info?.description].join(
        " ",
      ),
    )
  )
    throw new Error("Software WebGPU adapters are unsupported");
}
