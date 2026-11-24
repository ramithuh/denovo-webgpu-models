import { constants } from "../shared/constants.js";
import {
  loadWeights,
  cathIndices,
  validateInput,
  gaussianNoise,
  schedule,
  toPDB,
} from "../shared/index.js";
import { forwardNetwork, assertModelManifest } from "../shared/model.js";
import { acquireDevice } from "./device.js";
import * as S from "./shaders.js";

const f = Math.fround;
const bits = (v) => new Uint32Array(Float32Array.of(v).buffer)[0];

export class GpuBackend {
  constructor(meta, { useF16 = true } = {}) {
    Object.assign(this, meta);
    this.useF16 = useF16 && this.device.features.has("shader-f16");
    this.weights = {};
    this.buffers = new Set();
    this.pipelines = new Map();
    this.pass = null;
    this.destroyed = false;
    this.busy = false;
    this.device.lost.then(
      (info) => (this.lost = info.message || "WebGPU device lost"),
    );
    try {
      this.zero = this.buffer(new Float32Array(4));
      this.freq = this.buffer(Float32Array.from(constants.time));
      this.posfreq = this.buffer(Float32Array.from(constants.position));
      this.bounds = this.buffer(Float32Array.from(constants.distance));
    } catch (e) {
      this.destroy();
      throw e;
    }
  }

  check() {
    if (this.destroyed) throw new Error("Engine destroyed");
    if (this.lost) throw new Error(this.lost);
    this.signal?.throwIfAborted();
  }

  buffer(value, temp = false, pinned = false) {
    this.check();
    const data = typeof value === "number" ? null : value;
    const size = Math.max(4, Math.ceil((data?.byteLength ?? value) / 4) * 4);
    if (
      size > this.device.limits.maxBufferSize ||
      size > this.device.limits.maxStorageBufferBindingSize
    ) {
      throw new RangeError(
        `Buffer ${size} exceeds maxBufferSize/maxStorageBufferBindingSize`,
      );
    }
    let buf = null;
    if (temp && !pinned) {
      const list = this.pool.get(size);
      if (list && list.length) {
        if (!data) {
          buf = list.pop();
        } else {
          for (let i = list.length - 1; i >= 0; i--) {
            if (!this.dirtyInSubmit.has(list[i])) {
              buf = list.splice(i, 1)[0];
              break;
            }
          }
        }
      }
    }
    if (!buf) {
      buf = this.device.createBuffer({
        size,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
    }
    this.buffers.add(buf);
    if (temp) {
      this.temps.add(buf);
      if (!pinned) this.live.add(buf);
    }
    if (data) {
      const padded = new Uint8Array(size);
      padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      this.device.queue.writeBuffer(buf, 0, padded);
    }
    return buf;
  }

  _uniform(words) {
    const key = words.join(",");
    let buf = this.parameters.get(key);
    if (!buf) {
      const size = Math.max(16, Math.ceil(words.byteLength / 16) * 16);
      if (
        size > this.device.limits.maxBufferSize ||
        size > this.device.limits.maxStorageBufferBindingSize
      ) {
        throw new RangeError("Uniform exceeds device buffer limits");
      }
      buf = this.device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(buf, 0, words);
      this.buffers.add(buf);
      this.temps.add(buf);
      this.parameters.set(key, buf);
    }
    return buf;
  }

  tensor(data, rows, cols) {
    return { buffer: this.buffer(data, true), rows, cols };
  }

  async upload(name, t) {
    this.weights[name] = {
      buffer: this.buffer(t.bytes),
      dtype: t.dtype,
      rows: t.shape[0],
      cols: t.shape[1] ?? 1,
      scales: t.scales ? this.buffer(t.scales) : this.zero,
    };
  }

  memoryUsage() {
    const weightBytes = Object.values(this.weights).reduce(
      (n, w) =>
        n + w.buffer.size + (w.scales === this.zero ? 0 : w.scales.size),
      0,
    );
    return {
      weightBytes,
      totalBytes: [...this.buffers].reduce((n, b) => n + b.size, 0),
      bufferCount: this.buffers.size,
      transientBuffers: this.temps?.size ?? 0,
      scratchReuses: this.reuses ?? 0,
    };
  }

  keep(tensors) {
    const retained = new Set(tensors.filter(Boolean).map((t) => t.buffer));
    for (const t of this.pinnedTensors) retained.add(t.buffer);
    for (const b of this.live) {
      if (!retained.has(b)) {
        if (!this.pool.has(b.size)) this.pool.set(b.size, []);
        this.pool.get(b.size).push(b);
        this.live.delete(b);
        this.reuses = (this.reuses ?? 0) + 1;
      }
    }
  }

  _beginPass() {
    this.encoder ??= this.device.createCommandEncoder();
    this.pass ??= this.encoder.beginComputePass();
    return this.pass;
  }

  _endPass() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }

  _flush() {
    if (this.encoder) {
      this._endPass();
      const enc = this.encoder;
      this.encoder = null;
      this.dirtyInSubmit.clear();
      this.dispatches = 0;
      this.device.queue.submit([enc.finish()]);
    }
  }

  _getPipeline(key, code) {
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  dispatchSync(key, code, uniformWords, inputs, rows, cols, groups) {
    this.check();
    if (
      groups.some(
        (x) => x > this.device.limits.maxComputeWorkgroupsPerDimension,
      )
    ) {
      throw new RangeError("Dispatch exceeds WebGPU workgroup limits");
    }
    const y = { buffer: this.buffer(rows * cols * 4, true), rows, cols };
    const pipeline = this._getPipeline(key, code);
    const uBuf = this._uniform(uniformWords);
    const allStorage = [...inputs, y.buffer];
    const entries = [{ binding: 0, resource: { buffer: uBuf } }];
    for (let i = 0; i < allStorage.length; i++) {
      this.dirtyInSubmit.add(allStorage[i]);
      entries.push({ binding: i + 1, resource: { buffer: allStorage[i] } });
    }
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    const pass = this._beginPass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(...groups);
    if (++this.dispatches >= 512) this._flush();
    return y;
  }

  element(name, a, b, c, rows, cols, ex0 = 0, ex1 = 0) {
    return this.dispatchSync(
      `elem:${name}`,
      S.elements[name],
      new Uint32Array([rows * cols, cols, ex0, ex1]),
      [
        a?.buffer ?? this.zero,
        b?.buffer ?? this.zero,
        c?.buffer ?? this.zero,
      ],
      rows,
      cols,
      S.fold2(Math.ceil((rows * cols) / 256)),
    );
  }

  linear(a, name) {
    const w = this.weights[name + ".weight"];
    if (!w || w.cols !== a.cols) {
      throw new Error(`Linear shape mismatch ${name}`);
    }
    const bias = this.weights[name + ".bias"];
    const hasBias = !!bias;
    const native = w.dtype === "F16" && this.useF16;
    const u = new Uint32Array([a.rows, w.rows, w.cols, hasBias ? 1 : 0]);
    if (a.rows <= 4) {
      return this.dispatchSync(
        `gemv:${w.dtype}:${hasBias ? 1 : 0}:${native ? 1 : 0}`,
        S.gemv(w.dtype, hasBias, native),
        u,
        [a.buffer, w.buffer, bias?.buffer ?? this.zero, w.scales],
        a.rows,
        w.rows,
        S.fold2(Math.ceil((a.rows * w.rows) / 64)),
      );
    }
    const tileM = a.rows >= 192 ? 64 : 32;
    const groups = Math.ceil(a.rows / tileM) * Math.ceil(w.rows / 128);
    return this.dispatchSync(
      `matmul:${tileM}:${w.dtype}:${hasBias ? 1 : 0}:${native ? 1 : 0}`,
      S.matmul(w.dtype, hasBias, native, tileM),
      u,
      [a.buffer, w.buffer, bias?.buffer ?? this.zero, w.scales],
      a.rows,
      w.rows,
      S.fold2(groups),
    );
  }

  norm(a, name, opts = {}) {
    const width = opts.width ?? a.cols;
    const w = name ? this.weights[name + ".weight"] : null;
    const b = name ? this.weights[name + ".bias"] : null;
    return this.dispatchSync(
      "norm",
      S.norm,
      new Uint32Array([
        width,
        opts.stride ?? a.cols,
        opts.offset ?? 0,
        w ? 1 : 0,
        a.rows,
        0,
        0,
        0,
      ]),
      [a.buffer, w?.buffer ?? this.zero, b?.buffer ?? this.zero],
      a.rows,
      width,
      S.fold2(a.rows),
    );
  }

  fastAdaptive(x, g, b, mode) {
    return this.dispatchSync(
      "norm",
      S.norm,
      new Uint32Array([x.cols, x.cols, 0, 1, x.rows, 1, mode, 0]),
      [x.buffer, g.buffer, b.buffer],
      x.rows,
      x.cols,
      S.fold2(x.rows),
    );
  }

  pairStats(pair) {
    return this.dispatchSync(
      "norm",
      S.norm,
      new Uint32Array([pair.cols, pair.cols, 0, 0, pair.rows, 2, 0, 0]),
      [pair.buffer, this.zero, this.zero],
      pair.rows,
      2,
      S.fold2(pair.rows),
    );
  }

  pairLinearInit(x, name) {
    const L = x.rows;
    const pairs = L * L;
    const idx = this.dispatchSync(
      "pairBins",
      S.pairBins,
      new Uint32Array([L, 0, 0, 0]),
      [x.buffer, this.bounds],
      pairs,
      2,
      S.fold2(Math.ceil(pairs / 64)),
    );
    const w = this.weights[name + ".weight"];
    const bias = this.weights[name + ".bias"];
    const hasBias = !!bias;
    const native = w.dtype === "F16" && this.useF16;
    return this.dispatchSync(
      `sparsePair:${w.dtype}:${hasBias ? 1 : 0}:${native ? 1 : 0}`,
      S.sparsePairLinear(w.dtype, hasBias, native),
      new Uint32Array([pairs, w.rows, w.cols, 0]),
      [idx.buffer, w.buffer, bias?.buffer ?? this.zero, w.scales],
      pairs,
      w.rows,
      S.fold2(Math.ceil((pairs * w.rows) / 256)),
    );
  }

  pairBias(pair, stats, normPrefix, biasPrefix) {
    const nw = this.weights[normPrefix + ".weight"];
    const nb = this.weights[normPrefix + ".bias"];
    const bw = this.weights[biasPrefix + ".weight"];
    const bb = this.weights[biasPrefix + ".bias"];
    const hasBias = !!bb;
    const native = bw.dtype === "F16" && this.useF16;
    return this.dispatchSync(
      `pairBias:${bw.dtype}:${hasBias ? 1 : 0}:${native ? 1 : 0}`,
      S.pairBias(bw.dtype, hasBias, native),
      new Uint32Array([pair.rows, pair.cols, bw.rows, 0]),
      [
        pair.buffer,
        stats.buffer,
        nw.buffer,
        nb.buffer,
        bw.buffer,
        bb?.buffer ?? this.zero,
        bw.scales,
      ],
      pair.rows,
      bw.rows,
      S.fold2(Math.ceil(pair.rows / 16)),
    );
  }

  time(t) {
    return this.element("time", { buffer: this.freq }, null, null, 1, 256, bits(t));
  }

  position(L) {
    return this.element("position", { buffer: this.posfreq }, null, null, L, 132);
  }

  async condition(time, ids) {
    const out = this.tensor(1024 * 4, 1, 1024);
    this._endPass();
    this.encoder ??= this.device.createCommandEncoder();
    this.dirtyInSubmit.add(time.buffer);
    this.dirtyInSubmit.add(out.buffer);
    this.encoder.copyBufferToBuffer(time.buffer, 0, out.buffer, 0, 1024);
    for (let k = 0; k < 3; k++) {
      const w = this.weights[`cond_factory.feat_creators.1.embedding_${"CAT"[k]}.weight`];
      const native = w.dtype === "F16" && this.useF16;
      const g = this.dispatchSync(
        `gather:${w.dtype}:${native ? 1 : 0}`,
        S.gather(w.dtype, native),
        new Uint32Array([256, 0, ids[k], 0]),
        [this.zero, w.buffer, w.scales],
        1,
        256,
        [1, 1],
      );
      this._endPass();
      this.dirtyInSubmit.add(g.buffer);
      this.dirtyInSubmit.add(out.buffer);
      this.encoder.copyBufferToBuffer(g.buffer, 0, out.buffer, (256 + 256 * k) * 4, 1024);
    }
    return out;
  }

  padCondition(a) {
    return this.element("padCondition", a, null, null, 2, 512);
  }

  registers(a) {
    const w = this.weights.registers;
    const native = w.dtype === "F16" && this.useF16;
    const total = (a.rows + 10) * 768;
    return this.dispatchSync(
      `gather:${w.dtype}:${native ? 1 : 0}`,
      S.gather(w.dtype, native),
      new Uint32Array([total, 1, 0, 0]),
      [a.buffer, w.buffer, w.scales],
      a.rows + 10,
      768,
      S.fold2(Math.ceil(total / 256)),
    );
  }

  padPair(a, L) {
    return this.element("padPair", a, null, null, (L + 10) ** 2, 512, L + 10);
  }

  add(a, b) {
    return this.element("add", a, b, null, a.rows, a.cols);
  }

  swiglu(a) {
    return this.element("swiglu", a, null, null, a.rows, a.cols / 2);
  }

  residualGate(x, h, g) {
    return this.element("residualGateAdd", x, h, g, x.rows, x.cols);
  }

  unpad(a, n) {
    return this.element("unpad", a, null, null, a.rows - n, a.cols, n);
  }

  velocity(x, v, t) {
    return this.element("velocityAdd", x, v, null, x.rows, 3, bits(t));
  }

  attention(q, k, qkv, bias, g, N) {
    const tilesN = Math.ceil(N / 32);
    const sc = this.dispatchSync(
      "scores",
      S.scores,
      new Uint32Array([N, 0, 0, 0]),
      [q.buffer, k.buffer, bias.buffer],
      12 * N,
      N,
      S.fold2(12 * tilesN * tilesN),
    );
    const p = this.dispatchSync(
      "softmax",
      S.softmax,
      new Uint32Array([N, 12 * N, 0, 0]),
      [sc.buffer],
      12 * N,
      N,
      S.fold2(12 * N),
    );
    return this.dispatchSync(
      "context",
      S.context,
      new Uint32Array([N, 0, 0, 0]),
      [p.buffer, qkv.buffer, g.buffer],
      N,
      768,
      S.fold2(12 * tilesN),
    );
  }

  center(x) {
    const mean = this.dispatchSync(
      "coordMean",
      S.coordinateMean,
      new Uint32Array([x.rows, 0, 0, 0]),
      [x.buffer],
      1,
      3,
      [1, 1],
    );
    return this.element("center", x, mean, null, x.rows, 3);
  }

  scale(x, v) {
    return this.element("scale", x, null, null, x.rows, x.cols, bits(v));
  }

  mix(prev, noise, t, scale) {
    const z = this.center(this.scale(this.center(noise), scale));
    const p = this.center(prev);
    return this.add(this.scale(z, f(1 - t)), this.scale(p, t));
  }

  begin(signal) {
    this.check();
    if (this.busy) throw new Error("Engine busy");
    this.busy = true;
    this.signal = signal;
    this.temps = new Set();
    this.live = new Set();
    this.pinnedTensors = new Set();
    this.dirtyInSubmit = new Set();
    this.dispatches = 0;
    this.pool = new Map();
    this.parameters = new Map();
    this.encoder = this.device.createCommandEncoder();
    this.pass = null;
    this.device.pushErrorScope("validation");
  }

  async read(a) {
    this.check();
    this._flush();
    const size = a.rows * a.cols * 4;
    const readback = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.buffers.add(readback);
    this.temps.add(readback);
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(a.buffer, 0, readback, 0, size);
    this.device.queue.submit([enc.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      this.check();
      return new Float32Array(readback.getMappedRange().slice(0));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
      this.buffers.delete(readback);
      this.temps.delete(readback);
    }
  }

  async readMany(tensors) {
    this.check();
    this._flush();
    let totalBytes = 0;
    const offsets = [];
    for (const t of tensors) {
      offsets.push(totalBytes);
      totalBytes += t.rows * t.cols * 4;
    }
    const readback = this.device.createBuffer({
      size: Math.max(4, totalBytes),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.buffers.add(readback);
    this.temps.add(readback);
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < tensors.length; i++) {
      enc.copyBufferToBuffer(
        tensors[i].buffer,
        0,
        readback,
        offsets[i],
        tensors[i].rows * tensors[i].cols * 4,
      );
    }
    this.device.queue.submit([enc.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      this.check();
      const mapped = readback.getMappedRange();
      return tensors.map(
        (t, i) =>
          new Float32Array(
            mapped.slice(offsets[i], offsets[i] + t.rows * t.cols * 4),
          ),
      );
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
      this.buffers.delete(readback);
      this.temps.delete(readback);
    }
  }

  async end() {
    let validation;
    try {
      this._endPass();
      this.encoder = null;
      validation = await this.device.popErrorScope();
    } finally {
      for (const b of this.temps) {
        b.destroy();
        this.buffers.delete(b);
      }
      this.temps.clear();
      this.live.clear();
      this.pinnedTensors.clear();
      this.dirtyInSubmit.clear();
      this.pool.clear();
      this.parameters.clear();
      this.encoder = null;
      this.pass = null;
      this.busy = false;
      this.signal = null;
    }
    if (validation) throw new Error("WebGPU validation: " + validation.message);
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this._endPass();
    this.encoder = null;
    for (const b of this.buffers) b.destroy();
    this.buffers.clear();
    this.weights = {};
    this.temps?.clear();
    this.pipelines.clear();
    if (this.ownsDevice) this.device.destroy();
  }
}

export async function createSidProteinEngine({
  baseUrl,
  precision = "fp32",
  device,
  cache = "none",
  useF16 = true,
  allowSoftwareFallback = true,
  signal,
  onProgress,
  read,
} = {}) {
  if (!["fp32", "fp16", "int8"].includes(precision)) {
    throw new RangeError("Unknown precision");
  }
  signal?.throwIfAborted();
  const meta = await acquireDevice({ device, useF16, allowSoftwareFallback });
  const b = new GpuBackend(meta, { useF16 });
  try {
    const bundle = await loadWeights(
      baseUrl ?? new URL(`../weights/${precision}`, import.meta.url).href,
      { cache, signal, onProgress, read, onTensor: (n, t) => b.upload(n, t) },
    );
    assertModelManifest(bundle.manifest);
    if (bundle.manifest.precision !== precision) {
      throw new Error("Precision mismatch");
    }
    b.manifest = bundle.manifest;
  } catch (e) {
    b.destroy();
    throw e;
  }
  let sampling = false;
  async function run(coordinates, t, opts = {}, mixNoise) {
    validateInput(coordinates, t);
    b.begin(opts.signal);
    try {
      let x = b.tensor(coordinates, coordinates.length / 3, 3);
      if (mixNoise) {
        x = await b.mix(
          x,
          b.tensor(mixNoise.noise, x.rows, 3),
          t,
          mixNoise.scale,
        );
      }
      if (mixNoise && opts.onTrace) {
        await opts.onTrace("sampler.input", await b.read(x));
      }
      const out = await forwardNetwork(
        b,
        x,
        t,
        cathIndices(opts),
        opts.onTrace ? async (k, v) => opts.onTrace(k, await b.read(v)) : null,
      );
      return await b.read(out);
    } finally {
      await b.end();
    }
  }
  async function forward(coordinates, t, opts = {}) {
    if (sampling) throw new Error("Engine busy sampling");
    return run(coordinates, f(t), opts);
  }
  async function generate(opts = {}) {
    b.check();
    if (sampling || b.busy) throw new Error("Engine busy");
    const {
      length = 64,
      steps = 16,
      noiseScale = 0.45,
      seed = 5,
      tInit = 0.37,
      onestepGamma = 1,
      signal,
    } = opts;
    if (!Number.isInteger(length) || length < 1) {
      throw new RangeError("Invalid length");
    }
    if (
      !Number.isFinite(noiseScale) ||
      noiseScale < 0 ||
      !Number.isFinite(onestepGamma) ||
      onestepGamma < 0
    ) {
      throw new RangeError("Invalid noise scale");
    }
    const cath = cathIndices(opts);
    const times = schedule(steps, tInit);
    const noise = opts.noise ?? gaussianNoise(length, steps, seed);
    if (
      !(noise instanceof Float32Array) ||
      noise.length !== length * 3 * steps ||
      noise.some((v) => !Number.isFinite(v))
    ) {
      throw new RangeError("Invalid noise draws");
    }
    const scaleVal = steps === 1 ? onestepGamma : noiseScale;

    // Fast single-session GPU path when no per-step JS callback/trace is requested:
    // keeps scratch pool resident across all 16 steps and performs ONE batched readback at the end!
    if (!opts.onStep && !opts.onTrace) {
      sampling = true;
      b.begin(signal);
      try {
        let xGpu = b.tensor(new Float32Array(length * 3), length, 3);
        const trajGpu = [];
        for (let i = 0; i < steps; i++) {
          b.check();
          const stepT = f(times[i]);
          const noiseSlice = noise.subarray(i * length * 3, (i + 1) * length * 3);
          const mixed = b.mix(
            xGpu,
            b.tensor(noiseSlice, length, 3),
            stepT,
            scaleVal,
          );
          xGpu = await forwardNetwork(b, mixed, stepT, cath, null);
          trajGpu.push(xGpu);
          b.pinnedTensors.add(xGpu);
          b.keep([xGpu]);
          b._flush();
        }
        const trajectory = await b.readMany(trajGpu);
        const coordinates = trajectory[trajectory.length - 1];
        return { coordinates, trajectory, times, units: "nm", pdb: toPDB(coordinates) };
      } finally {
        sampling = false;
        await b.end();
      }
    }

    let x = new Float32Array(length * 3);
    const trajectory = [];
    sampling = true;
    try {
      for (let i = 0; i < steps; i++) {
        signal?.throwIfAborted();
        x = await run(x, f(times[i]), opts, {
          noise: noise.subarray(i * length * 3, (i + 1) * length * 3),
          scale: scaleVal,
        });
        trajectory.push(x);
        await opts.onStep?.({ step: i + 1, steps, coordinates: x });
      }
      return { coordinates: x, trajectory, times, units: "nm", pdb: toPDB(x) };
    } finally {
      sampling = false;
    }
  }
  return {
    forward,
    generate,
    adapterInfo: b.adapterInfo,
    isFallbackAdapter: b.isFallbackAdapter,
    precision,
    useF16: b.useF16,
    manifest: b.manifest,
    memoryUsage: () => b.memoryUsage(),
    destroy: () => b.destroy(),
  };
}
export { createSidProteinWorker } from "./worker-client.js";
