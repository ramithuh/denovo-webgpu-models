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
const f = Math.fround,
  bits = (v) => new Uint32Array(Float32Array.of(v).buffer)[0];
export class GpuBackend {
  constructor(meta, { useF16 = true } = {}) {
    Object.assign(this, meta);
    this.useF16 = useF16 && this.device.features.has("shader-f16");
    this.weights = {};
    this.buffers = new Set();
    this.pipelines = new Map();
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
    const data = typeof value === "number" ? null : value,
      size = Math.max(4, Math.ceil((data?.byteLength ?? value) / 4) * 4);
    if (
      size > this.device.limits.maxBufferSize ||
      size > this.device.limits.maxStorageBufferBindingSize
    )
      throw new RangeError(
        `Buffer ${size} exceeds maxBufferSize/maxStorageBufferBindingSize`,
      );
    const buf =
      (temp && !data && !pinned ? this.pool.get(size)?.pop() : null) ??
      this.device.createBuffer({
        size,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
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
  tensor(data, rows, cols) {
    return { buffer: this.buffer(data, true), rows, cols };
  }
  params(a) {
    const key = a.join(",");
    if (!this.parameters.has(key))
      this.parameters.set(key, this.buffer(Uint32Array.from(a), true, true));
    return this.parameters.get(key);
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
    const retained = new Set(tensors.map((t) => t.buffer));
    for (const b of this.live)
      if (!retained.has(b)) {
        if (!this.pool.has(b.size)) this.pool.set(b.size, []);
        this.pool.get(b.size).push(b);
        this.live.delete(b);
        this.reuses = (this.reuses ?? 0) + 1;
      }
  }
  async pipeline(code, count) {
    let promise = this.pipelines.get(code);
    if (!promise) {
      promise = (async () => {
        const module = this.device.createShaderModule({ code }),
          info = await module.getCompilationInfo(),
          errors = info.messages.filter((m) => m.type === "error");
        if (errors.length)
          throw new Error(errors.map((e) => e.message).join("\n"));
        const layout = this.device.createBindGroupLayout({
          entries: Array.from({ length: count }, (_, binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: binding === count - 1 ? "storage" : "read-only-storage",
            },
          })),
        });
        const pipeline = await this.device.createComputePipelineAsync({
          layout: this.device.createPipelineLayout({
            bindGroupLayouts: [layout],
          }),
          compute: { module, entryPoint: "main" },
        });
        return { pipeline, layout };
      })();
      this.pipelines.set(code, promise);
    }
    return promise;
  }
  async dispatch(code, inputs, rows, cols, groups) {
    this.check();
    const y = { buffer: this.buffer(rows * cols * 4, true), rows, cols };
    const buffers = [...inputs, y.buffer],
      { pipeline, layout } = await this.pipeline(code, buffers.length);
    const bind = this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    if (
      groups.some(
        (x) => x > this.device.limits.maxComputeWorkgroupsPerDimension,
      )
    )
      throw new RangeError("Dispatch exceeds WebGPU workgroup limits");
    pass.dispatchWorkgroups(...groups);
    pass.end();
    return y;
  }
  groups(n, size = 256) {
    const nwork = Math.ceil(n / size);
    return [Math.min(nwork, 65535), Math.ceil(nwork / 65535)];
  }
  element(name, a, b, c, rows, cols, extra = []) {
    return this.dispatch(
      S.elements[name],
      [
        a?.buffer ?? this.zero,
        b?.buffer ?? this.zero,
        c?.buffer ?? this.zero,
        this.params([rows * cols, cols, ...extra]),
      ],
      rows,
      cols,
      this.groups(rows * cols),
    );
  }
  linear(a, name) {
    const w = this.weights[name + ".weight"];
    if (!w || w.cols !== a.cols)
      throw new Error(`Linear shape mismatch ${name}`);
    const bias = this.weights[name + ".bias"];
    return this.dispatch(
      S.matmul(w.dtype, !!bias, w.dtype === "F16" && this.useF16),
      [
        a.buffer,
        w.buffer,
        bias?.buffer ?? this.zero,
        w.scales,
        this.params([a.rows, w.rows, w.cols]),
      ],
      a.rows,
      w.rows,
      [Math.ceil(w.rows / 16), Math.ceil(a.rows / 16)],
    );
  }
  norm(a, name, opts = {}) {
    const width = opts.width ?? a.cols,
      w = this.weights[name + ".weight"],
      b = this.weights[name + ".bias"];
    return this.dispatch(
      S.norm,
      [
        a.buffer,
        w?.buffer ?? this.zero,
        b?.buffer ?? this.zero,
        this.params([
          width,
          opts.stride ?? a.cols,
          opts.offset ?? 0,
          w ? 1 : 0,
          a.rows,
        ]),
      ],
      a.rows,
      width,
      [Math.min(a.rows, 65535), Math.ceil(a.rows / 65535)],
    );
  }
  time(t) {
    return this.element("time", { buffer: this.freq }, null, null, 1, 256, [
      bits(t),
    ]);
  }
  position(L) {
    return this.element(
      "position",
      { buffer: this.posfreq },
      null,
      null,
      L,
      132,
    );
  }
  async condition(time, ids) {
    const out = this.tensor(1024 * 4, 1, 1024);
    this.encoder.copyBufferToBuffer(time.buffer, 0, out.buffer, 0, 1024);
    for (let k = 0; k < 3; k++) {
      const w =
          this.weights[
            `cond_factory.feat_creators.1.embedding_${"CAT"[k]}.weight`
          ],
        g = await this.dispatch(
          S.gather(w.dtype, w.dtype === "F16" && this.useF16),
          [this.zero, w.buffer, w.scales, this.params([256, 0, ids[k]])],
          1,
          256,
          [1],
        );
      this.encoder.copyBufferToBuffer(
        g.buffer,
        0,
        out.buffer,
        (256 + 256 * k) * 4,
        1024,
      );
    }
    return out;
  }
  padCondition(a) {
    return this.element("padCondition", a, null, null, 2, 512);
  }
  registers(a) {
    const w = this.weights.registers;
    return this.dispatch(
      S.gather(w.dtype, w.dtype === "F16" && this.useF16),
      [a.buffer, w.buffer, w.scales, this.params([(a.rows + 10) * 768, 1, 0])],
      a.rows + 10,
      768,
      this.groups((a.rows + 10) * 768),
    );
  }
  pairFeatures(x) {
    return this.dispatch(
      S.pairFeatures,
      [x.buffer, this.bounds, this.params([x.rows])],
      x.rows * x.rows,
      319,
      this.groups(x.rows * x.rows * 319),
    );
  }
  padPair(a, L) {
    return this.element("padPair", a, null, null, (L + 10) ** 2, 512, [L + 10]);
  }
  add(a, b) {
    return this.element("add", a, b, null, a.rows, a.cols);
  }
  swiglu(a) {
    return this.element("swiglu", a, null, null, a.rows, a.cols / 2);
  }
  async adaptive(x, g, b, mode) {
    const scaled = await this.element(
      "adaptiveProduct",
      x,
      g,
      null,
      x.rows,
      x.cols,
      [mode],
    );
    return this.element("adaptiveBias", scaled, b, null, x.rows, x.cols, [
      mode,
    ]);
  }
  async residualGate(x, h, g) {
    const scaled = await this.element(
      "residualProduct",
      h,
      g,
      null,
      x.rows,
      x.cols,
    );
    return this.add(x, scaled);
  }
  unpad(a, n) {
    return this.element("unpad", a, null, null, a.rows - n, a.cols, [n]);
  }
  async velocity(x, v, t) {
    const delta = await this.element(
      "velocityProduct",
      v,
      null,
      null,
      x.rows,
      3,
      [bits(t)],
    );
    return this.add(x, delta);
  }
  async attention(q, k, qkv, bias, g, N) {
    const scores = await this.dispatch(
      S.scores,
      [q.buffer, k.buffer, bias.buffer, this.params([N])],
      12 * N,
      N,
      this.groups(12 * N * N, 64),
    );
    const p = await this.dispatch(
      S.softmax,
      [scores.buffer, this.params([N])],
      12 * N,
      N,
      [12 * N],
    );
    return this.dispatch(
      S.context,
      [p.buffer, qkv.buffer, g.buffer, this.params([N])],
      N,
      768,
      this.groups(N * 768),
    );
  }
  async center(x) {
    const mean = await this.dispatch(
      S.coordinateMean,
      [x.buffer, this.params([x.rows])],
      1,
      3,
      [1],
    );
    return this.element("center", x, mean, null, x.rows, 3);
  }
  scale(x, v) {
    return this.element("scale", x, null, null, x.rows, x.cols, [bits(v)]);
  }
  async mix(prev, noise, t, scale) {
    const z = await this.center(
      await this.scale(await this.center(noise), scale),
    );
    const p = await this.center(prev);
    return this.add(await this.scale(z, f(1 - t)), await this.scale(p, t));
  }
  begin(signal) {
    this.check();
    if (this.busy) throw new Error("Engine busy");
    this.busy = true;
    this.signal = signal;
    this.temps = new Set();
    this.live = new Set();
    this.pool = new Map();
    this.parameters = new Map();
    this.encoder = this.device.createCommandEncoder();
    this.device.pushErrorScope("validation");
  }
  async read(a) {
    this.check();
    const size = a.rows * a.cols * 4,
      readback = this.device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    this.buffers.add(readback);
    this.temps.add(readback);
    this.encoder.copyBufferToBuffer(a.buffer, 0, readback, 0, size);
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = this.device.createCommandEncoder();
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
  async end() {
    let validation;
    try {
      validation = await this.device.popErrorScope();
    } finally {
      for (const b of this.temps) {
        b.destroy();
        this.buffers.delete(b);
      }
      this.temps.clear();
      this.live.clear();
      this.pool.clear();
      this.parameters.clear();
      this.encoder = null;
      this.busy = false;
      this.signal = null;
    }
    if (validation) throw new Error("WebGPU validation: " + validation.message);
  }
  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
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
  if (!["fp32", "fp16", "int8"].includes(precision))
    throw new RangeError("Unknown precision");
  signal?.throwIfAborted();
  const meta = await acquireDevice({ device, useF16, allowSoftwareFallback }),
    b = new GpuBackend(meta, { useF16 });
  try {
    const bundle = await loadWeights(
      baseUrl ?? new URL(`../weights/${precision}`, import.meta.url).href,
      { cache, signal, onProgress, read, onTensor: (n, t) => b.upload(n, t) },
    );
    assertModelManifest(bundle.manifest);
    if (bundle.manifest.precision !== precision)
      throw new Error("Precision mismatch");
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
      if (mixNoise)
        x = await b.mix(
          x,
          b.tensor(mixNoise.noise, x.rows, 3),
          t,
          mixNoise.scale,
        );
      if (mixNoise && opts.onTrace)
        await opts.onTrace("sampler.input", await b.read(x));
      const out = await forwardNetwork(
        b,
        x,
        t,
        cathIndices(opts),
        opts.onTrace ? async (k, v) => opts.onTrace(k, await b.read(v)) : null,
      );
      const data = await b.read(out);
      return data;
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
    if (!Number.isInteger(length) || length < 1)
      throw new RangeError("Invalid length");
    if (
      !Number.isFinite(noiseScale) ||
      noiseScale < 0 ||
      !Number.isFinite(onestepGamma) ||
      onestepGamma < 0
    )
      throw new RangeError("Invalid noise scale");
    cathIndices(opts);
    const times = schedule(steps, tInit),
      noise = opts.noise ?? gaussianNoise(length, steps, seed);
    if (
      !(noise instanceof Float32Array) ||
      noise.length !== length * 3 * steps ||
      noise.some((v) => !Number.isFinite(v))
    )
      throw new RangeError("Invalid noise draws");
    let x = new Float32Array(length * 3);
    const trajectory = [];
    sampling = true;
    try {
      for (let i = 0; i < steps; i++) {
        signal?.throwIfAborted();
        x = await run(x, times[i], opts, {
          noise: noise.subarray(i * length * 3, (i + 1) * length * 3),
          scale: steps === 1 ? onestepGamma : noiseScale,
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
