import { OP_SHADERS, gemmSource } from "./shaders.js";
const floatScratch = new Float32Array(1);
const wordScratch = new Uint32Array(floatScratch.buffer);
const fbits = (x) => {
  floatScratch[0] = x;
  return wordScratch[0];
};
function half(v) {
  const x = fbits(v),
    sign = (x >>> 16) & 0x8000,
    e = ((x >>> 23) & 255) - 127 + 15,
    m = x & 0x7fffff;
  if (e >= 31) return sign | 0x7c00 | (m ? 0x200 : 0);
  if (e <= 0) {
    if (e < -10) return sign;
    const z = (m | 0x800000) >>> (1 - e);
    return sign | ((z + 0xfff + ((z >>> 13) & 1)) >>> 13);
  }
  return sign | ((e << 10) + ((m + 0xfff + ((m >>> 13) & 1)) >>> 13));
}
export class GpuBackend {
  constructor(device, weights, { shaderF16 = false } = {}) {
    this.device = device;
    this.weights = weights;
    this.shaderF16 = shaderF16;
    this.buffers = [];
    this.weightCache = new Map();
    this.pipelines = new Map();
    this.encoder = null;
    this.disposed = false;
    this.zero = this.upload(new Float32Array(4), 1, 4, true);
  }
  static async create(device, weights, opts) {
    const b = new GpuBackend(device, weights, opts);
    try {
      for (const [k, code] of Object.entries({
        ...OP_SHADERS,
        gemm: gemmSource(b.shaderF16),
      })) {
        const module = device.createShaderModule({
          label: `LASErMPNN ${k}`,
          code,
        });
        const info = await module.getCompilationInfo(),
          errors = info.messages.filter((m) => m.type === "error");
        if (errors.length)
          throw Error(`${k}: ${errors.map((e) => e.message).join("\n")}`);
        const entries = Array.from(
          { length: k === "gemm" ? 5 : 6 },
          (_, binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type:
                binding === (k === "gemm" ? 4 : 5)
                  ? "uniform"
                  : binding === (k === "gemm" ? 3 : 4)
                    ? "storage"
                    : "read-only-storage",
            },
          }),
        );
        const bindGroupLayout = device.createBindGroupLayout({ entries });
        const layout = device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout],
        });
        b.pipelines.set(
          k,
          await device.createComputePipelineAsync({
            layout,
            compute: { module, entryPoint: "main" },
          }),
        );
      }
      return b;
    } catch (e) {
      b.destroy();
      throw e;
    }
  }
  upload(data, r, c, persistent = false) {
    const buffer = this.device.createBuffer({
      size: Math.max(4, Math.ceil(data.byteLength / 4) * 4),
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    if (data.byteLength) this.device.queue.writeBuffer(buffer, 0, data);
    const t = { buffer, r, c };
    if (!persistent) this.buffers.push(t);
    return t;
  }
  from(data, r, c) {
    return this.upload(
      data instanceof Float32Array ? data : Float32Array.from(data),
      r,
      c,
    );
  }
  zeros(r, c) {
    return this.upload(new Float32Array(r * c), r, c);
  }
  weight(key) {
    if (this.weightCache.has(key)) return this.weightCache.get(key);
    const w = this.weights[key];
    if (!w) throw Error(`Missing weight ${key}`);
    const t = this.upload(
      Float32Array.from(w.data),
      w.shape[0],
      w.data.length / w.shape[0],
      true,
    );
    this.weightCache.set(key, t);
    return t;
  }
  linear(x, key) {
    const w = this.weights[key + ".weight"],
      bias = this.weights[key + ".bias"]
        ? this.weight(key + ".bias")
        : this.zero;
    let wt;
    if (this.shaderF16) {
      const k = key + "#half";
      wt = this.weightCache.get(k);
      if (!wt) {
        wt = this.upload(
          Uint16Array.from(w.data, half),
          w.shape[0],
          w.shape[1],
          true,
        );
        this.weightCache.set(k, wt);
      }
    } else wt = this.weight(key + ".weight");
    const out = this.zeros(x.r, w.shape[0]);
    if (!x.r) return out;
    this.dispatch(
      "gemm",
      [x, wt, bias, out],
      new Uint32Array([x.r, w.shape[0], x.c, bias === this.zero ? 0 : 1]),
      [Math.ceil(out.c / 16), Math.ceil(out.r / 16)],
    );
    return out;
  }
  dispatch(kind, inputs, params, groups) {
    if (this.disposed) throw Error("Engine destroyed");
    const pipeline = this.pipelines.get(kind),
      uniform = this.device.createBuffer({
        size: params.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.device.queue.writeBuffer(uniform, 0, params);
    this.buffers.push({ buffer: uniform });
    const entries = inputs.map((x, binding) => ({
      binding,
      resource: { buffer: x.buffer },
    }));
    entries.push({ binding: inputs.length, resource: { buffer: uniform } });
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    this.encoder ??= this.device.createCommandEncoder();
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(...groups);
    pass.end();
  }
  op(kind, inputs, r, c, k = 0, d = 0, e = 0) {
    const out = this.zeros(r, c);
    if (!r || !c) return out;
    const args = [...inputs];
    while (args.length < 4) args.push(this.zero);
    args.push(out);
    const groups = kind === "norm" ? r : Math.ceil((r * c) / 128);
    this.dispatch(kind, args, new Uint32Array([r, c, k, d, e, 0, 0, 0]), [
      Math.min(groups, 65535),
      Math.ceil(groups / 65535),
    ]);
    return out;
  }
  act(x, kind) {
    return this.op("act", [x], x.r, x.c, kind === "sigmoid" ? 1 : 0);
  }
  add(x, y) {
    return this.op("add", [x, y], x.r, x.c);
  }
  cat(xs) {
    return xs.reduce((a, b) => this.op("cat", [a, b], a.r, a.c + b.c, a.c));
  }
  gather(x, idx) {
    return this.op(
      "gather",
      [x, this.from(idx, idx.length, 1)],
      idx.length,
      x.c,
    );
  }
  assign(x, row, y) {
    if (!y.r * y.c) return;
    this.encoder ??= this.device.createCommandEncoder();
    this.encoder.copyBufferToBuffer(
      y.buffer,
      0,
      x.buffer,
      row * x.c * 4,
      y.r * y.c * 4,
    );
  }
  norm(x, key) {
    return this.op(
      "norm",
      [x, this.weight(key + ".weight"), this.weight(key + ".bias")],
      x.r,
      x.c,
    );
  }
  vecLinear(x, key) {
    const w = this.weight(key + ".weight");
    return this.op("vecLinear", [x, w], x.r, w.r * 3, w.c);
  }
  vecLength(x) {
    return this.op("vecLength", [x], x.r, x.c / 3);
  }
  vecScale(x, s) {
    return this.op("vecScale", [x, s], x.r, x.c);
  }
  vecNorm(x) {
    return this.op("vecNorm", [x], x.r, x.c);
  }
  rbf(x, key, sigma) {
    const w = this.weight(key);
    return this.op(
      "rbf",
      [x, w],
      x.r,
      x.c * w.r * w.c,
      w.r * w.c,
      fbits(sigma),
    );
  }
  dots(a, b) {
    return this.op("dots", [a, b], a.r, (a.c * b.c) / 9, a.c / 3, b.c / 3);
  }
  atten(raw, key) {
    const w = this.weight(key);
    return this.op("atten", [raw, w], raw.r, w.r, w.c);
  }
  pool(messages, logits, sinks, rows, heads, dim) {
    const lists = Array.from({ length: rows }, () => []);
    sinks.forEach((s, e) => lists[s].push(e));
    const offsets = [0];
    lists.forEach((l) => offsets.push(offsets.at(-1) + l.length));
    return this.op(
      "pool",
      [
        messages,
        logits,
        this.from(lists.flat(), sinks.length, 1),
        this.from(offsets, offsets.length, 1),
      ],
      rows,
      dim,
      messages.c,
      heads,
    );
  }
  rows(xs) {
    const out = this.zeros(
      xs.reduce((n, x) => n + x.r, 0),
      xs[0].c,
    );
    let row = 0;
    for (const x of xs) {
      this.assign(out, row, x);
      row += x.r;
    }
    return out;
  }
  flush() {
    if (this.encoder) {
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder = null;
    }
  }
  async read(x) {
    if (!x.r * x.c) return new Float32Array();
    const size = x.r * x.c * 4,
      staging = this.device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    this.encoder ??= this.device.createCommandEncoder();
    this.encoder.copyBufferToBuffer(x.buffer, 0, staging, 0, size);
    this.flush();
    try {
      await staging.mapAsync(GPUMapMode.READ);
      return new Float32Array(staging.getMappedRange().slice(0));
    } finally {
      staging.destroy();
    }
  }
  mark() {
    return this.buffers.length;
  }
  releaseSince(mark) {
    this.flush();
    for (const t of this.buffers.splice(mark)) t.buffer.destroy();
  }
  destroy() {
    if (this.disposed) return;
    this.flush();
    for (const t of this.buffers) t.buffer.destroy();
    for (const t of this.weightCache.values()) t.buffer.destroy();
    this.zero.buffer.destroy();
    this.buffers = [];
    this.weightCache.clear();
    this.disposed = true;
  }
}
