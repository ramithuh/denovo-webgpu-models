import { gpuRawFeatures } from "./geometry.js";
import {
  linearShader,
  elementShader,
  normShader,
  GELU_BODY,
} from "./shaders.js";
function floatToHalf(x) {
  const b = new ArrayBuffer(4),
    v = new DataView(b);
  v.setFloat32(0, x, true);
  const u = v.getUint32(0, true),
    sign = (u >>> 16) & 0x8000,
    e = ((u >>> 23) & 255) - 127 + 15,
    m = u & 0x7fffff;
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    return sign | ((m | 0x800000) >> (14 - e));
  }
  return sign | (e << 10) | (m >>> 13);
}
export class GpuOps {
  constructor(device, { half = false, useF16 = false } = {}) {
    this.device = device;
    this.half = half;
    this.useF16 = useF16;
    this.buffers = new Set();
    this.weights = new Set();
    this.pinned = new Set();
    this.retired = [];
    this.pipelines = new Map();
    this.modules = new Map();
    this.packed = new Map();
    this.encoder = null;
    this.destroyed = false;
    this.validationErrors = [];
    this.bytes = 0;
    this.peakBytes = 0;
    this.listener = (e) => this.validationErrors.push(e.error.message);
    device.addEventListener("uncapturederror", this.listener);
  }
  tensor(data, shape) {
    if (this.destroyed) throw Error("GPU engine destroyed");
    const size = Math.max(4, shape.reduce((a, b) => a * b, 1) * 4);
    if (size > this.device.limits.maxStorageBufferBindingSize)
      throw Error("Tensor exceeds GPU storage limit");
    const buffer = this.device.createBuffer({
      size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    if (data)
      this.device.queue.writeBuffer(
        buffer,
        0,
        data instanceof Float32Array ? data : Float32Array.from(data),
      );
    const t = { buffer, shape, size, data };
    this.buffers.add(t);
    this.bytes += size;
    this.peakBytes = Math.max(this.peakBytes, this.bytes);
    return t;
  }
  rawFeatures(input) {
    return gpuRawFeatures(this, input);
  }
  zeros(shape) {
    return this.tensor(null, shape);
  }
  keep(t) {
    this.pinned.add(t);
    return t;
  }
  sealWeights() {
    for (const t of this.buffers) this.weights.add(t);
  }
  pipeline(code) {
    let p = this.pipelines.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      this.modules.set(code, module);
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      this.pipelines.set(code, p);
    }
    return p;
  }
  dispatch(code, inputs, out, workgroups) {
    const pipeline = this.pipeline(code),
      entries = [...inputs, out].map((t, binding) => ({
        binding,
        resource: { buffer: t.buffer },
      })),
      bind = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
    this.encoder ??= this.device.createCommandEncoder();
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(...workgroups);
    pass.end();
    return out;
  }
  element(names, inputs, shape, body) {
    const n = shape.reduce((a, b) => a * b, 1),
      out = this.zeros(shape);
    return this.dispatch(
      elementShader([...names, "out"], body, n),
      inputs,
      out,
      [Math.min(65535, Math.ceil(n / 64)), Math.ceil(n / 4194240)],
    );
  }
  linear(x, w, b) {
    const I = w.shape[1],
      O = w.shape[0],
      R = x.shape[0];
    let weight = w;
    if (this.half) {
      weight = this.packed.get(w);
      if (!weight) {
        const packed = new Uint32Array(Math.ceil(w.data.length / 2));
        for (let i = 0; i < w.data.length; i++)
          packed[i >> 1] |= floatToHalf(w.data[i]) << ((i & 1) * 16);
        const buffer = this.device.createBuffer({
          size: packed.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(buffer, 0, packed);
        weight = { buffer, shape: w.shape, size: packed.byteLength };
        this.buffers.add(weight);
        this.weights.add(weight);
        this.bytes += weight.size;
        this.packed.set(w, weight);
      }
    }
    return this.dispatch(
      linearShader(R, I, O, { packed: this.half, f16: this.useF16, bias: !!b }),
      b ? [x, weight, b] : [x, weight],
      this.zeros([R, O]),
      [Math.ceil(O / 16), Math.ceil(R / 16)],
    );
  }
  norm(x, w, b) {
    const [rows, width] = x.shape;
    const out = this.zeros(x.shape);
    if (!rows || !width) return out;
    return this.dispatch(
      normShader(rows, width),
      [x, w, b],
      out,
      [Math.min(rows, 65535), Math.ceil(rows / 65535)],
    );
  }
  gelu(x) {
    return this.element(["x"], [x], x.shape, GELU_BODY);
  }
  add(a, b) {
    return this.element(["a", "b"], [a, b], a.shape, "out[i]=a[i]+b[i];");
  }
  mask(a, m) {
    return this.element(
      ["a", "m"],
      [a, m],
      a.shape,
      `out[i]=a[i]*m[i/${a.shape[1]}u];`,
    );
  }
  gather(a, idx) {
    const D = a.shape.at(-1),
      indices = this.tensor(idx, [idx.length, 1]);
    return this.element(
      ["a", "indices"],
      [a, indices],
      [idx.length, D],
      `out[i]=a[u32(indices[i/${D}u])*${D}u+i%${D}u];`,
    );
  }
  concat(...xs) {
    const R = xs[0].shape[0],
      D = xs.reduce((n, x) => n + x.shape[1], 0);
    let offset = 0,
      body = `let row=i/${D}u;let col=i%${D}u;`;
    xs.forEach((x, j) => {
      body += `${j ? "else " : ""}if(col<${offset + x.shape[1]}u){out[i]=x${j}[row*${x.shape[1]}u+col-${offset}u];}`;
      offset += x.shape[1];
    });
    return this.element(
      xs.map((_, i) => "x" + i),
      xs,
      [R, D],
      body,
    );
  }
  reduce(x, K) {
    const D = x.shape[1],
      R = x.shape[0] / K;
    return this.element(
      ["x"],
      [x],
      [R, D],
      `let row=i/${D}u;let col=i%${D}u;var sum=0.0;for(var k=0u;k<${K}u;k++){sum+=x[(row*${K}u+k)*${D}u+col];}out[i]=sum/30.0;`,
    );
  }
  scatter(target, row, value) {
    this.encoder ??= this.device.createCommandEncoder();
    this.encoder.copyBufferToBuffer(
      value.buffer,
      0,
      target.buffer,
      row * target.shape[1] * 4,
      value.shape.reduce((a, b) => a * b, 1) * 4,
    );
  }
  flush() {
    if (this.encoder) {
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder = null;
    }
    for (const t of this.retired) {
      t.buffer.destroy();
      this.bytes -= t.size;
    }
    this.retired = [];
  }
  async read(t) {
    const size = t.shape.reduce((a, b) => a * b, 1) * 4,
      buffer = this.device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    this.encoder ??= this.device.createCommandEncoder();
    this.encoder.copyBufferToBuffer(t.buffer, 0, buffer, 0, size);
    this.flush();
    try {
      await buffer.mapAsync(GPUMapMode.READ);
      return new Float32Array(buffer.getMappedRange().slice(0));
    } finally {
      buffer.destroy();
    }
  }
  scope(fn) {
    const before = new Set(this.buffers),
      result = fn(),
      keep = new Set();
    const visit = (v) => {
      if (v?.buffer) keep.add(v);
      else if (Array.isArray(v)) v.forEach(visit);
      else if (v && typeof v === "object") Object.values(v).forEach(visit);
    };
    visit(result);
    for (const t of this.buffers)
      if (
        !before.has(t) &&
        !keep.has(t) &&
        !this.pinned.has(t) &&
        !this.weights.has(t)
      ) {
        this.buffers.delete(t);
        this.retired.push(t);
      }
    return result;
  }
  clear() {
    this.flush();
    for (const t of this.buffers)
      if (!this.weights.has(t)) {
        t.buffer.destroy();
        this.bytes -= t.size;
        this.buffers.delete(t);
        this.pinned.delete(t);
      }
  }
  async checkShaders() {
    const errors = [];
    for (const [source, module] of this.modules) {
      const info = await module.getCompilationInfo();
      for (const m of info.messages)
        if (m.type === "error") errors.push(m.message);
    }
    if (errors.length || this.validationErrors.length)
      throw Error([...errors, ...this.validationErrors].join("\n"));
    return this.modules.size;
  }
  destroy() {
    if (this.destroyed) return;
    this.flush();
    for (const t of this.buffers) t.buffer.destroy();
    this.buffers.clear();
    this.weights.clear();
    this.pinned.clear();
    this.packed.clear();
    this.pipelines.clear();
    this.modules.clear();
    this.bytes = 0;
    this.destroyed = true;
    this.device.removeEventListener("uncapturederror", this.listener);
  }
}
