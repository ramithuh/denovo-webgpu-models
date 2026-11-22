import { tensor, sizeOf } from "../shared/tensor.js";
import { decodeTensor } from "../shared/loader.js";
import { shader } from "./shaders.js";

export class GPUBackend {
  constructor(device) {
    this.device = device;
    this.live = new Set();
    this.pool = new Map();
    this.pipelines = new Map();
    this.encoder = null;
    this.pending = 0;
    this.errors = [];
    this.allocations = 0;
    this.listener = (e) => {
      e.preventDefault();
      this.errors.push(e.error.message);
    };
    device.addEventListener("uncapturederror", this.listener);
  }
  storage(bytes, persistent = false) {
    bytes = Math.max(4, Math.ceil(bytes / 4) * 4);
    if (
      bytes > this.device.limits.maxStorageBufferBindingSize ||
      bytes > this.device.limits.maxBufferSize
    )
      throw Error(`Tensor exceeds GPU limits: ${bytes}`);
    const bucket = this.pool.get(bytes);
    let s = !persistent ? bucket?.pop() : null;
    if (!s) {
      s = {
        buffer: this.device.createBuffer({
          size: bytes,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
        }),
        bytes,
      };
      this.allocations++;
    }
    Object.assign(s, { persistent, refs: 0, pooled: false });
    this.live.add(s);
    return s;
  }
  upload(data, shape, persistent = true) {
    // Uploads must not overwrite pooled storage referenced by unsubmitted commands.
    const s = this.storage(data.byteLength, true);
    s.persistent = persistent;
    if (data.byteLength) this.device.queue.writeBuffer(s.buffer, 0, data);
    return tensor(s, shape);
  }
  async uploadPacked(t, scale) {
    if (t.dtype !== "F16" && !(t.dtype === "I8" && scale))
      return this.upload(decodeTensor(t), t.shape);
    const bytes = new Uint8Array(Math.ceil(t.bytes.byteLength / 4) * 4);
    bytes.set(t.bytes);
    const source = this.storage(bytes.byteLength, true);
    this.device.queue.writeBuffer(source.buffer, 0, bytes);
    const scales = scale ? this.upload(decodeTensor(scale), scale.shape) : null;
    source.encoding = t.dtype;
    source.columns = t.shape[1];
    source.scale = scales?.storage;
    return tensor(source, t.shape);
  }

  retain(t) {
    t.storage.refs++;
  }
  release(t) {
    const s = t.storage;
    if (--s.refs <= 0 && !s.persistent && !s.pooled) {
      s.pooled = true;
      this.live.delete(s);
      if (!this.pool.has(s.bytes)) this.pool.set(s.bytes, []);
      this.pool.get(s.bytes).push(s);
    }
  }
  compute(kind, inputs, shapes, attrs = {}, inplace = false) {
    const outputs = shapes.map((shape, i) =>
      inplace && i === 0
        ? attrs.target
        : tensor(this.storage(sizeOf(shape) * 4), shape),
    );
    if (kind === "cat" || kind === "stack") {
      const out = outputs[0],
        d = attrs.dim;
      let offset = 0;
      for (const input of inputs) {
        const stride =
          kind === "stack" ? out.stride.filter((_, i) => i !== d) : out.stride;
        const target = tensor(
          out.storage,
          input.shape,
          stride,
          out.offset + offset * out.stride[d],
        );
        this.compute("copy", [input], [input.shape], { target }, true);
        offset += kind === "stack" ? 1 : input.shape[d];
      }
      return out;
    }
    const built = shader(kind, inputs, outputs, attrs);
    this.lastShader = built.code;
    let pipeline = this.pipelines.get(built.code);
    if (!pipeline) {
      pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({
            label: kind,
            code: built.code,
          }),
          entryPoint: "main",
        },
      });
      if (this.pipelines.size >= 1024)
        this.pipelines.delete(this.pipelines.keys().next().value);
      this.pipelines.set(built.code, pipeline);
    }
    const group = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: built.bindings.map((b) => ({
        binding: b.binding,
        resource: { buffer: b.storage.buffer },
      })),
    });
    this.encoder ??= this.device.createCommandEncoder();
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...built.dispatch);
    pass.end();
    this.pending++;
    return outputs.length === 1 ? outputs[0] : outputs;
  }
  async flush(wait = false) {
    if (this.encoder) {
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder = null;
      this.pending = 0;
    }
    await this.device.queue.onSubmittedWorkDone();
    if (this.errors.length) throw Error(this.errors.splice(0).join("\n"));
  }
  async read(t) {
    let src = t;
    // Materialize views on GPU before mapping.
    const copy = this.compute("copy", [t], [t.shape]);
    this.retain(copy);
    src = copy;
    const size = Math.max(4, sizeOf(t.shape) * 4),
      buffer = this.device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    try {
      this.encoder.copyBufferToBuffer(src.storage.buffer, 0, buffer, 0, size);
      await this.flush();
      await buffer.mapAsync(GPUMapMode.READ);
      return new Float32Array(
        buffer.getMappedRange().slice(0, sizeOf(t.shape) * 4),
      );
    } finally {
      buffer.destroy();
      this.release(copy);
    }
  }
  clearTransient() {
    try {
      if (this.encoder) this.device.queue.submit([this.encoder.finish()]);
    } finally {
      // Even a submission/device error must release every scratch allocation.
      this.encoder = null;
      this.pending = 0;
      for (const s of this.live)
        if (!s.persistent) {
          s.buffer.destroy();
          this.live.delete(s);
        }
      for (const bucket of this.pool.values())
        for (const s of bucket) s.buffer.destroy();
      this.pool.clear();
    }
  }
  memoryUsage() {
    let persistentBytes = 0,
      transientBytes = 0;
    for (const s of this.live) {
      if (s.persistent) persistentBytes += s.bytes;
      else transientBytes += s.bytes;
    }
    for (const bucket of this.pool.values())
      for (const s of bucket) transientBytes += s.bytes;
    return {
      persistentBytes,
      transientBytes,
      totalBytes: persistentBytes + transientBytes,
      pipelines: this.pipelines.size,
      allocations: this.allocations,
    };
  }
  destroy() {
    this.encoder = null;
    for (const s of this.live) s.buffer.destroy();
    for (const bucket of this.pool.values())
      for (const s of bucket) s.buffer.destroy();
    this.live.clear();
    this.pool.clear();
    this.pipelines.clear();
    this.device.removeEventListener("uncapturederror", this.listener);
  }
}
