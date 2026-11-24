import { contiguous, tensor, sizeOf } from "../shared/tensor.js";
import { decodeTensor } from "../shared/loader.js";
import { shader } from "./shaders.js";

const UNIFORM_SLAB_BYTES = 262144;
const UNIFORM_SLOT_BYTES = 256;
const SLOTS_PER_SLAB = UNIFORM_SLAB_BYTES / UNIFORM_SLOT_BYTES;

export class GPUBackend {
  constructor(device) {
    this.device = device;
    this.hasF16 = Boolean(device.features?.has?.("shader-f16"));
    this.normSingleOutput = true;
    this.live = new Set();
    this.pool = new Map();
    this.pipelines = new Map();
    this.encoder = null;
    this.pass = null;
    this.pending = 0;
    this.errors = [];
    this.allocations = 0;
    this.uniformSlabs = [];
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
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
  allocUniform(words) {
    if (this.uniformSlotIdx >= SLOTS_PER_SLAB) {
      this.uniformSlabIdx++;
      this.uniformSlotIdx = 0;
    }
    while (this.uniformSlabs.length <= this.uniformSlabIdx) {
      const buffer = this.device.createBuffer({
        size: UNIFORM_SLAB_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.uniformSlabs.push({
        buffer,
        bytes: UNIFORM_SLAB_BYTES,
        cpu: new Uint32Array(UNIFORM_SLAB_BYTES / 4),
        usedWords: 0,
      });
      this.allocations++;
    }
    const slab = this.uniformSlabs[this.uniformSlabIdx];
    const byteOffset = this.uniformSlotIdx * UNIFORM_SLOT_BYTES;
    const wordOffset = byteOffset >>> 2;
    slab.cpu.set(words, wordOffset);
    slab.usedWords = wordOffset + words.length;
    this.uniformSlotIdx++;
    return {
      buffer: slab.buffer,
      offset: byteOffset,
      size: words.byteLength,
    };
  }
  flushUniforms() {
    const limit = Math.min(this.uniformSlabIdx + 1, this.uniformSlabs.length);
    for (let i = 0; i < limit; i++) {
      const slab = this.uniformSlabs[i];
      if (slab.usedWords > 0) {
        this.device.queue.writeBuffer(
          slab.buffer,
          0,
          slab.cpu.buffer,
          0,
          slab.usedWords * 4,
        );
        slab.usedWords = 0;
      }
    }
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
  }
  endPass() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }
  submit() {
    if (!this.encoder) return;
    this.endPass();
    this.flushUniforms();
    const cmd = this.encoder.finish();
    this.encoder = null;
    this.pending = 0;
    this.device.queue.submit([cmd]);
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
    const built = shader(kind, inputs, outputs, {
      ...attrs,
      hasF16: this.hasF16,
    });
    this.lastShader = built.code;
    let entry = this.pipelines.get(built.code);
    if (!entry) {
      const pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({
            label: kind,
            code: built.code,
          }),
          entryPoint: "main",
        },
      });
      entry = { pipeline, layout: pipeline.getBindGroupLayout(0) };
      if (this.pipelines.size >= 8192)
        this.pipelines.delete(this.pipelines.keys().next().value);
      this.pipelines.set(built.code, entry);
    }
    const entries = built.bindings.map((b) => ({
      binding: b.binding,
      resource: { buffer: b.storage.buffer },
    }));
    if (built.uniforms) {
      entries.push({
        binding: built.uniformBinding,
        resource: this.allocUniform(built.uniforms),
      });
    }
    const group = this.device.createBindGroup({
      layout: entry.layout,
      entries,
    });
    this.encoder ??= this.device.createCommandEncoder();
    this.pass ??= this.encoder.beginComputePass();
    this.pass.setPipeline(entry.pipeline);
    this.pass.setBindGroup(0, group);
    this.pass.dispatchWorkgroups(...built.dispatch);
    this.pending++;
    if (this.pending >= 512) this.submit();
    return outputs.length === 1 ? outputs[0] : outputs;
  }
  async flush(wait = false) {
    this.submit();
    if (wait) {
      await this.device.queue.onSubmittedWorkDone();
      if (this.errors.length) throw Error(this.errors.splice(0).join("\n"));
    }
  }
  async readMany(ts) {
    if (!ts.length) return [];
    const copies = ts.map((t) => {
      if (contiguous(t) && t.offset === 0 && !t.storage.encoding) return null;
      const c = this.compute("copy", [t], [t.shape]);
      this.retain(c);
      return c;
    });
    const byteSizes = ts.map((t) => sizeOf(t.shape) * 4);
    const alignedOffsets = [];
    let totalBytes = 0;
    for (const b of byteSizes) {
      alignedOffsets.push(totalBytes);
      totalBytes += Math.max(4, Math.ceil(b / 4) * 4);
    }
    totalBytes = Math.max(4, totalBytes);
    const buffer = this.device.createBuffer({
      size: totalBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    try {
      this.encoder ??= this.device.createCommandEncoder();
      this.endPass();
      for (let i = 0; i < ts.length; i++) {
        const src = copies[i] ?? ts[i];
        if (byteSizes[i] > 0) {
          this.encoder.copyBufferToBuffer(
            src.storage.buffer,
            0,
            buffer,
            alignedOffsets[i],
            byteSizes[i],
          );
        }
      }
      await this.flush(true);
      await buffer.mapAsync(GPUMapMode.READ);
      const mapped = buffer.getMappedRange();
      const out = byteSizes.map(
        (b, i) =>
          new Float32Array(mapped.slice(alignedOffsets[i], alignedOffsets[i] + b)),
      );
      buffer.unmap();
      return out;
    } finally {
      buffer.destroy();
      for (const c of copies) if (c) this.release(c);
    }
  }
  async read(t) {
    return (await this.readMany([t]))[0];
  }
  clearTransient() {
    try {
      if (this.encoder) {
        this.endPass();
        this.flushUniforms();
        this.device.queue.submit([this.encoder.finish()]);
      }
    } finally {
      // Even a submission/device error must release every scratch allocation.
      this.encoder = null;
      this.pass = null;
      this.pending = 0;
      for (const s of this.live)
        if (!s.persistent) {
          s.buffer.destroy();
          this.live.delete(s);
        }
      for (const bucket of this.pool.values())
        for (const s of bucket) s.buffer.destroy();
      this.pool.clear();
      for (const slab of this.uniformSlabs) slab.buffer.destroy();
      this.uniformSlabs.length = 0;
      this.uniformSlabIdx = 0;
      this.uniformSlotIdx = 0;
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
    for (const slab of this.uniformSlabs) transientBytes += slab.bytes;
    return {
      persistentBytes,
      transientBytes,
      totalBytes: persistentBytes + transientBytes,
      pipelines: this.pipelines.size,
      allocations: this.allocations,
    };
  }
  destroy() {
    this.endPass();
    this.encoder = null;
    for (const s of this.live) s.buffer.destroy();
    for (const bucket of this.pool.values())
      for (const s of bucket) s.buffer.destroy();
    for (const slab of this.uniformSlabs) slab.buffer.destroy();
    this.uniformSlabs.length = 0;
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
    this.live.clear();
    this.pool.clear();
    this.pipelines.clear();
    this.device.removeEventListener("uncapturederror", this.listener);
  }
}
