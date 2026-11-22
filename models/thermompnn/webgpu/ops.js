import * as S from "./shaders.js";
import { checkAbort } from "../shared/loader.js";
export class GPUOps {
  constructor(engine, input, options) {
    this.engine = engine;
    this.device = engine.device;
    this.input = input;
    this.options = options;
    this.buffers = [];
    this.pool = [];
    this.traces = {};
    this.encoder = this.device.createCommandEncoder();
    this.in = {};
    try {
      for (const name of ["X", "S", "mask", "chain", "residue_idx"])
        this.in[name] = this.upload(input[name]);
    } catch (error) {
      this.dispose();
      throw error;
    }
  }
  alloc(rows, cols = 1) {
    const size = rows * cols * 4;
    this.engine.admit(size);
    const index = this.pool.findIndex((t) => t.buffer.size >= size);
    let buffer;
    if (index >= 0) buffer = this.pool.splice(index, 1)[0].buffer;
    else {
      buffer = this.engine.buffer(
        size,
        GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
        "transient",
      );
      this.buffers.push(buffer);
    }
    return { buffer, rows, cols, size, freed: false };
  }
  upload(data) {
    const t = this.alloc(data.length);
    this.device.queue.writeBuffer(t.buffer, 0, data);
    return t;
  }
  zeros(rows, cols) {
    const t = this.alloc(rows, cols);
    this.encoder.clearBuffer(t.buffer, 0, t.size);
    return t;
  }
  free(t) {
    if (t && !t.freed) {
      t.freed = true;
      this.pool.push(t);
    }
  }
  dispatch(code, bindings, x, y = 1) {
    if (x > this.device.limits.maxComputeWorkgroupsPerDimension) {
      if (y !== 1 || !code.includes("@workgroup_size(64)"))
        throw new Error("Compute dispatch exceeds device limits");
      const width = this.device.limits.maxComputeWorkgroupsPerDimension;
      code = code.replaceAll("id.x", `(id.x+id.y*${width * 64}u)`);
      y = Math.ceil(x / width);
      x = width;
    }
    const p = this.engine.pipeline(code);
    const entries = bindings.map((t, binding) => ({
      binding,
      resource: { buffer: t.buffer, size: t.size },
    }));
    const group = this.device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries,
    });
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }
  trace(name, t) {
    if (!this.options.returnIntermediates) return;
    const out = this.alloc(t.rows, t.cols);
    this.encoder.copyBufferToBuffer(t.buffer, 0, out.buffer, 0, t.size);
    this.traces[name] = out;
  }
  async checkpoint(phase) {
    this.options.onProgress?.({ phase });
    checkAbort(this.options.signal);
    this.engine.ensureAlive();
    // Yield so UI cancellation can interrupt encoding; scratch buffers remain owned until finally.
    await new Promise((resolve) => setTimeout(resolve, 0));
    checkAbort(this.options.signal);
    this.engine.ensureAlive();
  }
  geometry(input) {
    const L = input.length,
      K = Math.min(48, L),
      xyz = this.alloc(L, 15),
      dist = this.alloc(L, L),
      indices = this.alloc(L, K),
      geometry = this.alloc(L * K, 416);
    this.dispatch(S.xyzShader(), [this.in.X, xyz], Math.ceil(L / 64));
    this.dispatch(
      S.distanceShader(L),
      [xyz, this.in.mask, dist],
      Math.ceil(L / 64),
    );
    this.dispatch(S.topkShader(L, K), [dist, indices], L);
    this.dispatch(
      S.geometryShader(L, K),
      [
        xyz,
        dist,
        indices,
        this.in.residue_idx,
        this.in.chain,
        this.engine.w["prot_mpnn.features.embeddings.linear.weight"],
        this.engine.w["prot_mpnn.features.embeddings.linear.bias"],
        geometry,
      ],
      Math.ceil((L * K * 416) / 64),
    );
    this.free(xyz);
    this.free(dist);
    return { geometry, indices };
  }
  dense(x, name, activation = "none", preRelu = false) {
    const w = this.engine.w[name + ".weight"],
      O = w.shape[0],
      I = x.cols,
      out = this.alloc(x.rows, O),
      bias = this.engine.w[name + ".bias"] ?? this.engine.zero(O),
      scale = w.scale ?? this.engine.zero(1);
    if (w.shape[1] !== I) throw new Error(`Weight shape mismatch ${name}`);
    this.dispatch(
      S.denseShader(
        I,
        O,
        w.dtype,
        activation === "gelu",
        preRelu,
        this.engine.useF16,
      ),
      [x, w, bias, scale, out],
      Math.ceil(O / 8),
      Math.ceil(x.rows / 8),
    );
    return out;
  }
  norm(x, delta, name, input) {
    const out = this.alloc(x.rows, x.cols);
    this.dispatch(
      S.normShader(!!delta, !!input),
      [
        x,
        delta ?? x,
        this.engine.w[name + ".weight"],
        this.engine.w[name + ".bias"],
        this.in.mask,
        out,
      ],
      x.rows,
    );
    return out;
  }
  concat(nodes, edges, indices, embedding, input, decoder) {
    const L = input.length,
      K = indices.cols,
      C = decoder ? 512 : 384,
      out = this.alloc(L * K, C);
    this.dispatch(
      S.concatShader(L, K, decoder),
      [nodes, edges, indices, embedding ?? nodes, this.in.mask, out],
      Math.ceil((out.rows * C) / 64),
    );
    return out;
  }
  reduce(msg, indices, input, encoder) {
    const out = this.alloc(input.length, 128);
    this.dispatch(
      S.reduceShader(input.length, indices.cols, encoder),
      [msg, indices, this.in.mask, out],
      Math.ceil((out.rows * 128) / 64),
    );
    return out;
  }
  embed(input, name) {
    const out = this.alloc(input.length, 128);
    this.dispatch(
      S.embedShader(),
      [this.in.S, this.engine.w[name], out],
      Math.ceil((out.rows * 128) / 64),
    );
    return out;
  }
  join(ts) {
    const out = this.alloc(ts[0].rows, 384);
    this.dispatch(
      S.joinShader(),
      [...ts, out],
      Math.ceil((out.rows * 384) / 64),
    );
    return out;
  }
  calibrate(x) {
    const out = this.dense({ ...x, rows: x.rows * x.cols, cols: 1 }, "ddg_out");
    out.rows = x.rows;
    out.cols = x.cols;
    return out;
  }
  subtract(scores, input) {
    const out = this.alloc(input.length, 20);
    this.dispatch(
      S.subtractShader(),
      [scores, this.in.S, out],
      Math.ceil((out.rows * 20) / 64),
    );
    return out;
  }
  async readAll(ddg) {
    const entries = [["output", ddg], ...Object.entries(this.traces)];
    const total = entries.reduce((s, [, t]) => s + t.size, 0);
    this.engine.admit(total, false);
    const staging = this.engine.buffer(
      total,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      "transient",
    );
    this.buffers.push(staging);
    let offset = 0;
    for (const [, t] of entries) {
      this.encoder.copyBufferToBuffer(t.buffer, 0, staging, offset, t.size);
      offset += t.size;
    }
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    await staging.mapAsync(GPUMapMode.READ);
    try {
      checkAbort(this.options.signal);
      this.engine.ensureAlive();
      const array = staging.getMappedRange(),
        values = {};
      offset = 0;
      for (const [name, t] of entries) {
        const Type = name === "E_idx" ? Int32Array : Float32Array;
        values[name] = new Type(array.slice(offset, offset + t.size));
        offset += t.size;
      }
      return values;
    } finally {
      staging.unmap();
    }
  }
  dispose() {
    for (const b of this.buffers) this.engine.release(b);
    this.buffers = [];
    this.pool = [];
    this.encoder = null;
  }
}
