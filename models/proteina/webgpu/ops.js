import * as S from "./shaders.js";
export class GPUOps {
  constructor(engine, input) {
    this.engine = engine;
    this.device = engine.device;
    this.input = input;
    this.active = new Set();
    this.pinned = new Set();
    this.transients = [];
    this.traces = {};
    this.encoder = this.device.createCommandEncoder();
  }
  init() {
    const input = this.input;
    const upload = (data, rows, cols) => {
      const t = this.tensor(rows, cols);
      this.device.queue.writeBuffer(t.buffer, 0, data);
      this.pinned.add(t);
      return t;
    };
    const L = input.length;
    this.x = upload(input.x, L, 3);
    this.sc = upload(
      Float32Array.from(input.xSC ?? new Float32Array(L * 3)),
      L,
      3,
    );
    this.indices = upload(
      Float32Array.from(
        input.residueIndices ?? Array.from({ length: L }, (_, i) => i + 1),
      ),
      L,
      1,
    );
    this.breaks = upload(
      Float32Array.from(input.chainBreaks ?? new Float32Array(L)),
      L,
      1,
    );
  }
  tensor(rows, cols) {
    const bytes = rows * cols * 4,
      b = this.engine.scratch(bytes);
    const t = { rows, cols, buffer: b, bytes };
    this.active.add(t);
    return t;
  }
  release(t) {
    if (!this.active.has(t) || this.pinned.has(t)) return;
    this.active.delete(t);
    this.engine.recycle(t.buffer);
  }
  retain(keep) {
    const set = new Set(keep);
    for (const t of this.active) if (!set.has(t)) this.release(t);
  }
  scope(fn) {
    const before = new Set(this.active);
    const out = fn();
    for (const t of this.active)
      if (!before.has(t) && t !== out) this.release(t);
    return out;
  }
  weight(n) {
    const w = this.engine.weights[n];
    if (!w) throw Error("Missing weight " + n);
    return w;
  }
  dispatch(key, source, buffers, values, groups) {
    const d = this.device;
    let pipeline = this.engine.pipelines.get(key);
    if (!pipeline) {
      const entries = buffers.map((_, i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: source.includes(`@binding(${i}) var<storage,read_write>`)
            ? "storage"
            : "read-only-storage",
        },
      }));
      entries.push({
        binding: buffers.length,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      });
      const layout = d.createPipelineLayout({
        bindGroupLayouts: [d.createBindGroupLayout({ entries })],
      });
      pipeline = d.createComputePipeline({
        layout,
        compute: {
          module: d.createShaderModule({ label: key, code: source }),
          entryPoint: "main",
        },
      });
      this.engine.pipelines.set(key, pipeline);
    }
    const data = new ArrayBuffer(48),
      u = new Uint32Array(data),
      f = new Float32Array(data);
    const names = ["n", "c", "k", "m", "a", "b", "d", "e"];
    for (let i = 0; i < 8; i++) u[i] = values[names[i]] ?? 0;
    f[8] = values.t ?? 0;
    const uniform = this.engine.allocate(
      48,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      "transient",
    );
    this.transients.push(uniform);
    d.queue.writeBuffer(uniform, 0, data);
    const bind = d.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [...buffers, uniform].map((b, i) => ({
        binding: i,
        resource: { buffer: b.buffer ?? b },
      })),
    });
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    if (groups.length === 1 && groups[0] > 65535)
      groups = [65535, Math.ceil(groups[0] / 65535)];
    if (
      groups.some(
        (n) => n > this.device.limits.maxComputeWorkgroupsPerDimension,
      )
    )
      throw RangeError("Dispatch exceeds device limits");
    pass.dispatchWorkgroups(...groups);
    pass.end();
  }
  linear(x, name) {
    const w = this.weight(name + ".weight"),
      [C, K] = w.shape;
    if (K !== x.cols) throw Error("Linear shape mismatch " + name);
    const y = this.tensor(x.rows, C),
      dtype = w.dtype,
      half = this.engine.useF16;
    if (x.sparse)
      this.dispatch(
        "sparse-" + dtype + "-" + half,
        S.sparseLinear(dtype, half),
        [x, w, w.scale ?? this.engine.zero, y],
        { n: x.rows, c: C, k: K },
        [Math.ceil((x.rows * C) / 256)],
      );
    else {
      const b = this.engine.weights[name + ".bias"];
      this.dispatch(
        "linear-" + dtype + "-" + half,
        S.linear(dtype, half),
        [x, w, b ?? this.engine.zero, w.scale ?? this.engine.zero, y],
        { n: x.rows, c: C, k: K, a: b ? 1 : 0 },
        [Math.ceil(C / 32), Math.ceil(x.rows / 8)],
      );
    }
    return y;
  }
  norm(x, name) {
    const y = this.tensor(x.rows, x.cols);
    this.dispatch(
      "norm",
      S.norm,
      [
        x,
        name ? this.weight(name + ".weight") : this.engine.zero,
        name ? this.weight(name + ".bias") : this.engine.zero,
        y,
      ],
      { n: x.rows, c: x.cols, a: name ? 1 : 0 },
      [x.rows],
    );
    return y;
  }
  element(x, a, b, mode, cols = x.cols) {
    const y = this.tensor(x.rows, cols);
    this.dispatch(
      "element",
      S.element,
      [x, a ?? this.engine.zero, b ?? this.engine.zero, y],
      {
        n: x.rows,
        c: cols,
        a: mode,
        m: mode === 1 ? a.rows * a.cols : mode === 2 ? b.rows * b.cols : 0,
      },
      [Math.ceil((x.rows * cols) / 256)],
    );
    return y;
  }
  add(x, b) {
    return this.element(x, b, null, 0);
  }
  adapt(x, g, b) {
    return this.element(x, g, b, 1);
  }
  scaledAdd(x, d, g) {
    return this.element(x, d, g, 2);
  }
  swiglu(x) {
    return this.element(x, null, null, 3, x.cols / 2);
  }
  rearrange(x, extra, rows, cols, mode, k = 0, b = 0) {
    const y = this.tensor(rows, cols);
    this.dispatch(
      "rearrange",
      S.rearrange,
      [x, extra ?? this.engine.zero, y],
      { n: rows, c: cols, k, a: mode, b },
      [Math.ceil((rows * cols) / 256)],
    );
    return y;
  }
  slice(x, start, width) {
    return this.rearrange(x, null, x.rows, width, 0, x.cols, start);
  }
  concat(a, b) {
    return this.rearrange(a, b, 1, a.cols + b.cols, 1, a.cols);
  }
  extendCond(x, L) {
    return this.rearrange(x, null, L + 10, 512, 2);
  }
  extendSequence(x, L) {
    return this.rearrange(x, this.weight("registers"), L + 10, 768, 3);
  }
  extendPair(x, L) {
    return this.rearrange(x, null, (L + 10) ** 2, 512, 4, L + 10);
  }
  removeRegisters(x, L) {
    return this.rearrange(x, null, L, 768, 5);
  }
  coordinates() {
    return this.x;
  }
  time(t) {
    const y = this.tensor(1, 256);
    this.dispatch("time", S.time, [this.engine.constants.timeFreq, y], { t }, [
      1,
    ]);
    return y;
  }
  fold(ids) {
    const y = this.tensor(1, 768);
    this.dispatch(
      "fold",
      S.fold,
      [
        ...["C", "A", "T"].map((c) =>
          this.weight(`cond_factory.feat_creators.1.embedding_${c}.weight`),
        ),
        y,
      ],
      { a: ids[0], b: ids[1], d: ids[2] },
      [1],
    );
    return y;
  }
  features(input) {
    const y = this.tensor(input.length, 132);
    this.dispatch(
      "features",
      S.features,
      [
        this.sc,
        this.indices,
        this.breaks,
        this.engine.constants.indexDivisor,
        y,
      ],
      { n: input.length },
      [Math.ceil((input.length * 132) / 256)],
    );
    return y;
  }
  pairFeatures(input) {
    const y = this.tensor(input.length ** 2, 3);
    this.dispatch(
      "pairs",
      S.pairs,
      [this.x, this.sc, this.indices, this.engine.constants.bins, y],
      { n: input.length, a: input.xSC ? 1 : 0 },
      [Math.ceil(input.length ** 2 / 256)],
    );
    y.cols = 319;
    y.sparse = true;
    return y;
  }
  gatedAttention(q, k, v, g, bias, N) {
    return this.scope(() => {
      const logits = this.tensor(N * 12, N),
        prob = this.tensor(N * 12, N),
        y = this.tensor(N, 768);
      this.dispatch("logits", S.logits, [q, k, bias, logits], { n: N }, [
        Math.ceil((N * N * 12) / 256),
      ]);
      this.dispatch("softmax", S.softmax, [logits, prob], { n: N * 12, c: N }, [
        N * 12,
      ]);
      this.dispatch("attend", S.attend, [prob, v, g, y], { n: N }, [
        Math.ceil((N * 768) / 256),
      ]);
      return y;
    });
  }
  trace(name, t) {
    const b = this.engine.allocate(
      t.bytes,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      "transient",
    );
    this.transients.push(b);
    this.encoder.copyBufferToBuffer(t.buffer, 0, b, 0, t.bytes);
    this.traces[name] = { buffer: b, bytes: t.bytes };
  }
  async finish(t, signal) {
    const read = this.engine.allocate(
      t.bytes,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      "transient",
    );
    this.transients.push(read);
    this.encoder.copyBufferToBuffer(t.buffer, 0, read, 0, t.bytes);
    signal?.throwIfAborted();
    this.device.queue.submit([this.encoder.finish()]);
    const map = async (b, bytes) => {
      await b.mapAsync(GPUMapMode.READ, 0, bytes);
      try {
        return new Float32Array(b.getMappedRange(0, bytes).slice(0));
      } finally {
        b.unmap();
      }
    };
    const coordinates = await map(read, t.bytes);
    signal?.throwIfAborted();
    const traces = {};
    for (const [name, v] of Object.entries(this.traces))
      traces[name] = await map(v.buffer, v.bytes);
    return { coordinates, trace: traces };
  }
  cleanup() {
    this.pinned.clear();
    for (const t of this.active) this.release(t);
    for (const b of this.transients) this.engine.free(b);
    this.transients = [];
  }
}
