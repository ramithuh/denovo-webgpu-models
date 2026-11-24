import * as S from "./shaders.js";

function fold2(groups) {
  return groups <= 65535 ? [Math.max(1, groups)] : [65535, Math.ceil(groups / 65535)];
}

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
    this.pass = null;
    this.dispatchCount = 0;
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
    this.engine.ensureUniformCapacity(512);
  }
  _endPass() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }
  tensor(rows, cols) {
    const bytes = rows * cols * 4,
      b = this.engine.scratch(bytes);
    const t = { rows, cols, buffer: b, bytes };
    this.active.add(t);
    return t;
  }
  release(t) {
    if (!t || !this.active.has(t) || this.pinned.has(t)) return;
    this.active.delete(t);
    this.engine.recycle(t.buffer);
  }
  retain(keep) {
    const set = new Set();
    for (const k of keep) {
      if (!k) continue;
      set.add(k);
      if (k.base) set.add(k.base);
      if (k._unit) set.add(k._unit);
    }
    for (const t of this.active) if (!set.has(t)) this.release(t);
  }
  scope(fn) {
    const before = new Set(this.active);
    const out = fn();
    const keepBase = out?.base ?? null;
    const keepUnit = out?._unit ?? null;
    for (const t of this.active)
      if (!before.has(t) && t !== out && t !== keepBase && t !== keepUnit)
        this.release(t);
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
          type: source.includes(`@binding(${i}) var<storage,read_write`)
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
      this.engine.hasUnvalidatedPipelines = true;
    }
    if (groups.length === 1 && groups[0] > 65535)
      groups = [65535, Math.ceil(groups[0] / 65535)];
    if (
      groups.some(
        (n) => n > this.device.limits.maxComputeWorkgroupsPerDimension,
      )
    )
      throw RangeError("Dispatch exceeds device limits");

    const slot = this.dispatchCount++;
    this.engine.ensureUniformCapacity(slot + 1);
    const baseWord = slot * 64;
    const u = this.engine.uniformU32,
      f = this.engine.uniformF32;
    u[baseWord] = values.n ?? 0;
    u[baseWord + 1] = values.c ?? 0;
    u[baseWord + 2] = values.k ?? 0;
    u[baseWord + 3] = values.m ?? 0;
    u[baseWord + 4] = values.a ?? 0;
    u[baseWord + 5] = values.b ?? 0;
    u[baseWord + 6] = values.d ?? 0;
    u[baseWord + 7] = values.e ?? 0;
    f[baseWord + 8] = values.t ?? 0;
    f[baseWord + 9] = values.s ?? 0;
    f[baseWord + 10] = values.z ?? 0;
    f[baseWord + 11] = values.w ?? 0;

    const entries = buffers.map((b, i) => ({
      binding: i,
      resource: { buffer: b.buffer ?? b },
    }));
    entries.push({
      binding: buffers.length,
      resource: {
        buffer: this.engine.uniformBuffer,
        offset: slot * 256,
        size: 48,
      },
    });
    const bind = d.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    if (!this.pass) this.pass = this.encoder.beginComputePass();
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, bind);
    this.pass.dispatchWorkgroups(...groups);
  }
  linear(x, name) {
    if (x.lazyPairNorm) {
      const w = this.weight(name + ".weight"),
        [C, K] = w.shape,
        b = this.engine.weights[name + ".bias"],
        dtype = w.dtype,
        half = this.engine.useF16;
      if (C === 12 && K === 512) {
        const y = this.tensor(x.rows, 12);
        this.dispatch(
          "pairBias-" + dtype + "-" + half,
          S.pairBias(dtype, half),
          [
            x.unit,
            x.normW,
            x.normB,
            w,
            b ?? this.engine.zero,
            w.scale ?? this.engine.zero,
            y,
          ],
          { n: x.rows, c: 12, k: 512, a: b ? 1 : 0 },
          fold2(Math.ceil(x.rows / 32)),
        );
        return y;
      }
      x = this._materializeLazyNorm(x);
    }
    if (x.viewSlice) x = this._materializeSlice(x);
    if (x.lazyUnitNorm) x = this._rawNorm(x.base, null);
    const w = this.weight(name + ".weight"),
      [C, K] = w.shape;
    if (K !== x.cols) throw Error("Linear shape mismatch " + name);
    const y = this.tensor(x.rows, C),
      dtype = w.dtype,
      half = this.engine.useF16;
    if (x.cond2) {
      y.cond2 = true;
      y.logicalRows = x.logicalRows;
    }
    if (x.sparse) {
      this.dispatch(
        "sparse-" + dtype + "-" + half,
        S.sparseLinear(dtype, half),
        [x, w, w.scale ?? this.engine.zero, y],
        { n: x.rows, c: C, k: K },
        fold2(Math.ceil((x.rows * C) / 256)),
      );
    } else {
      const b = this.engine.weights[name + ".bias"];
      const bufs = [x, w, b ?? this.engine.zero, w.scale ?? this.engine.zero, y];
      const vals = { n: x.rows, c: C, k: K, a: b ? 1 : 0 };
      if (x.rows <= 160) {
        this.dispatch(
          "linear32x64-" + dtype + "-" + half,
          S.linear32x64(dtype, half),
          bufs,
          vals,
          fold2(Math.ceil(x.rows / 32) * Math.ceil(C / 64)),
        );
      } else {
        this.dispatch(
          "linear64x128-" + dtype + "-" + half,
          S.linear64x128(dtype, half),
          bufs,
          vals,
          fold2(Math.ceil(x.rows / 64) * Math.ceil(C / 128)),
        );
      }
    }
    return y;
  }
  _rawNorm(x, name, cols = x.cols, stride = 0, offset = 0) {
    const y = this.tensor(x.rows, cols);
    if (x.cond2) {
      y.cond2 = true;
      y.logicalRows = x.logicalRows;
    }
    this.dispatch(
      "norm",
      S.norm,
      [
        x,
        name ? this.weight(name + ".weight") : this.engine.zero,
        name ? this.weight(name + ".bias") : this.engine.zero,
        y,
      ],
      { n: x.rows, c: cols, a: name ? 1 : 0, b: stride, d: offset },
      fold2(x.rows),
    );
    return y;
  }
  _materializeLazyNorm(x) {
    const y = this.tensor(x.rows, x.cols);
    this.dispatch(
      "element-pair-affine",
      S.element,
      [x.unit, x.normW, x.normB, y],
      { n: x.rows, c: x.cols, a: 1, m: x.cols },
      fold2(Math.ceil((x.rows * x.cols) / 256)),
    );
    return y;
  }
  _materializeSlice(s) {
    return this.rearrange(s.base, null, s.rows, s.cols, 0, s.rowStride, s.colOffset);
  }
  norm(x, name) {
    if (x.viewSlice) {
      return this._rawNorm(x.base, name, x.cols, x.rowStride, x.colOffset);
    }
    if (!name && !x.isStaticPair && !x.cond2) {
      return { lazyUnitNorm: true, base: x, rows: x.rows, cols: x.cols };
    }
    if (x.isStaticPair && name && x.cols === 512) {
      if (!x._unit) {
        x._unit = this._rawNorm(x, null);
        this.pinned.add(x._unit);
      }
      return {
        lazyPairNorm: true,
        base: x,
        unit: x._unit,
        normW: this.weight(name + ".weight"),
        normB: this.weight(name + ".bias"),
        rows: x.rows,
        cols: 512,
      };
    }
    return this._rawNorm(x, name);
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
      fold2(Math.ceil((x.rows * cols) / 256)),
    );
    return y;
  }
  add(x, b) {
    return this.element(x, b, null, 0);
  }
  adapt(x, g, b) {
    if (x.lazyUnitNorm) {
      const src = x.base;
      const y = this.tensor(src.rows, src.cols);
      this.dispatch(
        "normAdapt",
        S.normAdapt,
        [src, g, b, y],
        {
          n: src.rows,
          c: src.cols,
          a: g.cond2 ? 1 : 0,
          m: g.rows * g.cols,
        },
        fold2(src.rows),
      );
      return y;
    }
    if (g.cond2) return this.element(x, g, b, 4);
    return this.element(x, g, b, 1);
  }
  scaledAdd(x, d, g) {
    if (g.cond2) return this.element(x, d, g, 5);
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
      fold2(Math.ceil((rows * cols) / 256)),
    );
    return y;
  }
  slice(x, start, width) {
    if ((start & 3) === 0 && (width & 3) === 0 && (x.cols & 3) === 0) {
      return {
        viewSlice: true,
        base: x,
        buffer: x.buffer,
        rows: x.rows,
        cols: width,
        rowStride: x.cols,
        colOffset: start,
      };
    }
    return this.rearrange(x, null, x.rows, width, 0, x.cols, start);
  }
  concat(a, b) {
    return this.rearrange(a, b, 1, a.cols + b.cols, 1, a.cols);
  }
  extendCond(x, L) {
    const y = this.rearrange(x, null, 2, 512, 6);
    y.cond2 = true;
    y.logicalRows = L + 10;
    return y;
  }
  extendSequence(x, L) {
    return this.rearrange(x, this.weight("registers"), L + 10, 768, 3);
  }
  extendPair(x, L) {
    const y = this.rearrange(x, null, (L + 10) ** 2, 512, 4, L + 10);
    y.isStaticPair = true;
    return y;
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
      fold2(Math.ceil((input.length * 132) / 256)),
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
      fold2(Math.ceil(input.length ** 2 / 256)),
    );
    y.cols = 319;
    y.sparse = true;
    return y;
  }
  gatedAttention(q, k, v, g, bias, N) {
    return this.scope(() => {
      const y = this.tensor(N, 768);
      const vBuf = v.viewSlice ? v.base : v;
      const vStrideVec = v.viewSlice ? v.rowStride / 4 : 192;
      const vOffVec = v.viewSlice ? v.colOffset / 4 : 0;
      this.dispatch(
        "gatedAttention",
        S.gatedAttention,
        [q, k, vBuf, g, bias, y],
        { n: N, a: vStrideVec, b: vOffVec },
        [Math.ceil(N / 64), 12],
      );
      return y;
    });
  }
  trace(name, t) {
    if (t.viewSlice) t = this._materializeSlice(t);
    if (t.lazyUnitNorm) t = this._rawNorm(t.base, null);
    if (t.lazyPairNorm) t = this._materializeLazyNorm(t);
    this._endPass();
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
    this._endPass();
    if (this.dispatchCount > 0) {
      this.device.queue.writeBuffer(
        this.engine.uniformBuffer,
        0,
        this.engine.uniformData,
        0,
        this.dispatchCount * 256,
      );
    }
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
    this._endPass();
    this.pinned.clear();
    for (const t of this.active) this.release(t);
    for (const b of this.transients) this.engine.free(b);
    this.transients = [];
  }
}
