import {
  loadWeights,
  decodeTensor,
  normalizeBackboneCoords,
  normalizeResidueIndex,
  resolvePredictInputs,
  buildStructureFrames,
  buildKnnEdges,
  buildRelativePosIds,
} from '../../shared/dyna1/index.js';
import {
  matmulSource,
  matmulGrid,
  normSource,
  elementSource,
  gatherSource,
  gatherAddSource,
  ropeSource,
  streamingAttn64Source,
  scoresSource,
  softmaxSource,
  contextSource,
  geomRotateSource,
  geomScoresSource,
  geomContextSource,
  selectLocalSource,
  codebookArgminSource,
} from './shaders.js';

const STRUCT_CHUNK_SIZE = 192; // 192 * 16 * 8192 * 4 = 100.66 MB < 128 MB default maxStorageBufferBindingSize

function grid1d(count, wgSize = 256) {
  const wg = Math.ceil(count / wgSize);
  return wg <= 65535 ? [Math.max(1, wg)] : [65535, Math.ceil(wg / 65535)];
}

export async function requestWebGpuDevice({ useF16 = true } = {}) {
  if (!globalThis.navigator?.gpu) throw new Error('WebGPU requires a supported browser and HTTPS or localhost');
  let adapter = null;
  for (let attempt = 0; attempt < 3 && !adapter; attempt++) {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  }
  if (!adapter) adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const hasF16 = adapter.features.has('shader-f16');
  const device = await adapter.requestDevice({ requiredFeatures: useF16 && hasF16 ? ['shader-f16'] : [] });
  return { adapter, device, hasF16 };
}

class Engine {
  constructor(device, { ownsDevice = false, useF16 = true, structureEncoderUrl = null, loadOptions = {} } = {}) {
    this.device = device;
    this.ownsDevice = ownsDevice;
    this.useF16 = useF16 && device.features.has('shader-f16');
    this.structureEncoderUrl = structureEncoderUrl;
    this.loadOptions = loadOptions;
    this.weights = {};
    this.structWeights = null;
    this.structConfig = null;
    this.structManifest = null;
    this._structLoading = null;
    this.buffers = new Set();
    this.pipelines = new Map();
    this.readyPipelines = new Map();
    this.hasNewPipelines = false;
    this.destroyed = false;
    this.busy = false;
    this.lost = null;
    this.pass = null;
    this._uploadedBytes = 0;
    device.lost.then(info => { this.lost = info.message || 'Device lost'; });
    this.zero = this.buffer(new Float32Array(4));
  }
  _endPass() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }
  buffer(dataOrSize, temporary = false) {
    const data = typeof dataOrSize === 'number' ? null : dataOrSize;
    const size = Math.max(4, Math.ceil((data?.byteLength ?? dataOrSize) / 4) * 4);
    if (this.destroyed) throw new Error('Engine destroyed');
    const buffer = (temporary && !data ? this.pool.get(size)?.pop() : null) ??
      this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (data) {
      const bytes = new Uint8Array(size);
      bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      this.device.queue.writeBuffer(buffer, 0, bytes);
    }
    this.buffers.add(buffer);
    if (temporary) { this.temporary.add(buffer); this.liveTemps.add(buffer); }
    return buffer;
  }
  releaseExcept(keepList, traces) {
    const keep = new Set([...(Array.isArray(keepList) ? keepList : [keepList]), ...Object.values(traces).map(t => t.buffer)]);
    for (const b of this.liveTemps) {
      if (!keep.has(b)) {
        if (!this.pool.has(b.size)) this.pool.set(b.size, []);
        this.pool.get(b.size).push(b);
        this.liveTemps.delete(b);
      }
    }
  }
  memoryUsage() {
    return { bufferCount: this.buffers.size, totalBytes: [...this.buffers].reduce((n, b) => n + b.size, 0) };
  }
  async upload(name, t, target = this.weights) {
    if (this.destroyed) throw new Error('Engine destroyed');
    if (name === 'embedding.weight' || name === 'structure_tokens_embed_delta' || name === 'encoder.relative_positional_embedding.embedding.weight' || name === 'encoder.codebook.embeddings') {
      if (t.dtype !== 'F32') t = { dtype: 'F32', shape: t.shape, bytes: new Uint8Array(decodeTensor(t).buffer) };
    }
    target[name] = { buffer: this.buffer(t.bytes), dtype: t.dtype, shape: t.shape, scales: t.scales ? this.buffer(t.scales) : this.zero };
    if (name === 'encoder.codebook.embeddings') {
      const cb = decodeTensor(t);
      const [codes, dim] = t.shape;
      const norms = new Float32Array(codes);
      for (let k = 0; k < codes; k++) { let s = 0; for (let d = 0; d < dim; d++) s += cb[k * dim + d] ** 2; norms[k] = s; }
      target['encoder.codebook.norms'] = { buffer: this.buffer(norms), dtype: 'F32', shape: [codes], scales: this.zero };
    }
    this._uploadedBytes += t.bytes.byteLength;
    if (this._uploadedBytes >= 256 * 1024 * 1024) {
      this._uploadedBytes = 0;
      this.device.queue.submit([]);
      await this.device.queue.onSubmittedWorkDone();
    }
  }
  async preloadStructureEncoder({ onProgress, signal } = {}) {
    if (this.destroyed) throw new Error('Engine destroyed');
    if (this.lost) throw new Error(this.lost);
    if (this.structWeights) return;
    if (!this.structureEncoderUrl) throw new Error('structureEncoderUrl is not configured');
    signal?.throwIfAborted();
    if (!this._structLoading) {
      const staged = {};
      const stagedBuffers = [];
      const uploadStaged = async (name, t) => {
        const before = new Set(this.buffers);
        await this.upload(name, t, staged);
        for (const b of this.buffers) if (!before.has(b)) stagedBuffers.push(b);
      };
      this._structLoading = loadWeights(this.structureEncoderUrl, {
        ...this.loadOptions,
        onProgress: onProgress ?? this.loadOptions.onProgress,
        signal: signal ?? this.loadOptions.signal,
        progressPhase: 'structure-encoder',
        onTensor: uploadStaged,
      }).then(bundle => {
        if (this.destroyed) {
          for (const b of stagedBuffers) { b.destroy(); this.buffers.delete(b); }
          throw new Error('Engine destroyed');
        }
        this.structWeights = staged;
        this.structConfig = bundle.config;
        this.structManifest = bundle.manifest;
      }).catch(err => {
        for (const b of stagedBuffers) { b.destroy(); this.buffers.delete(b); }
        this._structLoading = null;
        throw err;
      });
    }
    await this._structLoading;
  }
  async pipeline(code, count) {
    const ready = this.readyPipelines.get(code);
    if (ready) return ready;
    let promise = this.pipelines.get(code);
    if (!promise) {
      promise = (async () => {
        const module = this.device.createShaderModule({ code });
        const info = await module.getCompilationInfo();
        const errors = info.messages.filter(m => m.type === 'error');
        if (errors.length) throw new Error(errors.map(e => e.message).join('\n'));
        const layout = this.device.createBindGroupLayout({
          entries: Array.from({ length: count }, (_, binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: binding === count - 1 ? 'storage' : 'read-only-storage' },
          })),
        });
        const pipeline = await this.device.createComputePipelineAsync({
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
          compute: { module, entryPoint: 'main' },
        });
        const entry = { pipeline, layout };
        this.readyPipelines.set(code, entry);
        this.hasNewPipelines = true;
        return entry;
      })();
      this.pipelines.set(code, promise);
    }
    return promise;
  }
  _recordDispatch(pipeline, layout, buffers, groups) {
    const bind = this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    if (!this.pass) this.pass = this.encoder.beginComputePass();
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, bind);
    this.pass.dispatchWorkgroups(...groups);
  }
  async dispatch(code, inputs, n, groups) {
    const y = this.buffer(n * 4, true), buffers = [...inputs, y];
    const ready = this.readyPipelines.get(code);
    if (ready) {
      this._recordDispatch(ready.pipeline, ready.layout, buffers, groups);
      return y;
    }
    const { pipeline, layout } = await this.pipeline(code, buffers.length);
    this._recordDispatch(pipeline, layout, buffers, groups);
    return y;
  }
  async _encodeStructureOps(coords, residueIndex, { signal, traces = {}, save = () => {} } = {}) {
    const sw = this.structWeights, cfg = this.structConfig;
    const L = coords.length / 9;
    const K = Math.min(cfg.knn ?? 16, L);
    const D = cfg.d_model ?? 1024, H = cfg.geom_heads ?? 128, F = cfg.ffn_size ?? 4096;
    const layers = cfg.layers ?? 2, codeDim = cfg.code_dim ?? 128, codes = cfg.codebook_size ?? 4096;
    const frames = buildStructureFrames(coords, L);
    const edges = buildKnnEdges(coords, frames, L, K);
    const relIds = buildRelativePosIds(edges, residueIndex, L, K, cfg.relative_position_bins ?? 32);

    const lnS = (x, p, R) => this.dispatch(
      normSource(R, D, { eps: 1e-5, bias: !!sw[p + '.bias'] }),
      [x, sw[p + '.weight'].buffer, sw[p + '.bias']?.buffer ?? this.zero],
      R * D,
      grid1d(R, 1)
    );
    const linS = (x, p, R, Kdim, Ndim) => {
      const weight = sw[p + '.weight'], bias = sw[p + '.bias'];
      return this.dispatch(
        matmulSource(R, Ndim, Kdim, weight.dtype, !!bias, this.useF16),
        [x, weight.buffer, bias?.buffer ?? this.zero, weight.scales],
        R * Ndim,
        matmulGrid(R, Ndim)
      );
    };
    const elemS = (n, exp, inputs) => this.dispatch(elementSource(n, exp, inputs.length), inputs, n, grid1d(n, 256));

    const allSelected = this.buffer(L * D * 4, true);
    const chunkSize = Math.min(L, STRUCT_CHUNK_SIZE);
    for (let b0 = 0; b0 < L; b0 += chunkSize) {
      signal?.throwIfAborted();
      if (this.destroyed) throw new Error('Engine destroyed');
      const B = Math.min(chunkSize, L - b0);
      const R = B * K;
      const chunkKnnFrames = new Float32Array(R * 13);
      for (let i = 0; i < R; i++) {
        const e = edges[b0 * K + i];
        chunkKnnFrames.set(frames.subarray(e * 13, e * 13 + 13), i * 13);
      }
      const chunkNodeFrames = frames.subarray(b0 * 13, (b0 + B) * 13);
      const chunkRelIds = new Uint32Array(relIds.subarray(b0 * K, (b0 + B) * K));

      const knnFramesBuf = this.buffer(chunkKnnFrames, true);
      const nodeFramesBuf = this.buffer(chunkNodeFrames, true);
      const relIdsBuf = this.buffer(chunkRelIds, true);
      let x = await this.dispatch(
        gatherSource(R, D),
        [relIdsBuf, sw['encoder.relative_positional_embedding.embedding.weight'].buffer],
        R * D,
        grid1d(R * D, 256)
      );
      for (let layer = 0; layer < layers; layer++) {
        signal?.throwIfAborted();
        const p = `encoder.transformer.blocks.${layer}`, g = `${p}.geom_attn`;
        const ns = await lnS(x, g + '.s_norm', R);
        const proj = await linS(ns, g + '.proj', R, D, H * 15);
        const vec = await this.dispatch(geomRotateSource(R, H), [proj, knnFramesBuf], R * H * 15, grid1d(R * H * 15, 64));
        const scores = await this.dispatch(
          geomScoresSource(K, H, B),
          [vec, knnFramesBuf, sw[g + '.distance_scale_per_head'].buffer, sw[g + '.rotation_scale_per_head'].buffer],
          B * H * K * K,
          grid1d(B * H * K * K, 64)
        );
        const prob = await this.dispatch(softmaxSource(K, B * H), [scores], B * H * K * K, grid1d(B * H * K, 1));
        const ctx = await this.dispatch(geomContextSource(K, H, B), [prob, vec, knnFramesBuf], R * H * 3, grid1d(R * H * 3, 64));
        const gOut = await linS(ctx, g + '.out_proj', R, H * 3, D);
        x = await elemS(R * D, 'y[i]=x[i]+b[i];', [x, gOut]);
        this.releaseExcept([x, knnFramesBuf, nodeFramesBuf, allSelected], traces);
        const ffnIn = await lnS(x, p + '.ffn.0', R);
        const ff = await linS(ffnIn, p + '.ffn.1', R, D, 2 * F);
        const act = await elemS(R * F, `let r=i/${F}u;let j=i%${F}u;let a=x[r*${2 * F}u+j];y[i]=(a/(1.0+exp(-a)))*x[r*${2 * F}u+${F}u+j];`, [ff]);
        const fOut = await linS(act, p + '.ffn.3', R, F, D);
        x = await elemS(R * D, 'y[i]=x[i]+b[i];', [x, fOut]);
        this.releaseExcept([x, knnFramesBuf, nodeFramesBuf, allSelected], traces);
      }
      const chunkSel = await this.dispatch(selectLocalSource(B, K, D), [x, nodeFramesBuf], B * D, grid1d(B * D, 256));
      this._endPass();
      this.encoder.copyBufferToBuffer(chunkSel, 0, allSelected, b0 * D * 4, B * D * 4);
      this.releaseExcept([allSelected], traces);
    }
    const preVq = await linS(allSelected, 'encoder.pre_vq_proj', L, D, codeDim);
    save('pre_vq', preVq, L * codeDim, 'F32');
    const cb = sw['encoder.codebook.embeddings'];
    const dots = await this.dispatch(
      matmulSource(L, codes, codeDim, 'F32', false, false),
      [preVq, cb.buffer, this.zero, this.zero],
      L * codes,
      matmulGrid(L, codes)
    );
    const structTokensBuf = await this.dispatch(
      codebookArgminSource(L, codes, codeDim),
      [preVq, dots, sw['encoder.codebook.norms'].buffer],
      L,
      grid1d(L, 1)
    );
    this.releaseExcept([structTokensBuf], traces);
    return structTokensBuf;
  }
  async encodeStructure(structureCoords, { residueIndex, signal, onProgress } = {}) {
    if (this.destroyed) throw new Error('Engine destroyed');
    if (this.lost) throw new Error(this.lost);
    if (this.busy) throw new Error('A prediction is already running');
    signal?.throwIfAborted();
    const { coords, length: L } = normalizeBackboneCoords(structureCoords, this.config.max_sequence_length);
    const resIdx = normalizeResidueIndex(residueIndex, L);
    await this.preloadStructureEncoder({ onProgress, signal });
    signal?.throwIfAborted();
    if (this.destroyed) throw new Error('Engine destroyed');
    this.busy = true;
    this.temporary = new Set();
    this.liveTemps = new Set();
    this.pool = new Map();
    this.encoder = this.device.createCommandEncoder();
    this.pass = null;
    this.hasNewPipelines = false;
    const started = performance.now();
    let checkValidation = false;
    try {
      const structTokensBuf = await this._encodeStructureOps(coords, resIdx, { signal });
      this._endPass();
      checkValidation = this.hasNewPipelines;
      if (checkValidation) this.device.pushErrorScope('validation');
      const readback = this.device.createBuffer({ size: L * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.buffers.add(readback);
      this.temporary.add(readback);
      this.encoder.copyBufferToBuffer(structTokensBuf, 0, readback, 0, L * 4);
      this.device.queue.submit([this.encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const structureTokens = new Int32Array(readback.getMappedRange().slice(0, L * 4));
      readback.unmap();
      signal?.throwIfAborted();
      return { structureTokens, timingMs: performance.now() - started };
    } finally {
      this._endPass();
      const validation = checkValidation ? await this.device.popErrorScope() : null;
      for (const b of this.temporary) { b.destroy(); this.buffers.delete(b); }
      this.temporary.clear(); this.liveTemps.clear(); this.pool.clear(); this.busy = false;
      if (validation) throw new Error(`WebGPU validation: ${validation.message}`);
    }
  }
  async predict(sequence, opts = {}) {
    const { returnIntermediates = false, signal, onProgress } = opts;
    if (this.destroyed) throw new Error('Engine destroyed');
    if (this.lost) throw new Error(this.lost);
    if (this.busy) throw new Error('A prediction is already running');
    signal?.throwIfAborted();
    const resolved = resolvePredictInputs(sequence, opts, this.config.max_sequence_length);
    const L = resolved.length, D = this.config.d_model, tokens = resolved.seqTokens;
    if (resolved.mode !== 'seq-only') {
      await this.preloadStructureEncoder({ onProgress, signal });
      signal?.throwIfAborted();
      if (this.destroyed) throw new Error('Engine destroyed');
    }
    this.busy = true;
    this.temporary = new Set();
    this.liveTemps = new Set();
    this.pool = new Map();
    this.encoder = this.device.createCommandEncoder();
    this.pass = null;
    this.hasNewPipelines = false;
    const started = performance.now(), traces = {}, w = this.weights;
    let checkValidation = false;
    const save = (name, value, n = L * D, dtype = 'F32') => { if (returnIntermediates) traces[name] = { buffer: value, n, dtype }; };
    const elem = (n, exp, inputs) => this.dispatch(elementSource(n, exp, inputs.length), inputs, n, grid1d(n, 256));
    const ln = (x, p, { stride = D, offset = 0, eps = 1e-5 } = {}) => this.dispatch(
      normSource(L, D, { stride, offset, eps, bias: !!w[p + '.bias'] }),
      [x, w[p + '.weight'].buffer, w[p + '.bias']?.buffer ?? this.zero],
      L * D,
      grid1d(L, 1)
    );
    const linear = (x, p, K = D, N = D, raw = false) => {
      const weight = w[p + (raw ? '_weight' : '.weight')], bias = w[p + (raw ? '_bias' : '.bias')];
      return this.dispatch(
        matmulSource(L, N, K, weight.dtype, !!bias, this.useF16),
        [x, weight.buffer, bias?.buffer ?? this.zero, weight.scales],
        L * N,
        matmulGrid(L, N)
      );
    };
    const residual = (x, y, scale = 1) => elem(L * D, `y[i]=x[i]+b[i]/${scale};`, [x, y]);
    const attn = async (q, k, qkv, H, layout) => {
      if (!layout?.packedQkv && D / H === 64) {
        return this.dispatch(streamingAttn64Source(D, H), [q, k, qkv], L * D, [Math.ceil(L / 64), H]);
      }
      const scores = await this.dispatch(scoresSource(L, D, H, layout), [q, k], H * L * L, grid1d(H * L * L, 64));
      const prob = await this.dispatch(softmaxSource(L, H), [scores], H * L * L, grid1d(H * L, 1));
      return this.dispatch(contextSource(L, D, H, { stride: 3 * D, offset: 2 * D }), [prob, qkv], L * D, grid1d(L * D, 64));
    };
    try {
      let x;
      let structTokensBuf = null;
      if (resolved.mode === 'seq-only') {
        x = await this.dispatch(gatherSource(L, D), [this.buffer(tokens, true), w['embedding.weight'].buffer], L * D, grid1d(L * D, 256));
      } else {
        if (resolved.structureTokens) {
          structTokensBuf = this.buffer(new Uint32Array(resolved.structureTokens), true);
        } else {
          structTokensBuf = await this._encodeStructureOps(resolved.coords, resolved.residueIndex, { signal, traces, save });
        }
        traces.__structure_tokens = { buffer: structTokensBuf, n: L, dtype: 'I32' };
        if (returnIntermediates) traces.structure_tokens = { buffer: structTokensBuf, n: L, dtype: 'I32' };
        x = await this.dispatch(
          gatherAddSource(L, D),
          [this.buffer(tokens, true), structTokensBuf, w['embedding.weight'].buffer, this.structWeights['structure_tokens_embed_delta'].buffer],
          L * D,
          grid1d(L * D, 256)
        );
      }
      save('embedding', x);
      for (let i = 0; i < this.config.backbone_layers; i++) {
        signal?.throwIfAborted();
        if (this.destroyed) throw new Error('Engine destroyed');
        const p = `backbone.${i}`, F = this.config.ffn_size, H = this.config.backbone_heads;
        const qkv = await linear(await ln(x, p + '.attn.layernorm_qkv.0'), p + '.attn.layernorm_qkv.1', D, 3 * D);
        let q = await ln(qkv, p + '.attn.q_ln', { stride: 3 * D }), k = await ln(qkv, p + '.attn.k_ln', { stride: 3 * D, offset: D });
        q = await this.dispatch(ropeSource(L, D, H), [q], L * D, grid1d(L * D, 256));
        k = await this.dispatch(ropeSource(L, D, H), [k], L * D, grid1d(L * D, 256));
        x = await residual(x, await linear(await attn(q, k, qkv, H), p + '.attn.out_proj'), this.config.residual_scale);
        const ff = await linear(await ln(x, p + '.ffn.0'), p + '.ffn.1', D, 2 * F);
        const act = await elem(L * F, `let r=i/${F}u;let j=i%${F}u;let a=x[r*${2 * F}u+j];y[i]=(a/(1.0+exp(-a)))*x[r*${2 * F}u+${F}u+j];`, [ff]);
        x = await residual(x, await linear(act, p + '.ffn.3', F, D), this.config.residual_scale);
        save(`backbone.${i}`, x);
        this.releaseExcept(x, traces);
      }
      save('final_embeddings', x);
      x = await ln(x, 'head.norm1', { eps: 1e-12 });
      for (let i = 0; i < this.config.head_layers; i++) {
        signal?.throwIfAborted();
        const p = `head.encoder.layers.${i}`, F = this.config.head_ffn_size;
        const qkv = await linear(await ln(x, p + '.norm1'), p + '.self_attn.in_proj', D, 3 * D, true);
        const context = await attn(qkv, qkv, qkv, this.config.head_heads, { packedQkv: true });
        x = await residual(x, await linear(context, p + '.self_attn.out_proj'));
        const a = await linear(await ln(x, p + '.norm2'), p + '.linear1', D, F);
        const relu = await elem(L * F, 'y[i]=max(x[i],0.0);', [a]);
        x = await residual(x, await linear(relu, p + '.linear2', F, D));
        save(`head.${i}`, x);
        this.releaseExcept(x, traces);
      }
      const logits = await linear(await ln(x, 'head.norm2', { eps: 1e-12 }), 'head.output_layer', D, 1);
      const probs = await elem(L, `y[i]=1.0/(1.0+exp(-(x[i]-(${Math.log(.05 / .95)}))));`, [logits]);
      traces.logits = { buffer: logits, n: L, dtype: 'F32' };
      traces.probs = { buffer: probs, n: L, dtype: 'F32' };
      this._endPass();
      checkValidation = this.hasNewPipelines;
      if (checkValidation) this.device.pushErrorScope('validation');
      const size = Object.values(traces).reduce((a, t) => a + t.n * 4, 0);
      const readback = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.buffers.add(readback);
      this.temporary.add(readback);
      let offset = 0;
      for (const t of Object.values(traces)) {
        t.offset = offset;
        this.encoder.copyBufferToBuffer(t.buffer, 0, readback, offset, t.n * 4);
        offset += t.n * 4;
      }
      this.device.queue.submit([this.encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange(), values = {};
      for (const [name, t] of Object.entries(traces)) {
        const slice = mapped.slice(t.offset, t.offset + t.n * 4);
        values[name] = t.dtype === 'I32' ? new Int32Array(slice) : new Float32Array(slice);
      }
      readback.unmap();
      signal?.throwIfAborted();
      for (const array of [values.probs, values.logits]) {
        for (const v of array) if (!Number.isFinite(v)) throw new Error('Nonfinite GPU output');
      }
      const structureTokens = values.__structure_tokens;
      delete values.__structure_tokens;
      return {
        probs: values.probs,
        logits: values.logits,
        residues: resolved.residues,
        ...(structureTokens ? { structureTokens } : {}),
        mode: resolved.mode,
        timingMs: performance.now() - started,
        ...(returnIntermediates ? { intermediates: { tokens, ...values } } : {}),
      };
    } finally {
      this._endPass();
      const validation = checkValidation ? await this.device.popErrorScope() : null;
      for (const b of this.temporary) { b.destroy(); this.buffers.delete(b); }
      this.temporary.clear(); this.liveTemps.clear(); this.pool.clear(); this.busy = false;
      if (validation) throw new Error(`WebGPU validation: ${validation.message}`);
    }
  }
  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this._endPass();
    for (const b of this.buffers) b.destroy();
    this.buffers.clear();
    this.weights = {};
    this.structWeights = null;
    this._structLoading = null;
    this.pipelines.clear();
    this.readyPipelines.clear();
    if (this.ownsDevice) this.device.destroy();
  }
}

export async function createDyna1Engine({
  baseUrl,
  precision = 'int8',
  structureEncoderUrl,
  preloadStructureEncoder = false,
  device,
  onProgress,
  signal,
  useF16 = true,
  cache = 'cache',
  read,
} = {}) {
  if (!baseUrl) throw new Error('baseUrl must point to the directory containing precision bundles');
  if (!['fp32', 'fp16', 'int8'].includes(precision)) throw new Error('Unknown precision');
  const rootUrl = baseUrl.replace(/\/$/, '');
  const structUrl = structureEncoderUrl ?? `${rootUrl}/structure_encoder`;
  const ownsDevice = !device;
  if (!device) ({ device } = await requestWebGpuDevice({ useF16 }));
  const engine = new Engine(device, {
    ownsDevice,
    useF16,
    structureEncoderUrl: structUrl,
    loadOptions: { onProgress, cache, read },
  });
  try {
    const bundle = await loadWeights(`${rootUrl}/${precision}`, {
      onProgress,
      signal,
      cache,
      read,
      onTensor: (name, t) => engine.upload(name, t),
    });
    engine.config = bundle.config;
    engine.manifest = bundle.manifest;
    if (preloadStructureEncoder) {
      await engine.preloadStructureEncoder({ onProgress, signal });
    }
    return engine;
  } catch (e) {
    engine.destroy();
    throw e;
  }
}
