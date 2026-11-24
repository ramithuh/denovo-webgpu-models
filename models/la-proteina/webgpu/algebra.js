import { edgeSoftmax } from './edge-softmax.js';
import { numericWGSL } from '../shared/numeric.js';
import { Algebra, sizeOf, stridesOf } from '../shared/tensor.js';
import constants from '../shared/constants.js';
import motifConstants from '../shared/motif-constants.js';
import {
  fold2,
  ACCURATE_DIV_SQRT_WGSL,
  gemmSource,
  gemvSource,
  CASCADE_NORM_SOURCE,
  ROW_SOFTMAX_SOURCE,
  PAIR_BINS_SOURCE,
  STRUCT_SEQ_FEATS_SOURCE,
  AE3_PAIR_BINS_SOURCE,
  LD4_MOTIF_BINS_SOURCE,
  LD4_PAIR_BINS_SOURCE,
  PAD_PAIR_SOURCE,
  sparsePairLinearSource,
  pairBiasPermuteSource,
  ATTN_LOGITS_SOURCE,
  ATTN_CONTEXT_GATED_SOURCE,
  FUSED_OPS_SOURCE,
  dynamicElementSource,
} from './kernels.js';

export class GPUAlgebra extends Algebra {
  constructor(device, { useF16 = false } = {}) {
    super();
    this.device = device;
    this.useF16 = Boolean(useF16 && device.features.has('shader-f16'));
    this.buffers = new Set();
    this.persistent = new Set();
    this.pipelines = new Map();
    this.encoder = null;
    this.pass = null;
    this.dispatches = 0;
    this.pool = new Map();
    this.available = new Set();
    this.dirtyInSubmit = new Set();
    this.uFree = [];
    this.uUsed = [];
    this.dummy = null;
    this.binsBuf = null;
    this.motifAuxBuf = null;
    this.readbackBuf = null;
    this.readbackSize = 0;
    this.validationError = null;
    this.destroyed = false;
    this.allocations = 0;
    this.reusedBuffers = 0;
    this.listener = (e) => {
      this.validationError = e.error;
    };
    device.addEventListener('uncapturederror', this.listener);
  }

  _ensureAux() {
    if (!this.dummy) {
      this.dummy = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    if (!this.binsBuf) {
      const bins = Float32Array.from(constants.distance_bins);
      this.binsBuf = this.device.createBuffer({
        size: Math.ceil(bins.byteLength / 16) * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(this.binsBuf, 0, bins);
    }
  }

  _ensureMotifAux() {
    this._ensureAux();
    if (!this.motifAuxBuf) {
      const words = new Uint32Array(460);
      const floats = new Float32Array(words.buffer);
      let pos = 0;
      for (const res of motifConstants.chi_atom_indices) {
        for (const chi of res) {
          for (const a of chi) words[pos++] = a >>> 0;
        }
      }
      for (const res of motifConstants.chi_angles_mask) {
        for (const m of res) words[pos++] = m > 0 ? 1 : 0;
      }
      for (let k = 0; k < 20; k++) floats[420 + k] = motifConstants.angle_bins[k];
      for (let k = 0; k < 20; k++) floats[440 + k] = motifConstants.bb_dist_bins[k];
      this.motifAuxBuf = this.device.createBuffer({
        size: words.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(this.motifAuxBuf, 0, words);
    }
  }

  _uniform(words) {
    const size = Math.max(16, Math.ceil(words.byteLength / 16) * 16);
    // Only reuse uniform buffers from uFree (which were retired before the last flush/submit).
    let buf = null;
    for (let i = this.uFree.length - 1; i >= 0; i--) {
      if (this.uFree[i].size >= size) {
        buf = this.uFree.splice(i, 1)[0];
        break;
      }
    }
    if (!buf) {
      buf = this.device.createBuffer({
        size: Math.max(96, size),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(buf, 0, words.buffer, words.byteOffset, words.byteLength);
    this.uUsed.push(buf);
    return buf;
  }

  _pipeline(key, code) {
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      this.pipelines.set(key, p);
    }
    return p;
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

  _dispatch(key, code, uniformWords, storageBuffers, groupsX, groupsY = 1) {
    const pipeline = this._pipeline(key, code);
    const uBuf = this._uniform(uniformWords);
    const entries = [{ binding: 0, resource: { buffer: uBuf } }];
    for (let i = 0; i < storageBuffers.length; i++) {
      const b = storageBuffers[i];
      this.dirtyInSubmit.add(b);
      entries.push({ binding: i + 1, resource: { buffer: b } });
    }
    const bg = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    const pass = this._beginPass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(groupsX, groupsY);
    if (++this.dispatches >= 512) this.flush();
  }

  _weightMode(w) {
    if (!w.dtype || w.dtype === 'F32') return 'F32';
    if (w.dtype === 'F16') return this.useF16 ? 'F16_NATIVE' : 'F16';
    return 'I8';
  }

  alloc(shape, persistent = false, forHostUpload = false) {
    if (this.destroyed) throw Error('GPU algebra destroyed');
    const size = sizeOf(shape);
    const bytes = Math.max(4, size * 4);
    const limit = Math.min(
      this.device.limits.maxBufferSize,
      this.device.limits.maxStorageBufferBindingSize,
    );
    if (bytes > limit) throw Error(`GPU buffer ${bytes} exceeds ${limit}`);
    let pooled = null;
    if (!persistent) {
      const list = this.pool.get(bytes);
      if (list && list.length) {
        if (!forHostUpload) {
          pooled = list.pop();
        } else {
          for (let i = list.length - 1; i >= 0; i--) {
            if (!this.dirtyInSubmit.has(list[i])) {
              pooled = list.splice(i, 1)[0];
              break;
            }
          }
        }
      }
    }
    if (pooled) this.reusedBuffers++;
    else this.allocations++;
    const buffer = pooled || this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.available.delete(buffer);
    this.buffers.add(buffer);
    if (persistent) this.persistent.add(buffer);
    return { shape, size, buffer };
  }

  upload(shape, data, persistent = false) {
    const out = this.alloc(shape, persistent, true);
    const view = data instanceof Float32Array || ArrayBuffer.isView(data) ? data : Float32Array.from(data);
    this.device.queue.writeBuffer(out.buffer, 0, view);
    return out;
  }

  tensor(shape, data) {
    return this.upload(shape, data instanceof Float32Array ? data : Float32Array.from(data));
  }

  constant(shape, value = 0) {
    this._ensureAux();
    const out = this.alloc(shape);
    const u = new Uint32Array(8);
    const f = new Float32Array(u.buffer);
    u[0] = out.size; u[1] = 1; u[2] = 3; f[3] = value;
    const [gx, gy] = fold2(Math.ceil(out.size / 256));
    this._dispatch('fused-ops', FUSED_OPS_SOURCE, u, [this.dummy, this.dummy, this.dummy, out.buffer], gx, gy);
    return out;
  }

  uploadWeight(t) {
    if (this.destroyed) throw Error('GPU algebra destroyed');
    if (t.dtype === 'F32') return this.upload(t.shape, t.data, true);
    const rawBytes = t.data.byteLength;
    const scaleOffset = Math.ceil(rawBytes / 4);
    const bytes = (scaleOffset + (t.scale?.length ?? 0)) * 4;
    const limit = Math.min(
      this.device.limits.maxBufferSize,
      this.device.limits.maxStorageBufferBindingSize,
    );
    if (bytes > limit) throw Error('Weight exceeds GPU limits');
    const packed = new Uint8Array(bytes);
    packed.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength));
    if (t.scale) new Float32Array(packed.buffer, scaleOffset * 4).set(t.scale);
    const buffer = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.allocations++;
    this.buffers.add(buffer);
    this.persistent.add(buffer);
    this.device.queue.writeBuffer(buffer, 0, packed);
    return {
      shape: t.shape,
      size: sizeOf(t.shape),
      buffer,
      dtype: t.dtype,
      scaleOffset,
    };
  }

  linear(x, w, b) {
    this._ensureAux();
    const K = x.shape.at(-1);
    const N = w.shape[0];
    const M = x.size / K;
    const out = this.alloc([...x.shape.slice(0, -1), N]);
    const wMode = this._weightMode(w);

    if (M <= 4) {
      const u = new Uint32Array([M, N, K, b ? 1 : 0, w.scaleOffset ?? 0, 0, 0, 0]);
      const [gx, gy] = fold2(Math.ceil((M * N) / 64));
      const key = `gemv:${wMode}`;
      this._dispatch(key, gemvSource(wMode), u, [x.buffer, w.buffer, b?.buffer ?? this.dummy, out.buffer], gx, gy);
      return out;
    }

    const tileM = M >= 192 ? 64 : 32;
    const u = new Uint32Array(24);
    const f = new Float32Array(u.buffer);
    u[0] = M; u[1] = N; u[2] = K; u[3] = 1;
    u[4] = 0; u[5] = 0; u[6] = 0; u[7] = K;
    u[8] = 1; u[9] = 0; u[10] = 0; u[11] = 0;
    u[12] = K; u[13] = 1; u[14] = 0; u[15] = 0;
    u[16] = 0; u[17] = N; u[18] = 1; u[19] = b ? 1 : 0;
    u[20] = 0; f[21] = 1.0; u[22] = 1; u[23] = w.scaleOffset ?? 0;
    const groups = Math.ceil(M / tileM) * Math.ceil(N / 128);
    const [gx, gy] = fold2(groups);
    const key = `gemm:${tileM}:${wMode}`;
    this._dispatch(key, gemmSource({ tileM, wMode }), u, [x.buffer, w.buffer, b?.buffer ?? this.dummy, out.buffer], gx, gy);
    return out;
  }

  einsum(eq, ...xs) {
    if (xs.length === 2 && !xs[0].dtype && !xs[1].dtype) {
      const [left, right] = eq.replaceAll(' ', '').split('->');
      const [ta, tb] = left.split(',');
      if (
        ta.length === xs[0].shape.length &&
        tb.length === xs[1].shape.length &&
        new Set(ta).size === ta.length &&
        new Set(tb).size === tb.length &&
        new Set(right).size === right.length
      ) {
        const dims = {};
        [...ta].forEach((c, i) => { dims[c] = xs[0].shape[i]; });
        [...tb].forEach((c, i) => { dims[c] = xs[1].shape[i]; });
        const batchAxes = [...right].filter((c) => ta.includes(c) && tb.includes(c));
        const mAxes = [...right].filter((c) => ta.includes(c) && !tb.includes(c));
        const nAxes = [...right].filter((c) => !ta.includes(c) && tb.includes(c));
        const kAxes = Object.keys(dims).filter((c) => !right.includes(c) && ta.includes(c) && tb.includes(c)).sort();
        // Fast strided batched GEMM when batchAxes.length <= 2, mAxes.length == 1, nAxes.length == 1, kAxes.length == 1
        if (batchAxes.length <= 2 && mAxes.length === 1 && nAxes.length === 1 && kAxes.length === 1) {
          this._ensureAux();
          const stA = stridesOf(xs[0].shape);
          const stB = stridesOf(xs[1].shape);
          const outShape = [...right].map((c) => dims[c]);
          const stC = stridesOf(outShape);
          const strideIn = (term, st, axis) => (axis ? st[term.indexOf(axis)] : 0);
          const b0Axis = batchAxes[0] ?? null;
          const b1Axis = batchAxes[1] ?? null;
          const B0 = b0Axis ? dims[b0Axis] : 1;
          const B1 = b1Axis ? dims[b1Axis] : 1;
          const M = dims[mAxes[0]];
          const N = dims[nAxes[0]];
          const K = dims[kAxes[0]];
          const tileM = M >= 192 ? 64 : 32;
          const out = this.alloc(outShape);
          const u = new Uint32Array(24);
          const f = new Float32Array(u.buffer);
          u[0] = M; u[1] = N; u[2] = K; u[3] = B1;
          u[4] = 0;
          u[5] = strideIn(ta, stA, b0Axis);
          u[6] = strideIn(ta, stA, b1Axis);
          u[7] = strideIn(ta, stA, mAxes[0]);
          u[8] = strideIn(ta, stA, kAxes[0]);
          u[9] = 0;
          u[10] = strideIn(tb, stB, b0Axis);
          u[11] = strideIn(tb, stB, b1Axis);
          u[12] = strideIn(tb, stB, nAxes[0]);
          u[13] = strideIn(tb, stB, kAxes[0]);
          u[14] = 0;
          u[15] = strideIn(right, stC, b0Axis);
          u[16] = strideIn(right, stC, b1Axis);
          u[17] = strideIn(right, stC, mAxes[0]);
          u[18] = strideIn(right, stC, nAxes[0]);
          u[19] = 0; u[20] = 0; f[21] = 1.0; u[22] = B0 * B1; u[23] = 0;
          const groups = Math.ceil(M / tileM) * Math.ceil(N / 128) * B0 * B1;
          const [gx, gy] = fold2(groups);
          const key = `gemm:${tileM}:F32`;
          this._dispatch(key, gemmSource({ tileM, wMode: 'F32' }), u, [xs[0].buffer, xs[1].buffer, this.dummy, out.buffer], gx, gy);
          return out;
        }
      }
    }
    return super.einsum(eq, ...xs);
  }

  norm(x, w = null, b = null, eps = 1e-5) {
    const N = x.shape.at(-1);
    if (N % 4 === 0 && N <= 2048) {
      this._ensureAux();
      const rows = x.size / N;
      const out = this.alloc(x.shape);
      const u = new Uint32Array(12);
      const f = new Float32Array(u.buffer);
      u[0] = rows; u[1] = N; u[2] = N; u[3] = 0;
      u[4] = w ? 1 : 0; u[5] = 0; f[6] = eps; u[7] = 0;
      const [gx, gy] = fold2(rows);
      this._dispatch('cascade-norm', CASCADE_NORM_SOURCE, u, [x.buffer, w?.buffer ?? this.dummy, b?.buffer ?? this.dummy, out.buffer], gx, gy);
      return out;
    }
    return super.norm(x, w, b, eps);
  }

  normStrided(x, rows, width, stride, offset, w, b, eps = 1e-5) {
    this._ensureAux();
    const out = this.alloc([rows, width]);
    const u = new Uint32Array(12);
    const f = new Float32Array(u.buffer);
    u[0] = rows; u[1] = width; u[2] = stride; u[3] = offset;
    u[4] = w ? 1 : 0; u[5] = 0; f[6] = eps; u[7] = 0;
    const [gx, gy] = fold2(rows);
    this._dispatch('cascade-norm', CASCADE_NORM_SOURCE, u, [x.buffer, w?.buffer ?? this.dummy, b?.buffer ?? this.dummy, out.buffer], gx, gy);
    return out;
  }

  fastAdaptive(x, gammaRaw, beta, eps = 1e-5, splitRow = 0) {
    this._ensureAux();
    const N = x.shape.at(-1);
    const rows = x.size / N;
    const out = this.alloc(x.shape);
    const u = new Uint32Array(12);
    const f = new Float32Array(u.buffer);
    u[0] = rows; u[1] = N; u[2] = N; u[3] = 0;
    u[4] = 1; u[5] = 1; f[6] = eps; u[7] = 0;
    u[8] = splitRow;
    const [gx, gy] = fold2(rows);
    this._dispatch('cascade-norm', CASCADE_NORM_SOURCE, u, [x.buffer, gammaRaw.buffer, beta.buffer, out.buffer], gx, gy);
    return out;
  }

  rowStats(x, eps = 1e-5) {
    this._ensureAux();
    const N = x.shape.at(-1);
    const rows = x.size / N;
    const out = this.alloc([rows, 2]);
    const u = new Uint32Array(12);
    const f = new Float32Array(u.buffer);
    u[0] = rows; u[1] = N; u[2] = N; u[3] = 0;
    u[4] = 0; u[5] = 2; f[6] = eps; u[7] = 0;
    const [gx, gy] = fold2(rows);
    this._dispatch('cascade-norm', CASCADE_NORM_SOURCE, u, [x.buffer, this.dummy, this.dummy, out.buffer], gx, gy);
    return out;
  }

  structSeqFeatures(coordsNm, coordMask, resType, extra, mode, coordScale = 1.0) {
    this._ensureMotifAux();
    const L = resType.shape[0];
    const rows = mode === 3 ? extra.shape[0] : L;
    const width = mode === 0 ? 468 : mode === 1 ? 504 : 205;
    const out = this.alloc([rows, width]);
    const u = new Uint32Array(8);
    const f = new Float32Array(u.buffer);
    u[0] = rows; u[1] = L; u[2] = width; u[3] = mode;
    f[4] = coordScale;
    const [gx, gy] = fold2(Math.ceil(rows / 64));
    this._dispatch(
      'struct-seq-feats',
      STRUCT_SEQ_FEATS_SOURCE,
      u,
      [coordsNm.buffer, coordMask.buffer, resType.buffer, extra?.buffer ?? this.dummy, this.motifAuxBuf, out.buffer],
      gx,
      gy,
    );
    return out;
  }

  ae3PairLinear(coordsNm, coordMask, w, b) {
    this._ensureMotifAux();
    const L = coordMask.shape[0];
    const pairs = L * L;
    const indices = this.alloc([pairs, 12]);
    const uBins = new Uint32Array([L, 0, 0, 0]);
    const [bx, by] = fold2(Math.ceil(pairs / 64));
    this._dispatch('ae3-pair-bins', AE3_PAIR_BINS_SOURCE, uBins, [coordsNm.buffer, coordMask.buffer, this.motifAuxBuf, indices.buffer], bx, by);

    const N = w.shape[0];
    const K = w.shape[1];
    const out = this.alloc([L, L, N]);
    const wMode = this._weightMode(w);
    const uLin = new Uint32Array([pairs, N, K, b ? 1 : 0, w.scaleOffset ?? 0, 10, 12, 0]);
    const [lx, ly] = fold2(Math.ceil((pairs * N) / 256));
    const key = `sparse-pair:${wMode}`;
    this._dispatch(key, sparsePairLinearSource(wMode), uLin, [indices.buffer, w.buffer, b?.buffer ?? this.dummy, out.buffer], lx, ly);
    return out;
  }

  ld4MotifBins(xMotif, motifMask) {
    this._ensureMotifAux();
    const L = motifMask.shape[0];
    const pairs = L * L;
    const out = this.alloc([pairs, 4]);
    const u = new Uint32Array([L, 0, 0, 0]);
    const [gx, gy] = fold2(Math.ceil(pairs / 64));
    this._dispatch('ld4-motif-bins', LD4_MOTIF_BINS_SOURCE, u, [xMotif.buffer, motifMask.buffer, this.motifAuxBuf, out.buffer], gx, gy);
    return out;
  }

  ld4PairLinear(ca, scCa, motifBins, w, b) {
    this._ensureAux();
    const L = ca.shape[0];
    const pairs = L * L;
    const indices = this.alloc([pairs, 8]);
    const uBins = new Uint32Array([L, scCa ? 1 : 0, 0, 0]);
    const [bx, by] = fold2(Math.ceil(pairs / 64));
    this._dispatch(
      'ld4-pair-bins',
      LD4_PAIR_BINS_SOURCE,
      uBins,
      [ca.buffer, scCa?.buffer ?? this.dummy, this.binsBuf, motifBins.buffer, indices.buffer],
      bx,
      by,
    );

    const N = w.shape[0];
    const K = w.shape[1];
    const out = this.alloc([L, L, N]);
    const wMode = this._weightMode(w);
    const uLin = new Uint32Array([pairs, N, K, b ? 1 : 0, w.scaleOffset ?? 0, 7, 8, 0]);
    const [lx, ly] = fold2(Math.ceil((pairs * N) / 256));
    const key = `sparse-pair:${wMode}`;
    this._dispatch(key, sparsePairLinearSource(wMode), uLin, [indices.buffer, w.buffer, b?.buffer ?? this.dummy, out.buffer], lx, ly);
    return out;
  }

  padPair(pair, L, N) {
    const C = pair.shape[2];
    const total = N * N * C;
    const out = this.alloc([N, N, C]);
    const u = new Uint32Array([L, N, C, total]);
    const [gx, gy] = fold2(Math.ceil(total / 256));
    this._dispatch('pad-pair', PAD_PAIR_SOURCE, u, [pair.buffer, out.buffer], gx, gy);
    return out;
  }

  pairLinearInit(ca, scCa, w, b) {
    this._ensureAux();
    const L = ca.shape[0];
    const pairs = L * L;
    const indices = this.alloc([pairs, 4]);
    const uBins = new Uint32Array([L, scCa ? 1 : 0, 0, 0]);
    const [bx, by] = fold2(Math.ceil(pairs / 64));
    this._dispatch('pair-bins', PAIR_BINS_SOURCE, uBins, [ca.buffer, scCa?.buffer ?? this.dummy, this.binsBuf, indices.buffer], bx, by);

    const N = w.shape[0];
    const K = w.shape[1];
    const out = this.alloc([L, L, N]);
    const wMode = this._weightMode(w);
    const uLin = new Uint32Array([pairs, N, K, b ? 1 : 0, w.scaleOffset ?? 0, 0, 0, 0]);
    const [lx, ly] = fold2(Math.ceil((pairs * N) / 256));
    const key = `sparse-pair:${wMode}`;
    this._dispatch(key, sparsePairLinearSource(wMode), uLin, [indices.buffer, w.buffer, b?.buffer ?? this.dummy, out.buffer], lx, ly);
    return out;
  }

  pairBiasPermute(pair, stats, nw, nb, bw, bb, L, H) {
    this._ensureAux();
    const pairs = L * L;
    const K = pair.shape.at(-1);
    const out = this.alloc([H, L, L]);
    const wMode = this._weightMode(bw);
    const u = new Uint32Array([pairs, K, H, bb ? 1 : 0, bw.scaleOffset ?? 0, 0, 0, 0]);
    const [gx, gy] = fold2(Math.ceil(pairs / 16));
    const key = `pair-bias-perm:${wMode}`;
    this._dispatch(
      key,
      pairBiasPermuteSource(wMode),
      u,
      [pair.buffer, stats.buffer, nw.buffer, nb.buffer, bw.buffer, bb?.buffer ?? this.dummy, out.buffer],
      gx,
      gy,
    );
    return out;
  }

  attnLogits(q, k, pairBias, L, H, D) {
    const out = this.alloc([H, L, L]);
    const u = new Uint32Array(4);
    const f = new Float32Array(u.buffer);
    u[0] = L; u[1] = H; u[2] = D; f[3] = Math.fround(1 / Math.sqrt(D));
    const tilesL = Math.ceil(L / 32);
    const [gx, gy] = fold2(H * tilesL * tilesL);
    this._dispatch('attn-logits', ATTN_LOGITS_SOURCE, u, [q.buffer, k.buffer, pairBias.buffer, out.buffer], gx, gy);
    return out;
  }

  attnContextGated(probs, qkv, gate, L, H, D) {
    const C = H * D;
    const out = this.alloc([L, C]);
    const u = new Uint32Array([L, H, D, 3 * C, 2 * C, 0, 0, 0]);
    const [gx, gy] = fold2(H * Math.ceil(L / 32));
    this._dispatch('attn-ctx-gated', ATTN_CONTEXT_GATED_SOURCE, u, [probs.buffer, qkv.buffer, gate.buffer, out.buffer], gx, gy);
    return out;
  }

  fastSwiglu(h) {
    this._ensureAux();
    const width = h.shape.at(-1) / 2;
    const outShape = [...h.shape.slice(0, -1), width];
    const total = sizeOf(outShape);
    const out = this.alloc(outShape);
    const u = new Uint32Array([total, width, 0, 0, 0, 0, 0, 0]);
    const [gx, gy] = fold2(Math.ceil(total / 256));
    this._dispatch('fused-ops', FUSED_OPS_SOURCE, u, [h.buffer, this.dummy, this.dummy, out.buffer], gx, gy);
    return out;
  }

  fastResidualScaleAdd(seq, update, gammaRaw, splitRow = 0) {
    const width = seq.shape.at(-1);
    const total = seq.size;
    const out = this.alloc(seq.shape);
    const u = new Uint32Array([total, width, 1, 0, splitRow, 0, 0, 0]);
    const [gx, gy] = fold2(Math.ceil(total / 256));
    this._dispatch('fused-ops', FUSED_OPS_SOURCE, u, [seq.buffer, update.buffer, gammaRaw.buffer, out.buffer], gx, gy);
    return out;
  }

  fastScale(x, gammaRaw, splitRow = 0) {
    this._ensureAux();
    const width = x.shape.at(-1);
    const total = x.size;
    const out = this.alloc(x.shape);
    const u = new Uint32Array([total, width, 2, 0, splitRow, 0, 0, 0]);
    const [gx, gy] = fold2(Math.ceil(total / 256));
    this._dispatch('fused-ops', FUSED_OPS_SOURCE, u, [x.buffer, gammaRaw.buffer, this.dummy, out.buffer], gx, gy);
    return out;
  }

  softmax(x, axis = -1, dgl = false) {
    if (dgl) return edgeSoftmax(this, x, axis);
    let ax = axis < 0 ? axis + x.shape.length : axis;
    if (ax === x.shape.length - 1) {
      const len = x.shape[ax];
      const rows = x.size / len;
      const out = this.alloc(x.shape);
      const u = new Uint32Array([rows, len, 0, 0]);
      const [gx, gy] = fold2(rows);
      this._dispatch('row-softmax', ROW_SOFTMAX_SOURCE, u, [x.buffer, out.buffer], gx, gy);
      return out;
    }
    return super.softmax(x, axis, false);
  }

  element(expression, ...xs) {
    const ts = xs.filter((x) => typeof x !== 'number');
    const rank = Math.max(...ts.map((x) => x.shape.length));
    if (ts.length >= 1 && ts.length <= 2 && rank <= 4 && ts.every((t) => !t.dtype || t.dtype === 'F32')) {
      const shape = Array(rank).fill(1);
      for (const t of ts) {
        t.shape.forEach((d, j) => {
          const k = rank - t.shape.length + j;
          if (shape[k] !== 1 && d !== 1 && shape[k] !== d) throw Error('broadcast');
          shape[k] = Math.max(shape[k], d);
        });
      }
      const scalars = [0, 0, 0, 0];
      let sIdx = 0;
      let tIdx = 0;
      const tMap = new Map();
      const sFields = ['p.scalars.x', 'p.scalars.y', 'p.scalars.z', 'p.scalars.w'];
      const wgExpr = expression
        .replace(/\$(\d+)/g, (_, k) => {
          const idx = Number(k);
          const arg = xs[idx];
          if (typeof arg === 'number') {
            scalars[sIdx] = Math.fround(arg);
            return sFields[sIdx++];
          }
          if (!tMap.has(idx)) tMap.set(idx, `v${tIdx++}`);
          return tMap.get(idx);
        })
        .replaceAll('exp(', 'rfExp(')
        .replaceAll('sqrt(', 'rfSqrt(');

      const pad4 = (arr, fill) => {
        const out = [fill, fill, fill, fill];
        const off = 4 - arr.length;
        for (let i = 0; i < arr.length; i++) out[off + i] = arr[i];
        return out;
      };
      const outDims = pad4(shape, 1);
      const outStrides = pad4(stridesOf(shape), 1);
      const inStrides = (t) => {
        const st = stridesOf(t.shape);
        const aligned = t.shape.map((d, j) => (d === 1 ? 0 : st[j]));
        return pad4(aligned, 0);
      };
      const s0 = inStrides(ts[0]);
      const s1 = ts[1] ? inStrides(ts[1]) : [0, 0, 0, 0];
      const total = sizeOf(shape);
      const out = this.alloc(shape);
      const u = new Uint32Array(24);
      const f = new Float32Array(u.buffer);
      u[0] = total; u[1] = rank; u[2] = 0; u[3] = 0;
      u.set(outStrides, 4);
      u.set(outDims, 8);
      u.set(s0, 12);
      u.set(s1, 16);
      f.set(scalars, 20);
      const key = `dyn-elem:${ts.length}:${wgExpr}`;
      const [gx, gy] = fold2(Math.ceil(total / 256));
      this._dispatch(key, dynamicElementSource(wgExpr, ts.length), u, [...ts.map((t) => t.buffer), out.buffer], gx, gy);
      return out;
    }
    return super.element(expression, ...xs);
  }

  sigmoid(a) {
    return this.element('rfdiv(1.0,1.0+exp(-$0))', a);
  }

  silu(a) {
    return this.element('rfdiv($0,1.0+exp(-$0))', a);
  }

  kernel(shape, inputs, js, wgsl) {
    const size = sizeOf(shape);
    const rewrite = (code) => {
      let result = '';
      let last = 0;
      const re = /a(\d+)\[/g;
      let match;
      while ((match = re.exec(code))) {
        let depth = 1;
        let end = re.lastIndex;
        while (depth && end < code.length) {
          if (code[end] === '[') depth++;
          if (code[end] === ']') depth--;
          end++;
        }
        result += code.slice(last, match.index) + `load${match[1]}(${rewrite(code.slice(re.lastIndex, end - 1))})`;
        last = end;
        re.lastIndex = end;
      }
      return result + code.slice(last);
    };
    const declarations = inputs
      .map((t, j) => {
        const raw = t.dtype && t.dtype !== 'F32';
        const binding = `@group(0) @binding(${j}) var<storage,read> input${j}:array<${raw ? 'u32' : 'f32'}>;`;
        let expression = `input${j}[index]`;
        if (t.dtype === 'F16') expression = `unpack2x16float(input${j}[index/2u])[index%2u]`;
        if (t.dtype === 'I8') {
          expression = `f32(i32((input${j}[index/4u]>>((index%4u)*8u))<<24u)>>24)*bitcast<f32>(input${j}[${t.scaleOffset}u+index/${t.shape[1]}u])`;
        }
        return binding + `fn load${j}(index:u32)->f32{return ${expression};}`;
      })
      .join('\n');
    const code =
      numericWGSL +
      ACCURATE_DIV_SQRT_WGSL +
      declarations +
      `@group(0) @binding(${inputs.length}) var<storage,read_write> o:array<f32>;\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x+gid.y*65535u*64u;if(i>=${size}u){return;}${rewrite(wgsl).replaceAll('sqrt(', 'rfSqrt(')}}`;
    const pipeline = this._pipeline(code, code);
    const out = this.alloc(shape);
    for (const t of inputs) this.dirtyInSubmit.add(t.buffer);
    this.dirtyInSubmit.add(out.buffer);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [...inputs, out].map((t, binding) => ({
        binding,
        resource: { buffer: t.buffer },
      })),
    });
    const pass = this._beginPass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const groups = Math.ceil(size / 64);
    pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
    if (++this.dispatches >= 512) this.flush();
    return out;
  }

  flush() {
    if (this.encoder) {
      this._endPass();
      const enc = this.encoder;
      this.encoder = null;
      this.dirtyInSubmit.clear();
      this.dispatches = 0;
      const used = this.uUsed;
      this.uUsed = [];
      for (let i = 0; i < used.length; i++) this.uFree.push(used[i]);
      this.device.queue.submit([enc.finish()]);
    }
  }

  async read(t) {
    if (this.destroyed) throw Error('GPU algebra destroyed');
    this.flush();
    const b = this.device.createBuffer({
      size: Math.max(4, t.size * 4),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.buffers.add(b);
    try {
      const e = this.device.createCommandEncoder();
      e.copyBufferToBuffer(t.buffer, 0, b, 0, t.size * 4);
      this.device.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ);
      if (this.validationError) throw this.validationError;
      return new Float32Array(b.getMappedRange()).slice(0, t.size);
    } finally {
      b.destroy();
      this.buffers.delete(b);
    }
  }

  async readMany(tensors) {
    if (this.destroyed) throw Error('GPU algebra destroyed');
    this.flush();
    let totalBytes = 0;
    const offsets = [];
    for (const t of tensors) {
      offsets.push(totalBytes);
      totalBytes += Math.ceil((t.size * 4) / 4) * 4;
    }
    const b = this.device.createBuffer({
      size: Math.max(4, totalBytes),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.buffers.add(b);
    try {
      const e = this.device.createCommandEncoder();
      for (let i = 0; i < tensors.length; i++) {
        e.copyBufferToBuffer(tensors[i].buffer, 0, b, offsets[i], tensors[i].size * 4);
      }
      this.device.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ);
      if (this.validationError) throw this.validationError;
      const mapped = b.getMappedRange();
      return tensors.map((t, i) => new Float32Array(mapped.slice(offsets[i], offsets[i] + t.size * 4)));
    } finally {
      b.destroy();
      this.buffers.delete(b);
    }
  }

  _recycle(keep = []) {
    const live = new Set(
      keep
        .flat(Infinity)
        .filter(Boolean)
        .map((t) => t.buffer),
    );
    for (const b of this.buffers) {
      if (!this.persistent.has(b) && !live.has(b)) {
        if (!this.available.has(b)) {
          const list = this.pool.get(b.size) ?? [];
          list.push(b);
          this.pool.set(b.size, list);
          this.available.add(b);
        }
      }
    }
  }

  async checkpoint(keep = []) {
    this._recycle(keep);
    if (this.validationError) throw this.validationError;
  }

  async stepBoundary(keep = [], step = 0) {
    this._recycle(keep);
    this.flush();
    if ((step + 1) % 16 === 0) {
      await this.device.queue.onSubmittedWorkDone();
    }
    if (this.validationError) throw this.validationError;
  }

  clear() {
    this.flush();
    this.pool.clear();
    this.available.clear();
    this.dirtyInSubmit.clear();
    for (const b of this.uFree) b.destroy();
    for (const b of this.uUsed) b.destroy();
    this.uFree.length = 0;
    this.uUsed.length = 0;
    for (const b of this.buffers) {
      if (!this.persistent.has(b)) {
        b.destroy();
        this.buffers.delete(b);
      }
    }
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.device.removeEventListener('uncapturederror', this.listener);
    this._endPass();
    this.encoder = null;
    this.pool.clear();
    this.available.clear();
    this.dirtyInSubmit.clear();
    for (const b of this.uFree) b.destroy();
    for (const b of this.uUsed) b.destroy();
    this.uFree.length = 0;
    this.uUsed.length = 0;
    if (this.dummy) { this.dummy.destroy(); this.dummy = null; }
    if (this.binsBuf) { this.binsBuf.destroy(); this.binsBuf = null; }
    if (this.motifAuxBuf) { this.motifAuxBuf.destroy(); this.motifAuxBuf = null; }
    for (const b of this.buffers) b.destroy();
    this.buffers.clear();
    this.persistent.clear();
    this.pipelines.clear();
  }

  memoryUsage() {
    let bytes = 0;
    let transientBuffers = 0;
    for (const b of this.buffers) {
      bytes += b.size;
      if (!this.persistent.has(b)) transientBuffers++;
    }
    return {
      bytes,
      transientBuffers,
      allocations: this.allocations,
      reusedBuffers: this.reusedBuffers,
    };
  }
}
