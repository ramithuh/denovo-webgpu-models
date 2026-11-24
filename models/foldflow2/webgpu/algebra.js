import { edgeSoftmax } from './edge-softmax.js';
import { numericWGSL, erfWGSL, trigWGSL } from '../shared/numeric.js';
import { Algebra, sizeOf, stridesOf } from '../shared/tensor.js';
import {
  fold2,
  ACCURATE_DIV_SQRT_WGSL,
  gemmSource,
  gemvSource,
  FF2_CASCADE_NORM_SOURCE,
  ROW_SOFTMAX_SOURCE,
  FF2_EMBED_FEATURES_SOURCE,
  FF2_GELU_FILL_SOURCE,
  FF2_SO3_DQ_SOURCE,
  dynamicElementSource,
} from './kernels.js';

const POS_DENOM = new Float32Array(
  Array.from({ length: 16 }, (_, c) => Math.fround(2056 ** Math.fround(c / 16))),
);

export class GPUAlgebra extends Algebra {
  constructor(device, { useF16 = true } = {}) {
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
    this.validationError = null;
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
  }

  _uniform(words) {
    const size = Math.max(16, Math.ceil(words.byteLength / 16) * 16);
    let buf = null;
    for (let i = this.uFree.length - 1; i >= 0; i--) {
      if (this.uFree[i].size >= size) {
        buf = this.uFree.splice(i, 1)[0];
        break;
      }
    }
    if (!buf) {
      buf = this.device.createBuffer({
        size: Math.max(208, size),
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
    if (this.destroyed) throw Error('GPU backend destroyed');
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
    const u = new Uint32Array(4);
    const f = new Float32Array(u.buffer);
    u[0] = out.size; u[1] = 1; f[2] = value; u[3] = 0;
    const [gx, gy] = fold2(Math.ceil(out.size / 256));
    this._dispatch('ff2-gelu-fill', FF2_GELU_FILL_SOURCE, u, [this.dummy, out.buffer], gx, gy);
    return out;
  }

  fastGelu(x) {
    const out = this.alloc(x.shape);
    const u = new Uint32Array(4);
    u[0] = x.size; u[1] = 0; u[2] = 0; u[3] = 0;
    const [gx, gy] = fold2(Math.ceil(x.size / 256));
    this._dispatch('ff2-gelu-fill', FF2_GELU_FILL_SOURCE, u, [x.buffer, out.buffer], gx, gy);
    return out;
  }

  fastEmbedFeatures(seqIdxTensor, time, sc, L) {
    this._ensureAux();
    const f32 = Math.fround;
    const te = new Float32Array(32);
    for (let c = 0; c < 32; c++) {
      const a = f32(f32(f32(time) * 10000) * f32(Math.exp(f32((c % 16) * f32(-Math.log(10000) / 15)))));
      te[c] = f32(c < 16 ? Math.sin(a) : Math.cos(a));
    }
    const buildUniform = (mode, total) => {
      const u = new Uint32Array(52);
      const f = new Float32Array(u.buffer);
      u[0] = L; u[1] = mode; u[2] = total; u[3] = 0;
      f.set(te, 4);
      f.set(POS_DENOM, 36);
      return u;
    };
    const single = this.alloc([L, 65]);
    const [sx, sy] = fold2(Math.ceil((L * 65) / 256));
    this._dispatch('ff2-embed-feat', FF2_EMBED_FEATURES_SOURCE, buildUniform(0, L * 65), [seqIdxTensor.buffer, this.dummy, single.buffer], sx, sy);

    const edge = this.alloc([L, L, 120]);
    const [ex, ey] = fold2(Math.ceil((L * L * 120) / 256));
    this._dispatch('ff2-embed-feat', FF2_EMBED_FEATURES_SOURCE, buildUniform(1, L * L * 120), [seqIdxTensor.buffer, sc.buffer, edge.buffer], ex, ey);
    return [single, edge];
  }

  fastSo3Dq(rel, alpha) {
    const L = rel.shape[0];
    const out = this.alloc([L, 4]);
    const u = new Uint32Array(4);
    const f = new Float32Array(u.buffer);
    u[0] = L; f[1] = alpha; u[2] = 0; u[3] = 0;
    const [gx, gy] = fold2(Math.ceil((L * 4) / 64));
    this._dispatch('ff2-so3-dq', FF2_SO3_DQ_SOURCE, u, [rel.buffer, out.buffer], gx, gy);
    return out;
  }

  uploadWeight(t) {
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
    this.device.queue.writeBuffer(buffer, 0, packed);
    this.buffers.add(buffer);
    this.persistent.add(buffer);
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
    if (w.shape[1] !== K) throw Error('Linear shape mismatch');
    const out = this.alloc([...x.shape.slice(0, -1), N]);
    const wMode = this._weightMode(w);

    if (M <= 4 || N <= 8) {
      const u = new Uint32Array([M, N, K, b ? 1 : 0, w.scaleOffset ?? 0, 0, 0, 0]);
      const [gx, gy] = fold2(Math.ceil((M * N) / 64));
      const key = `gemv:${wMode}`;
      this._dispatch(key, gemvSource(wMode), u, [x.buffer, w.buffer, b?.buffer ?? this.dummy, out.buffer], gx, gy);
      return out;
    }

    const tileM = M >= 192 ? 64 : 32;
    const tileN = N <= 48 ? 32 : N <= 96 ? 64 : 128;
    const u = new Uint32Array(24);
    const f = new Float32Array(u.buffer);
    u[0] = M; u[1] = N; u[2] = K; u[3] = 1;
    u[4] = 0; u[5] = 0; u[6] = 0; u[7] = K;
    u[8] = 1; u[9] = 0; u[10] = 0; u[11] = 0;
    u[12] = K; u[13] = 1; u[14] = 0; u[15] = 0;
    u[16] = 0; u[17] = N; u[18] = 1; u[19] = b ? 1 : 0;
    u[20] = 0; f[21] = 1.0; u[22] = 1; u[23] = w.scaleOffset ?? 0;
    const groups = Math.ceil(M / tileM) * Math.ceil(N / tileN);
    const [gx, gy] = fold2(groups);
    const key = `gemm:${tileM}x${tileN}:${wMode}`;
    this._dispatch(key, gemmSource({ tileM, tileN, wMode }), u, [x.buffer, w.buffer, b?.buffer ?? this.dummy, out.buffer], gx, gy);
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
        if (batchAxes.length <= 2 && mAxes.length === 1 && nAxes.length === 1 && kAxes.length === 1) {
          const M = dims[mAxes[0]];
          const N = dims[nAxes[0]];
          const K = dims[kAxes[0]];
          if (M >= 8 && N >= 8 && K >= 8) {
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
            const tileM = M >= 192 ? 64 : 32;
            const tileN = N <= 48 ? 32 : N <= 96 ? 64 : 128;
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
            const groups = Math.ceil(M / tileM) * Math.ceil(N / tileN) * B0 * B1;
            const [gx, gy] = fold2(groups);
            const key = `gemm:${tileM}x${tileN}:F32`;
            this._dispatch(key, gemmSource({ tileM, tileN, wMode: 'F32' }), u, [xs[0].buffer, xs[1].buffer, this.dummy, out.buffer], gx, gy);
            return out;
          }
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
      const u = new Uint32Array(8);
      const f = new Float32Array(u.buffer);
      u[0] = rows; u[1] = N; u[2] = N; u[3] = 0;
      u[4] = w ? 1 : 0; u[5] = 0; f[6] = eps; u[7] = 0;
      const [gx, gy] = fold2(rows);
      this._dispatch('ff2-cascade-norm', FF2_CASCADE_NORM_SOURCE, u, [x.buffer, w?.buffer ?? this.dummy, b?.buffer ?? this.dummy, out.buffer], gx, gy);
      return out;
    }
    return super.norm(x, w, b, eps);
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
      (wgsl.includes('rfErf(') ? erfWGSL : '') +
      (/\b(sin|cos)\(/.test(wgsl) ? trigWGSL : '') +
      ACCURATE_DIV_SQRT_WGSL +
      declarations +
      `@group(0) @binding(${inputs.length}) var<storage,read_write> o:array<f32>;\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x+gid.y*65535u*64u;if(i>=${size}u){return;}${rewrite(wgsl).replaceAll('sqrt(', 'rfSqrt(').replaceAll('sin(', 'rfSin(').replaceAll('cos(', 'rfCos(')}}`;
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

  groupInverse(mom, eps) {
    const shape = [mom.shape[0]];
    const epsLo = eps - Math.fround(eps);
    const x = this.kernel(shape, [mom], '', `o[i]=a0[i*2u+1u]+${eps};`);
    const r = this.kernel(shape, [x], '', 'o[i]=rfdiv(1.0,sqrt(a0[i]));');
    const square = this.mul(r, r);
    return this.kernel(
      shape,
      [mom, x, r, square],
      '',
      `let v=a0[i*2u+1u];let x=a1[i];let r=a2[i];let r2=a3[i];let low=((v-x)+${eps})+${epsLo};let r2low=fma(r,r,-r2);let error=(fma(-r2,x,1.0)-r2low*x)-r2*low;o[i]=r+(0.5*r)*error;`,
    );
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
    this.flush();
    const b = this.device.createBuffer({
      size: Math.max(4, t.size * 4),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    try {
      const e = this.device.createCommandEncoder();
      e.copyBufferToBuffer(t.buffer, 0, b, 0, t.size * 4);
      this.device.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ);
      if (this.validationError) throw this.validationError;
      return new Float32Array(b.getMappedRange()).slice(0, t.size);
    } finally {
      b.destroy();
    }
  }

  async readMany(tensors) {
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
    return { bytes, transientBuffers };
  }
}
