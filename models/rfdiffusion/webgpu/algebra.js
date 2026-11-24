import { EXP_TABLE, numericWGSL } from "../shared/numeric.js";
import constants from "../shared/constants.js";
import { Algebra, sizeOf, stridesOf } from "../shared/tensor.js";
import {
  foldGrid2D,
  gemm64x128Source,
  gemm16x64Source,
  gemm64x16Source,
  gemm32x32Source,
  SE3_EOC_ECK_SOURCE,
  NORM_SOURCE,
  SOFTMAX_SOURCE,
  EDGE_SOFTMAX_SOURCE,
  permuteOpSource,
  BINARY_OP_SOURCE,
  UNARY_OP_SOURCE,
  SLICE_OP_SOURCE,
  gatherOpSource,
  SUM_AXIS0_SOURCE,
  RBF_FUSED_SOURCE,
} from "./kernels.js";

const SLAB_SLOTS = 4096;
const SLOT_BYTES = 256;
const SLAB_BYTES = SLAB_SLOTS * SLOT_BYTES;
const FLUSH_DISPATCH_LIMIT = 1024;

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);
function f32Bits(x) {
  _f32[0] = x;
  return _u32[0];
}

export class GPUAlgebra extends Algebra {
  constructor(device, { useF16 = false } = {}) {
    super();
    this.device = device;
    this.useF16 = Boolean(useF16 && device.features?.has?.("shader-f16"));
    this.buffers = new Set();
    this.persistent = new Set();
    this.internalTransient = new Set();
    this.usedInEncoder = new Set();
    this.pipelines = new Map();
    this.encoder = null;
    this.pass = null;
    this.dispatches = 0;
    this.pool = new Map();
    this.available = new Set();
    this.slab = null;
    this.dummyBuf = null;
    this.edgeTableBuf = null;
    this.rbfMuBuf = null;
    this.readbackBuf = null;
    this.readbackBytes = 0;
    this.validationError = null;
    this.listener = (e) => {
      this.validationError = e.error;
    };
    device.addEventListener("uncapturederror", this.listener);
  }

  _dummy() {
    if (!this.dummyBuf) {
      this.dummyBuf = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.buffers.add(this.dummyBuf);
      this.internalTransient.add(this.dummyBuf);
    }
    return this.dummyBuf;
  }

  _edgeTables() {
    if (!this.edgeTableBuf) {
      const f = Math.fround;
      const arr = new Float32Array(64);
      for (let i = 0; i < 32; i++) {
        const hi = f(EXP_TABLE[i]);
        arr[i] = hi;
        arr[32 + i] = f(EXP_TABLE[i] - hi);
      }
      this.edgeTableBuf = this.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(this.edgeTableBuf, 0, arr);
      this.buffers.add(this.edgeTableBuf);
      this.internalTransient.add(this.edgeTableBuf);
    }
    return this.edgeTableBuf;
  }

  _rbfMu() {
    if (!this.rbfMuBuf) {
      const arr = Float32Array.from(constants.rbfMu);
      this.rbfMuBuf = this.device.createBuffer({
        size: arr.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(this.rbfMuBuf, 0, arr);
      this.buffers.add(this.rbfMuBuf);
      this.internalTransient.add(this.rbfMuBuf);
    }
    return this.rbfMuBuf;
  }

  _allocUniform(words) {
    if (this.slab && this.slab.slot >= SLAB_SLOTS) {
      this.flush();
    }
    if (!this.slab) {
      const gpu = this.device.createBuffer({
        size: SLAB_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const cpuBytes = new Uint8Array(SLAB_BYTES);
      this.slab = {
        gpu,
        cpuBytes,
        cpuU32: new Uint32Array(cpuBytes.buffer),
        slot: 0,
      };
      this.buffers.add(gpu);
      this.internalTransient.add(gpu);
    }
    const slot = this.slab.slot++;
    const byteOff = slot * SLOT_BYTES;
    const u32View =
      words instanceof Uint32Array
        ? words
        : new Uint32Array(words.buffer, words.byteOffset, words.byteLength / 4);
    this.slab.cpuU32.set(u32View, byteOff >> 2);
    const bindSize = Math.max(16, Math.ceil(words.byteLength / 16) * 16);
    return { buffer: this.slab.gpu, offset: byteOff, size: bindSize };
  }

  _pipeline(key, source) {
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code: source });
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  _dispatchUniform(key, source, groups, buffers, uniformWords) {
    if (this.slab && this.slab.slot >= SLAB_SLOTS) {
      this.flush();
    }
    const pipeline = this._pipeline(key, source);
    const uRes = this._allocUniform(uniformWords);
    const entries = [{ binding: 0, resource: uRes }];
    for (let i = 0; i < buffers.length; i++) {
      const buf = buffers[i];
      entries.push({ binding: i + 1, resource: { buffer: buf } });
      this.usedInEncoder.add(buf);
    }
    this.usedInEncoder.add(uRes.buffer);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    this.encoder ??= this.device.createCommandEncoder();
    this.pass ??= this.encoder.beginComputePass();
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, bindGroup);
    const [gx, gy] = foldGrid2D(groups);
    this.pass.dispatchWorkgroups(gx, gy);
    if (++this.dispatches >= FLUSH_DISPATCH_LIMIT) this.flush();
  }

  alloc(shape, persistent = false, hostWrite = false) {
    const size = sizeOf(shape),
      bytes = Math.max(4, size * 4),
      limit = Math.min(
        this.device.limits.maxBufferSize,
        this.device.limits.maxStorageBufferBindingSize,
      );
    if (bytes > limit) throw Error(`GPU buffer ${bytes} exceeds ${limit}`);
    let buffer = null;
    if (!persistent) {
      const list = this.pool.get(bytes);
      if (list && list.length) {
        if (!hostWrite) {
          buffer = list.pop();
        } else {
          for (let i = list.length - 1; i >= 0; i--) {
            if (!this.usedInEncoder.has(list[i])) {
              buffer = list[i];
              list.splice(i, 1);
              break;
            }
          }
        }
      }
    }
    if (!buffer) {
      buffer = this.device.createBuffer({
        size: bytes,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
    }
    this.available.delete(buffer);
    this.buffers.add(buffer);
    if (persistent) this.persistent.add(buffer);
    return { shape, size, buffer };
  }

  upload(shape, data, persistent = false) {
    const out = this.alloc(shape, persistent, true);
    const arr =
      data instanceof Float32Array || data instanceof Uint8Array
        ? data
        : Float32Array.from(data);
    this.device.queue.writeBuffer(out.buffer, 0, arr);
    return out;
  }

  tensor(shape, data) {
    return this.upload(
      shape,
      data instanceof Float32Array ? data : Float32Array.from(data),
    );
  }

  constant(shape, value = 0) {
    const out = this.alloc(shape);
    const u = new Uint32Array([out.size, 4, f32Bits(value), 0]);
    this._dispatchUniform(
      "unary-op",
      UNARY_OP_SOURCE,
      Math.ceil(out.size / 256),
      [this._dummy(), out.buffer],
      u,
    );
    return out;
  }

  uploadWeight(t) {
    if (t.dtype === "F32") return this.upload(t.shape, t.data, true);
    const rawBytes = t.data.byteLength,
      scaleOffset = Math.ceil(rawBytes / 4),
      bytes = (scaleOffset + (t.scale?.length ?? 0)) * 4,
      limit = Math.min(
        this.device.limits.maxBufferSize,
        this.device.limits.maxStorageBufferBindingSize,
      );
    if (bytes > limit) throw Error("Weight exceeds GPU limits");
    const packed = new Uint8Array(bytes);
    packed.set(
      new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength),
    );
    if (t.scale) new Float32Array(packed.buffer, scaleOffset * 4).set(t.scale);
    const buffer = this.device.createBuffer({
      size: bytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
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

  // === Core Tiled GEMM & Linear ===

  _gemm({
    a,
    b,
    bias = null,
    out = null,
    outShape,
    m,
    n,
    k,
    batches = 1,
    sA0 = 0,
    sB0 = 0,
    sC0 = 0,
    act = 0,
    alpha = 1.0,
    accMode = 0,
  }) {
    const dtype = b.dtype ?? "F32";
    const useF16 = dtype === "F16" && this.useF16;
    const target = out ?? this.alloc(outShape);
    const u = new Uint32Array([
      m,
      n,
      k,
      batches,
      sA0,
      sB0,
      sC0,
      b.scaleOffset ?? 0,
      bias ? 1 : 0,
      act,
      f32Bits(alpha),
      accMode,
      0,
      0,
      0,
      0,
    ]);
    const buffers = [
      a.buffer,
      b.buffer,
      bias ? bias.buffer : this._dummy(),
      target.buffer,
    ];

    if (dtype === "F32" && batches > 1 && m <= 128 && n <= 128) {
      const groups = batches * Math.ceil(m / 32) * Math.ceil(n / 32);
      this._dispatchUniform("gemm32x32", gemm32x32Source(), groups, buffers, u);
      return target;
    }
    if (n <= 32 && m > 32) {
      const key = `gemm64x16:${dtype}:${useF16 ? 1 : 0}`;
      const groups = batches * Math.ceil(m / 64) * Math.ceil(n / 16);
      this._dispatchUniform(
        key,
        gemm64x16Source({ dtype, useF16 }),
        groups,
        buffers,
        u,
      );
      return target;
    }
    if (m <= 96) {
      const key = `gemm16x64:${dtype}:${useF16 ? 1 : 0}`;
      const groups = batches * Math.ceil(m / 16) * Math.ceil(n / 64);
      this._dispatchUniform(
        key,
        gemm16x64Source({ dtype, useF16 }),
        groups,
        buffers,
        u,
      );
      return target;
    }
    const key = `gemm64x128:${dtype}:${useF16 ? 1 : 0}`;
    const groups = batches * Math.ceil(m / 64) * Math.ceil(n / 128);
    this._dispatchUniform(
      key,
      gemm64x128Source({ dtype, useF16 }),
      groups,
      buffers,
      u,
    );
    return target;
  }

  linear(x, w, b, { act = 0 } = {}) {
    const K = x.shape.at(-1),
      N = w.shape[0],
      M = x.size / K,
      outShape = [...x.shape.slice(0, -1), N];
    return this._gemm({
      a: x,
      b: w,
      bias: b ?? null,
      outShape,
      m: M,
      n: N,
      k: K,
      act,
    });
  }

  linearRelu(x, w, b) {
    return this.linear(x, w, b, { act: 1 });
  }

  se3Contraction(a, b) {
    const [E, O, C] = a.shape;
    const K = b.shape[2];
    const out = this.alloc([E, O, K]);
    const tilesO = Math.ceil(O / 64);
    const edgeGroups = Math.ceil(E / 4);
    this._dispatchUniform(
      "se3-eoc-eck",
      SE3_EOC_ECK_SOURCE,
      edgeGroups * tilesO,
      [a.buffer, b.buffer, out.buffer],
      new Uint32Array([E, O, C, K]),
    );
    return out;
  }

  permuteTensor(x, order) {
    const rank = order.length;
    const outShape = order.map((k) => x.shape[k]);
    const isIdentity =
      (!x.dtype || x.dtype === "F32") && order.every((k, i) => k === i);
    if (isIdentity) return this.reshape(x, outShape);
    if (rank > 6) return super.permute(x, order);
    const out = this.alloc(outShape);
    const outSt = stridesOf(outShape);
    const xSt = stridesOf(x.shape);
    const inSt = order.map((k) => xSt[k]);
    const u = new Uint32Array(28);
    u[0] = out.size;
    u[1] = rank;
    u[2] = x.scaleOffset ?? 0;
    u[3] = x.shape[1] ?? 1;
    for (let d = 0; d < rank; d++) {
      u[4 + d] = outShape[d];
      u[12 + d] = outSt[d];
      u[20 + d] = inSt[d];
    }
    const dtype = x.dtype ?? "F32";
    this._dispatchUniform(
      `permute:${dtype}`,
      permuteOpSource(dtype),
      Math.ceil(out.size / 256),
      [x.buffer, out.buffer],
      u,
    );
    return out;
  }

  permute(x, order) {
    return this.permuteTensor(x, order);
  }

  sumAxis0(x) {
    const A = x.shape[0];
    const outShape = x.shape.slice(1);
    const M = sizeOf(outShape);
    const out = this.alloc(outShape);
    const u = new Uint32Array([A, M, 0, 0]);
    this._dispatchUniform(
      "sum-axis0",
      SUM_AXIS0_SOURCE,
      Math.ceil(M / 256),
      [x.buffer, out.buffer],
      u,
    );
    return out;
  }

  einsum(eq, ...xs) {
    const clean = eq.replaceAll(" ", ""),
      [left, right] = clean.split("->"),
      terms = left.split(","),
      dims = {};
    terms.forEach((s, j) => {
      if (s.length !== xs[j].shape.length) throw Error(`einsum rank ${eq}`);
      [...s].forEach((c, k) => {
        const d = xs[j].shape[k];
        if (dims[c] && dims[c] !== d) throw Error(`einsum dimension ${eq}`);
        dims[c] = d;
      });
    });

    if (terms.length === 1) {
      const s0 = terms[0];
      if (
        s0.length === right.length &&
        new Set(s0).size === s0.length &&
        new Set(right).size === right.length &&
        [...right].every((c) => s0.includes(c))
      ) {
        return this.permuteTensor(
          xs[0],
          [...right].map((c) => s0.indexOf(c)),
        );
      }
      if (
        s0.length >= 2 &&
        s0.slice(1) === right &&
        !right.includes(s0[0]) &&
        new Set(s0).size === s0.length &&
        (!xs[0].dtype || xs[0].dtype === "F32")
      ) {
        return this.sumAxis0(xs[0]);
      }
      return super.einsum(eq, ...xs);
    }

    if (terms.length === 2) {
      if (
        clean === "eoc,eck->eok" &&
        !xs[0].dtype &&
        !xs[1].dtype &&
        xs[1].shape[2] <= 4
      ) {
        return this.se3Contraction(xs[0], xs[1]);
      }
      const sA = [...terms[0]],
        sB = [...terms[1]],
        sOut = [...right];
      if (
        new Set(sA).size === sA.length &&
        new Set(sB).size === sB.length &&
        new Set(sOut).size === sOut.length
      ) {
        const kAxes = Object.keys(dims)
          .filter((c) => !sOut.includes(c))
          .sort();
        const batchAxes = sOut.filter((c) => sA.includes(c) && sB.includes(c));
        const mAxes = sOut.filter((c) => sA.includes(c) && !sB.includes(c));
        const nAxes = sOut.filter((c) => !sA.includes(c) && sB.includes(c));
        if (
          kAxes.length >= 1 &&
          kAxes.every((c) => sA.includes(c) && sB.includes(c)) &&
          sA.length === batchAxes.length + mAxes.length + kAxes.length &&
          sB.length === batchAxes.length + nAxes.length + kAxes.length &&
          sOut.length === batchAxes.length + mAxes.length + nAxes.length
        ) {
          const Bat = sizeOf(batchAxes.map((c) => dims[c])),
            M0 = sizeOf(mAxes.map((c) => dims[c])),
            N0 = sizeOf(nAxes.map((c) => dims[c])),
            K = sizeOf(kAxes.map((c) => dims[c]));
          if (Bat * M0 * N0 * K >= 2048 && M0 >= 4 && N0 >= 4) {
            let X = xs[0],
              sX = sA,
              xAxes = mAxes;
            let Y = xs[1],
              sY = sB,
              yAxes = nAxes;
            if (X.dtype && X.dtype !== "F32" && (!Y.dtype || Y.dtype === "F32")) {
              [X, sX, xAxes, Y, sY, yAxes] = [Y, sY, yAxes, X, sX, xAxes];
            }
            const targetX = [...batchAxes, ...xAxes, ...kAxes];
            if (sX.join("") !== targetX.join("") || (X.dtype && X.dtype !== "F32")) {
              X = this.permuteTensor(
                X,
                targetX.map((c) => sX.indexOf(c)),
              );
            }
            const targetY = [...batchAxes, ...yAxes, ...kAxes];
            if (
              sY.join("") !== targetY.join("") ||
              (batchAxes.length > 0 && Y.dtype === "I8")
            ) {
              Y = this.permuteTensor(
                Y,
                targetY.map((c) => sY.indexOf(c)),
              );
            }
            const Mx = sizeOf(xAxes.map((c) => dims[c]));
            const Ny = sizeOf(yAxes.map((c) => dims[c]));
            const gemmAxes = [...batchAxes, ...xAxes, ...yAxes];
            const gemmShape = gemmAxes.map((c) => dims[c]);
            const gemmOut = this._gemm({
              a: X,
              b: Y,
              outShape: gemmShape,
              m: Mx,
              n: Ny,
              k: K,
              batches: Bat,
              sA0: batchAxes.length ? Mx * K : 0,
              sB0: batchAxes.length ? Ny * K : 0,
              sC0: batchAxes.length ? Mx * Ny : 0,
            });
            if (gemmAxes.join("") !== sOut.join("")) {
              return this.permuteTensor(
                gemmOut,
                sOut.map((c) => gemmAxes.indexOf(c)),
              );
            }
            return gemmOut;
          }
        }
      }
    }
    return super.einsum(eq, ...xs);
  }

  // === Elementwise, Slice, Gather, Norm, Softmax ===

  _binaryOp(op, a, b) {
    if (typeof a === "number" && typeof b !== "number") {
      const out = this.alloc(b.shape);
      const u = new Uint32Array(28);
      u[0] = out.size;
      u[1] = op;
      u[2] = 2;
      u[4] = f32Bits(a);
      this._dispatchUniform(
        "binary-op",
        BINARY_OP_SOURCE,
        Math.ceil(out.size / 256),
        [b.buffer, this._dummy(), out.buffer],
        u,
      );
      return out;
    }
    if (typeof b === "number" && typeof a !== "number") {
      const out = this.alloc(a.shape);
      const u = new Uint32Array(28);
      u[0] = out.size;
      u[1] = op;
      u[2] = 1;
      u[4] = f32Bits(b);
      this._dispatchUniform(
        "binary-op",
        BINARY_OP_SOURCE,
        Math.ceil(out.size / 256),
        [a.buffer, this._dummy(), out.buffer],
        u,
      );
      return out;
    }
    const rank = Math.max(a.shape.length, b.shape.length);
    if (rank > 5 || a.dtype || b.dtype) {
      const expr = ["$0+$1", "$0-$1", "$0*$1", "rfdiv($0,$1)"][op];
      return this.element(expr, a, b);
    }
    const sameShape =
      a.shape.length === b.shape.length &&
      a.shape.every((d, k) => d === b.shape[k]);
    const outShape = Array(rank).fill(1);
    for (const t of [a, b]) {
      t.shape.forEach((d, j) => {
        const k = rank - t.shape.length + j;
        if (outShape[k] !== 1 && d !== 1 && outShape[k] !== d)
          throw Error("broadcast");
        outShape[k] = Math.max(outShape[k], d);
      });
    }
    const out = this.alloc(outShape);
    const u = new Uint32Array(28);
    u[0] = out.size;
    u[1] = op;
    u[2] = 0;
    u[3] = sameShape ? 0 : rank;
    if (!sameShape) {
      const aPad = Array(rank - a.shape.length)
        .fill(1)
        .concat(a.shape);
      const bPad = Array(rank - b.shape.length)
        .fill(1)
        .concat(b.shape);
      const outSt = stridesOf(outShape);
      const aStRaw = stridesOf(aPad);
      const bStRaw = stridesOf(bPad);
      for (let d = 0; d < Math.min(rank, 4); d++) {
        u[8 + d] = outShape[d];
        u[12 + d] = outSt[d];
        u[16 + d] = aPad[d] === 1 ? 0 : aStRaw[d];
        u[20 + d] = bPad[d] === 1 ? 0 : bStRaw[d];
      }
      if (rank === 5) {
        u[24] = outShape[4];
        u[25] = outSt[4];
        u[26] = aPad[4] === 1 ? 0 : aStRaw[4];
        u[27] = bPad[4] === 1 ? 0 : bStRaw[4];
      }
    }
    this._dispatchUniform(
      "binary-op",
      BINARY_OP_SOURCE,
      Math.ceil(out.size / 256),
      [a.buffer, b.buffer, out.buffer],
      u,
    );
    return out;
  }

  add(a, b) {
    return this._binaryOp(0, a, b);
  }
  sub(a, b) {
    return this._binaryOp(1, a, b);
  }
  mul(a, b) {
    return this._binaryOp(2, a, b);
  }
  div(a, b) {
    return this._binaryOp(3, a, b);
  }
  gatedMul(a, b) {
    return this._binaryOp(4, a, b);
  }

  relu(a) {
    const out = this.alloc(a.shape);
    const u = new Uint32Array([out.size, 0, 0, 0]);
    this._dispatchUniform(
      "unary-op",
      UNARY_OP_SOURCE,
      Math.ceil(out.size / 256),
      [a.buffer, out.buffer],
      u,
    );
    return out;
  }

  sigmoid(a) {
    const out = this.alloc(a.shape);
    const u = new Uint32Array([out.size, 1, 0, 0]);
    this._dispatchUniform(
      "unary-op",
      UNARY_OP_SOURCE,
      Math.ceil(out.size / 256),
      [a.buffer, out.buffer],
      u,
    );
    return out;
  }

  slice(x, axis, start, end) {
    if (axis < 0) axis += x.shape.length;
    if (x.dtype && x.dtype !== "F32") return super.slice(x, axis, start, end);
    const shape = x.shape.slice();
    shape[axis] = end - start;
    const inner = sizeOf(x.shape.slice(axis + 1));
    const out = this.alloc(shape);
    const u = new Uint32Array([
      out.size,
      x.shape[axis],
      shape[axis],
      inner,
      start,
      0,
      0,
      0,
    ]);
    this._dispatchUniform(
      "slice-op",
      SLICE_OP_SOURCE,
      Math.ceil(out.size / 256),
      [x.buffer, out.buffer],
      u,
    );
    return out;
  }

  gather(x, index) {
    const width = sizeOf(x.shape.slice(1));
    const outShape = [...index.shape, ...x.shape.slice(1)];
    const out = this.alloc(outShape);
    const dtype = x.dtype ?? "F32";
    const u = new Uint32Array([out.size, width, x.scaleOffset ?? 0, 0]);
    this._dispatchUniform(
      `gather:${dtype}`,
      gatherOpSource(dtype),
      Math.ceil(out.size / 256),
      [x.buffer, index.buffer, out.buffer],
      u,
    );
    return out;
  }

  _normImpl(x, w, b, isGroup, eps, hasRelu = false) {
    const width = x.shape.at(-1);
    const rows = x.size / width;
    const out = this.alloc(x.shape);
    const u = new Uint32Array(8);
    const f = new Float32Array(u.buffer);
    u[0] = rows;
    u[1] = width;
    u[2] = isGroup ? 1 : 0;
    u[3] = hasRelu ? 1 : 0;
    f[4] = eps;
    f[5] = eps - Math.fround(eps);
    this._dispatchUniform(
      "norm-op",
      NORM_SOURCE,
      rows,
      [x.buffer, w.buffer, b.buffer, out.buffer],
      u,
    );
    return out;
  }

  norm(x, w, b, eps = 1e-5) {
    return this._normImpl(x, w, b, false, eps, false);
  }

  normRelu(x, w, b, eps = 1e-5) {
    return this._normImpl(x, w, b, false, eps, true);
  }

  groupNorm(x, w, b, eps = 1e-5) {
    return this._normImpl(x, w, b, true, eps, false);
  }

  groupNormRelu(x, w, b, eps = 1e-5) {
    return this._normImpl(x, w, b, true, eps, true);
  }

  softmax(x, axis = -1, dgl = false) {
    if (dgl) {
      if (axis !== 0) throw Error("Edge softmax expects incoming edges on axis 0");
      const n = x.shape[0];
      const inner = x.size / n;
      const out = this.alloc(x.shape);
      const u = new Uint32Array([n, inner, 0, 0]);
      this._dispatchUniform(
        "edge-softmax-op",
        EDGE_SOFTMAX_SOURCE,
        inner,
        [x.buffer, this._edgeTables(), out.buffer],
        u,
      );
      return out;
    }
    if (axis < 0) axis += x.shape.length;
    const n = x.shape[axis],
      inner = sizeOf(x.shape.slice(axis + 1)),
      outer = sizeOf(x.shape.slice(0, axis)),
      slices = outer * inner;
    const out = this.alloc(x.shape);
    const u = new Uint32Array([slices, n, inner, 0]);
    this._dispatchUniform(
      "softmax-op",
      SOFTMAX_SOURCE,
      slices,
      [x.buffer, out.buffer],
      u,
    );
    return out;
  }

  // === RFdiffusion Domain Fusions ===

  rbfFused(dist) {
    const outShape = [...dist.shape, 36];
    const out = this.alloc(outShape);
    const u = new Uint32Array(4);
    const f = new Float32Array(u.buffer);
    u[0] = out.size;
    f[1] = 20 / 36;
    this._dispatchUniform(
      "rbf-fused",
      RBF_FUSED_SOURCE,
      Math.ceil(out.size / 256),
      [dist.buffer, this._rbfMu(), out.buffer],
      u,
    );
    return out;
  }

  // === Legacy Scalar Kernel Fallback ===

  kernel(shape, inputs, js, wgsl) {
    const size = sizeOf(shape);
    const rewrite = (code) => {
      let result = "",
        last = 0;
      const re = /a(\d+)\[/g;
      let match;
      while ((match = re.exec(code))) {
        let depth = 1,
          end = re.lastIndex;
        while (depth && end < code.length) {
          if (code[end] === "[") depth++;
          if (code[end] === "]") depth--;
          end++;
        }
        result +=
          code.slice(last, match.index) +
          `load${match[1]}(${rewrite(code.slice(re.lastIndex, end - 1))})`;
        last = end;
        re.lastIndex = end;
      }
      return result + code.slice(last);
    };
    const declarations = inputs
      .map((t, j) => {
        const raw = t.dtype && t.dtype !== "F32",
          binding = `@group(0) @binding(${j}) var<storage,read> input${j}:array<${raw ? "u32" : "f32"}>;`;
        let expression = `input${j}[index]`;
        if (t.dtype === "F16")
          expression = `select(unpack2x16float(input${j}[index/2u]).x,unpack2x16float(input${j}[index/2u]).y,(index%2u)!=0u)`;
        if (t.dtype === "I8")
          expression = `f32((i32(input${j}[index/4u]>>((index%4u)*8u))<<24u)>>24u)*bitcast<f32>(input${j}[${t.scaleOffset}u+index/${t.shape[1]}u])`;
        return binding + `fn load${j}(index:u32)->f32{return ${expression};}`;
      })
      .join("\n");
    const accurateMath = `fn rfdiv(a:f32,b:f32)->f32 {
 let signBit=(bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u;
 let x=abs(a);let y=abs(b);var bits=bitcast<u32>(x/y);
 if(bits==0u || bits>=0x7f800000u){return a/b;}
 for(var k=0u;k<4u;k++){
  let q=bitcast<f32>(bits);let e=fma(-q,y,x);
  let up=e>0.0;let nextBits=select(bits-1u,bits+1u,up);
  let halfGap=abs(bitcast<f32>(nextBits)-q)*0.5;
  let boundary=halfGap*y;
  if(abs(e)>boundary || (abs(e)==boundary && (bits&1u)==1u)){bits=nextBits;}else{break;}
 }
 return bitcast<f32>(bits|signBit);
}
fn rfSqrt(x:f32)->f32 {let y=sqrt(x);if(x==0.0){return 0.0;}return y+fma(-y,y,x)/(2.0*y);}
`;
    const code =
      numericWGSL +
      accurateMath +
      declarations +
      `@group(0) @binding(${inputs.length}) var<storage,read_write> o:array<f32>;\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x+gid.y*65535u*64u;if(i>=${size}u){return;}${rewrite(wgsl).replaceAll("sqrt(", "rfSqrt(")}}`;
    const pipeline = this._pipeline(code, code);
    const out = this.alloc(shape);
    const all = [...inputs, out];
    for (const t of all) this.usedInEncoder.add(t.buffer);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: all.map((t, binding) => ({
        binding,
        resource: { buffer: t.buffer },
      })),
    });
    this.encoder ??= this.device.createCommandEncoder();
    this.pass ??= this.encoder.beginComputePass();
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, bindGroup);
    const [gx, gy] = foldGrid2D(Math.ceil(size / 64));
    this.pass.dispatchWorkgroups(gx, gy);
    if (++this.dispatches >= FLUSH_DISPATCH_LIMIT) this.flush();
    return out;
  }

  // === Execution & Memory Lifecycle ===

  flush() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    if (this.encoder) {
      if (this.slab && this.slab.slot > 0) {
        this.device.queue.writeBuffer(
          this.slab.gpu,
          0,
          this.slab.cpuBytes,
          0,
          this.slab.slot * SLOT_BYTES,
        );
        this.slab.slot = 0;
      }
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder = null;
      this.dispatches = 0;
      this.usedInEncoder.clear();
    }
  }

  _ensureReadback(bytes) {
    const aligned = Math.max(16, Math.ceil(bytes / 16) * 16);
    if (!this.readbackBuf || this.readbackBytes < aligned) {
      if (this.readbackBuf) {
        this.readbackBuf.destroy();
        this.buffers.delete(this.readbackBuf);
        this.internalTransient.delete(this.readbackBuf);
      }
      this.readbackBuf = this.device.createBuffer({
        size: aligned,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this.readbackBytes = aligned;
      this.buffers.add(this.readbackBuf);
      this.internalTransient.add(this.readbackBuf);
    }
    return this.readbackBuf;
  }

  async readMany(tensors) {
    let totalBytes = 0;
    const slices = tensors.map((t) => {
      const bytes = t.size * 4;
      const off = totalBytes;
      totalBytes += Math.max(4, Math.ceil(bytes / 4) * 4);
      return { t, off, bytes };
    });
    const rb = this._ensureReadback(Math.max(4, totalBytes));
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    this.encoder ??= this.device.createCommandEncoder();
    if (this.slab && this.slab.slot > 0) {
      this.device.queue.writeBuffer(
        this.slab.gpu,
        0,
        this.slab.cpuBytes,
        0,
        this.slab.slot * SLOT_BYTES,
      );
      this.slab.slot = 0;
    }
    for (const s of slices) {
      if (s.bytes > 0) {
        this.encoder.copyBufferToBuffer(s.t.buffer, 0, rb, s.off, s.bytes);
      }
    }
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    this.dispatches = 0;
    this.usedInEncoder.clear();

    await rb.mapAsync(GPUMapMode.READ, 0, Math.max(4, totalBytes));
    try {
      const mapped = rb.getMappedRange(0, Math.max(4, totalBytes));
      if (this.validationError) throw this.validationError;
      return slices.map((s) =>
        new Float32Array(mapped, s.off, s.t.size).slice(),
      );
    } finally {
      rb.unmap();
    }
  }

  async read(t) {
    const [arr] = await this.readMany([t]);
    return arr;
  }

  async checkpoint(keep = [], { sync = false } = {}) {
    if (sync || this.dispatches >= FLUSH_DISPATCH_LIMIT) {
      this.flush();
    }
    const live = new Set(
      keep
        .flat(Infinity)
        .filter(Boolean)
        .map((t) => t.buffer),
    );
    for (const b of this.buffers)
      if (
        !this.persistent.has(b) &&
        !this.internalTransient.has(b) &&
        !live.has(b)
      ) {
        if (!this.available.has(b)) {
          const list = this.pool.get(b.size) ?? [];
          list.push(b);
          this.pool.set(b.size, list);
          this.available.add(b);
        }
      }
    if (sync) {
      await this.device.queue.onSubmittedWorkDone();
    }
    if (this.validationError) throw this.validationError;
  }

  clear() {
    this.flush();
    this.pool.clear();
    this.available.clear();
    this.slab = null;
    this.dummyBuf = null;
    this.edgeTableBuf = null;
    this.rbfMuBuf = null;
    this.readbackBuf = null;
    this.readbackBytes = 0;
    this.internalTransient.clear();
    for (const b of this.buffers)
      if (!this.persistent.has(b)) {
        b.destroy();
        this.buffers.delete(b);
      }
  }

  destroy() {
    this.device.removeEventListener("uncapturederror", this.listener);
    this.pool.clear();
    this.available.clear();
    this.usedInEncoder.clear();
    this.pass = null;
    this.encoder = null;
    this.slab = null;
    this.dummyBuf = null;
    this.edgeTableBuf = null;
    this.rbfMuBuf = null;
    this.readbackBuf = null;
    this.readbackBytes = 0;
    this.internalTransient.clear();
    for (const b of this.buffers) b.destroy();
    this.buffers.clear();
    this.persistent.clear();
    this.pipelines.clear();
  }

  memoryUsage() {
    let bytes = 0,
      transientBuffers = 0;
    for (const b of this.buffers) {
      bytes += b.size;
      if (!this.persistent.has(b)) transientBuffers++;
    }
    return { bytes, transientBuffers };
  }
}
