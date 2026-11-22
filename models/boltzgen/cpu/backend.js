import {
  broadcast,
  physical,
  sizeOf,
  strides,
  tensor,
} from "../shared/tensor.js";
const f = Math.fround;
const scalar = (t, i = 0) =>
  typeof t === "number" ? t : t.storage.data[physical(t, i)];

export class CPUBackend {
  constructor() {
    this.storages = new Set();
  }
  upload(data, shape, persistent = true) {
    const storage = { data, refs: 0, persistent };
    this.storages.add(storage);
    return tensor(storage, shape);
  }
  retain(t) {
    t.storage.refs++;
  }
  release(t) {
    if (--t.storage.refs <= 0 && !t.storage.persistent)
      this.storages.delete(t.storage);
  }
  clearTransient() {
    for (const s of this.storages) if (!s.persistent) this.storages.delete(s);
  }
  destroy() {
    this.storages.clear();
  }
  async flush() {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  read(t) {
    const out = new Float32Array(sizeOf(t.shape));
    for (let i = 0; i < out.length; i++) out[i] = scalar(t, i);
    return out;
  }
  memoryUsage() {
    let persistentBytes = 0,
      transientBytes = 0;
    for (const s of this.storages) {
      if (s.persistent) persistentBytes += s.data.byteLength;
      else transientBytes += s.data.byteLength;
    }
    return {
      persistentBytes,
      transientBytes,
      totalBytes: persistentBytes + transientBytes,
    };
  }
  compute(kind, inputs, shapes, attrs = {}, inplace = false) {
    const outputs = shapes.map((s, i) =>
      inplace && i === 0
        ? attrs.target
        : this.upload(new Float32Array(sizeOf(s)), s, false),
    );
    const out = outputs[0],
      shape = shapes[0],
      count = sizeOf(shape),
      data = out.storage.data;
    const set = (i, x) => {
      data[physical(out, i)] = x;
    };
    const [a, b, c] = inputs;
    switch (kind) {
      case "onehot":
        for (let i = 0; i < count; i++)
          set(
            i,
            Math.floor(scalar(a, Math.floor(i / attrs.width))) ===
              i % attrs.width
              ? 1
              : 0,
          );
        break;
      case "window": {
        const d = shape[2],
          n = a.shape[1];
        for (let i = 0; i < count; i++) {
          const window = Math.floor(i / (128 * d)),
            key = Math.floor(i / d) % 128,
            j = window * 32 + key - 48;
          set(i, j >= 0 && j < n ? scalar(a, j * d + (i % d)) : 0);
        }
        break;
      }
      case "fill":
        data.fill(attrs.value);
        break;
      case "range":
        for (let i = 0; i < count; i++) set(i, attrs.start + i * attrs.step);
        break;
      case "eye":
        for (let i = 0; i < count; i++)
          set(i, Math.floor(i / attrs.width) === i % attrs.width ? 1 : 0);
        break;
      case "copy":
      case "cast": {
        const t = broadcast(a, shape);
        for (let i = 0; i < count; i++) {
          let v = scalar(t, i);
          if (kind === "cast") {
            if (attrs.dtype === "torch.bool") v = v !== 0 ? 1 : 0;
            else if (/torch.int/.test(attrs.dtype)) v = Math.trunc(v);
          }
          set(i, v);
        }
        break;
      }
      case "pointwise": {
        const ts = inputs.map((t) =>
          typeof t === "number" || t === null ? t : broadcast(t, shape),
        );
        for (let i = 0; i < count; i++) {
          const x = scalar(ts[0], i),
            y = ts.length > 1 && ts[1] !== null ? scalar(ts[1], i) : 0,
            z = ts.length > 2 && ts[2] !== null ? scalar(ts[2], i) : 0;
          let v;
          switch (attrs.kind) {
            case "add":
              v = x + y * attrs.alpha;
              break;
            case "sub":
              v = x - y * attrs.alpha;
              break;
            case "rsub":
              v = y - x * attrs.alpha;
              break;
            case "mul":
              v = x * y;
              break;
            case "div":
              v = x / y;
              break;
            case "pow":
              v = x ** y;
              break;
            case "sigmoid":
              v = 1 / (1 + Math.exp(-x));
              break;
            case "silu":
              v = x / (1 + Math.exp(-x));
              break;
            case "relu":
              v = Math.max(0, x);
              break;
            case "sqrt":
              v = Math.sqrt(x);
              break;
            case "exp":
              v = Math.exp(x);
              break;
            case "log":
              v = Math.log(x);
              break;
            case "cos":
              v = Math.cos(x);
              break;
            case "reciprocal":
              v = 1 / x;
              break;
            case "neg":
              v = -x;
              break;
            case "abs":
              v = Math.abs(x);
              break;
            case "eq":
              v = x === y;
              break;
            case "ne":
              v = x !== y;
              break;
            case "gt":
              v = x > y;
              break;
            case "lt":
              v = x < y;
              break;
            case "le":
              v = x <= y;
              break;
            case "bitwise_and":
              v = x & y;
              break;
            case "bitwise_not":
              v = !x;
              break;
            case "where":
              v = x ? y : z;
              break;
            case "clamp":
              v = Math.min(
                ts[2] == null ? Infinity : z,
                Math.max(ts[1] == null ? -Infinity : y, x),
              );
              break;
            default:
              throw Error(`Unsupported operation ${attrs.kind}`);
          }
          set(i, v);
        }
        break;
      }
      case "matmul": {
        const M = a.shape.at(-2),
          K = a.shape.at(-1),
          N = b.shape.at(-1),
          batch = count / (M * N),
          ad = a.storage.data,
          bd = b.storage.data;
        const as = a.stride,
          bs = b.stride,
          ar = as.at(-2),
          ak = as.at(-1),
          bk = bs.at(-2),
          bc = bs.at(-1);
        const bias = c ? broadcast(c, shape) : null;
        for (let h = 0; h < batch; h++) {
          const ao = physical(a, h * M * K),
            bo = physical(b, h * K * N);
          for (let m = 0; m < M; m++)
            for (let n = 0; n < N; n++) {
              let sum = 0;
              const ai = ao + m * ar,
                bi = bo + n * bc;
              for (let k = 0; k < K; k++)
                sum += ad[ai + k * ak] * bd[bi + k * bk];
              const idx = (h * M + m) * N + n;
              set(
                idx,
                sum * (attrs.alpha ?? 1) +
                  (bias ? scalar(bias, idx) * (attrs.beta ?? 1) : 0),
              );
            }
        }
        break;
      }
      case "norm": {
        const width = attrs.width,
          rows = count / width;
        for (let r = 0; r < rows; r++) {
          let mean = 0;
          for (let j = 0; j < width; j++) mean += scalar(a, r * width + j);
          mean /= width;
          let variance = 0;
          for (let j = 0; j < width; j++) {
            const d = scalar(a, r * width + j) - mean;
            variance += d * d;
          }
          const inv = 1 / Math.sqrt(variance / width + attrs.epsilon);
          outputs[1].storage.data[r] = mean;
          outputs[2].storage.data[r] = inv;
          for (let j = 0; j < width; j++)
            set(
              r * width + j,
              (scalar(a, r * width + j) - mean) * inv * (b ? scalar(b, j) : 1) +
                (c ? scalar(c, j) : 0),
            );
        }
        break;
      }
      case "softmax": {
        const d = attrs.dim,
          width = shape[d],
          inner = sizeOf(shape.slice(d + 1)),
          rows = count / width;
        for (let r = 0; r < rows; r++) {
          const base = Math.floor(r / inner) * width * inner + (r % inner);
          let max = -Infinity;
          for (let j = 0; j < width; j++)
            max = Math.max(max, scalar(a, base + j * inner));
          let sum = 0;
          for (let j = 0; j < width; j++)
            sum += Math.exp(scalar(a, base + j * inner) - max);
          for (let j = 0; j < width; j++)
            set(
              base + j * inner,
              max === -Infinity
                ? 0
                : Math.exp(scalar(a, base + j * inner) - max) / sum,
            );
        }
        break;
      }
      case "reduce": {
        const dims = new Set(attrs.dims),
          rd = a.shape.filter((_, i) => dims.has(i)),
          non = a.shape.filter((_, i) => !dims.has(i)),
          rn = sizeOf(rd);
        const rv = tensor(
            a.storage,
            rd,
            a.stride.filter((_, i) => dims.has(i)),
            0,
          ),
          nv = tensor(
            a.storage,
            non,
            a.stride.filter((_, i) => !dims.has(i)),
            a.offset,
          );
        for (let i = 0; i < count; i++) {
          let acc =
              attrs.kind === "min"
                ? Infinity
                : attrs.kind === "max"
                  ? -Infinity
                  : 0,
            base = physical(nv, i);
          for (let j = 0; j < rn; j++) {
            const v = a.storage.data[base + physical(rv, j)];
            switch (attrs.kind) {
              case "min":
                acc = Math.min(acc, v);
                break;
              case "max":
                acc = Math.max(acc, v);
                break;
              case "any":
                acc = acc || v !== 0 ? 1 : 0;
                break;
              case "linalg_vector_norm":
                acc += v * v;
                break;
              default:
                acc += v;
            }
          }
          set(i, attrs.kind === "linalg_vector_norm" ? Math.sqrt(acc) : acc);
        }
        break;
      }
      case "cat":
      case "stack": {
        const d = attrs.dim,
          inner = sizeOf(shape.slice(d + 1)),
          width = shape[d];
        let offset = 0;
        for (const t of inputs) {
          const w = kind === "stack" ? 1 : t.shape[d],
            n = sizeOf(t.shape);
          for (let j = 0; j < n; j++) {
            const outer = Math.floor(j / (w * inner)),
              rest = j % (w * inner);
            set(outer * width * inner + offset * inner + rest, scalar(t, j));
          }
          offset += w;
        }
        break;
      }
      case "embedding": {
        const width = a.shape[1];
        for (let i = 0; i < count; i++)
          set(
            i,
            scalar(
              a,
              Math.trunc(scalar(b, Math.floor(i / width))) * width +
                (i % width),
            ),
          );
        break;
      }
      case "scatter": {
        const dim = attrs.dim,
          is = strides(b.shape),
          os = strides(shape);
        for (let i = 0; i < sizeOf(b.shape); i++) {
          let target = 0;
          for (let d = 0; d < shape.length; d++)
            target +=
              (d === dim
                ? Math.trunc(scalar(b, i))
                : Math.floor(i / is[d]) % b.shape[d]) * os[d];
          set(target, attrs.value);
        }
        break;
      }
      case "repeat": {
        const os = strides(shape),
          is = strides(a.shape),
          shift = shape.length - a.shape.length;
        for (let i = 0; i < count; i++) {
          let j = 0;
          for (let d = shift; d < shape.length; d++)
            j += (Math.floor(i / os[d]) % a.shape[d - shift]) * is[d - shift];
          set(i, scalar(a, j));
        }
        break;
      }
      case "pad": {
        const os = strides(shape),
          is = strides(a.shape),
          p = attrs.pad;
        for (let i = 0; i < count; i++) {
          let j = 0,
            valid = true;
          for (let d = 0; d < shape.length; d++) {
            const before = p[2 * (shape.length - 1 - d)] ?? 0,
              q = (Math.floor(i / os[d]) % shape[d]) - before;
            valid &&= q >= 0 && q < a.shape[d];
            j += q * is[d];
          }
          set(i, valid ? scalar(a, j) : attrs.value);
        }
        break;
      }
      case "cdist": {
        const m = a.shape.at(-2),
          n = b.shape.at(-2),
          k = a.shape.at(-1);
        for (let i = 0; i < count; i++) {
          const batch = Math.floor(i / (m * n)),
            r = Math.floor(i / n) % m,
            col = i % n;
          let sum = 0;
          for (let j = 0; j < k; j++) {
            const delta =
              scalar(a, (batch * m + r) * k + j) -
              scalar(b, (batch * n + col) * k + j);
            sum += delta * delta;
          }
          set(i, Math.sqrt(sum));
        }
        break;
      }
      case "svd": {
        for (let h = 0; h < sizeOf(a.shape) / 9; h++) {
          const A = Array.from({ length: 9 }, (_, i) => scalar(a, h * 9 + i));
          const { u, s, vh } = svd3(A);
          outputs[0].storage.data.set(u, h * 9);
          outputs[1].storage.data.set(s, h * 3);
          outputs[2].storage.data.set(vh, h * 9);
        }
        break;
      }
      case "det": {
        for (let h = 0; h < count; h++) {
          const m = Array.from({ length: 9 }, (_, i) => scalar(a, h * 9 + i));
          set(
            h,
            m[0] * (m[4] * m[8] - m[5] * m[7]) -
              m[1] * (m[3] * m[8] - m[5] * m[6]) +
              m[2] * (m[3] * m[7] - m[4] * m[6]),
          );
        }
        break;
      }
      default:
        throw Error(`Unsupported kernel ${kind}`);
    }
    return outputs.length === 1 ? out : outputs;
  }
}

export function svd3(a) {
  const b = [...a],
    v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 16; sweep++)
    for (let p = 0; p < 2; p++)
      for (let q = p + 1; q < 3; q++) {
        let aa = 0,
          bb = 0,
          ab = 0;
        for (let i = 0; i < 3; i++) {
          aa += b[i * 3 + p] ** 2;
          bb += b[i * 3 + q] ** 2;
          ab += b[i * 3 + p] * b[i * 3 + q];
        }
        if (Math.abs(ab) < 1e-14 * Math.sqrt(aa * bb)) continue;
        const tau = (bb - aa) / (2 * ab),
          t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau)),
          c = 1 / Math.sqrt(1 + t * t),
          s = t * c;
        for (let i = 0; i < 3; i++) {
          let x = b[i * 3 + p],
            y = b[i * 3 + q];
          b[i * 3 + p] = c * x - s * y;
          b[i * 3 + q] = s * x + c * y;
          x = v[i * 3 + p];
          y = v[i * 3 + q];
          v[i * 3 + p] = c * x - s * y;
          v[i * 3 + q] = s * x + c * y;
        }
      }
  const norms = [0, 1, 2].map((j) => Math.hypot(b[j], b[3 + j], b[6 + j])),
    order = [0, 1, 2].sort((i, j) => norms[j] - norms[i]),
    u = [],
    vh = [],
    s = order.map((i) => norms[i]);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) u.push(b[i * 3 + order[j]] / (s[j] || 1));
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) vh.push(v[j * 3 + order[i]]);
  return { u, s, vh };
}
