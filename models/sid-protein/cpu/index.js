import { expF32 } from "../shared/numeric.js";
import { fmaF32 } from "../shared/fma.js";
import { constants } from "../shared/constants.js";
import {
  loadWeights,
  decodeTensor,
  cathIndices,
  validateInput,
  gaussianNoise,
  schedule,
  toPDB,
} from "../shared/index.js";
import { forwardNetwork, assertModelManifest } from "../shared/model.js";
const f = Math.fround,
  sig = (x) => f(1 / f(1 + expF32(-x)));
const tensor = (data, rows, cols) => ({ data, rows, cols });
export class CpuBackend {
  constructor(weights) {
    this.weights = weights;
  }
  check() {
    this.signal?.throwIfAborted();
    if (this.destroyed) throw new Error("Engine destroyed");
  }
  keep() {}
  time(t) {
    const a = new Float32Array(256);
    for (let i = 0; i < 128; i++) {
      const z = f(f(t * 2000) * constants.time[i]);
      a[i] = Math.sin(z);
      a[128 + i] = Math.cos(z);
    }
    return tensor(a, 1, 256);
  }
  position(L) {
    const a = new Float32Array(L * 132);
    for (let r = 0; r < L; r++)
      for (let k = 0; k < 64; k++) {
        const z = f(f((r + 1) * f(Math.PI)) / constants.position[k]);
        a[r * 132 + k] = Math.sin(z);
        a[r * 132 + k + 64] = Math.cos(z);
      }
    return tensor(a, L, 132);
  }
  condition(time, ids) {
    const a = new Float32Array(1024);
    a.set(time.data);
    for (let k = 0; k < 3; k++)
      a.set(
        this.weights[
          `cond_factory.feat_creators.1.embedding_${"CAT"[k]}.weight`
        ].data.subarray(ids[k] * 256, (ids[k] + 1) * 256),
        256 + k * 256,
      );
    return tensor(a, 1, 1024);
  }
  linear(a, name) {
    const w = this.weights[name + ".weight"],
      bias = this.weights[name + ".bias"]?.data,
      M = a.rows,
      N = w.rows,
      K = w.cols,
      x = a.data,
      v = w.data,
      y = new Float32Array(M * N);
    // Sparse feature matrices have at most two nonzero entries per row.
    if (a.sparse) {
      for (let r = 0; r < M; r++)
        for (let n = 0; n < N; n++) {
          let s = bias?.[n] ?? 0;
          for (const k of a.sparse[r]) s += v[n * K + k];
          y[r * N + n] = s;
        }
      return tensor(y, M, N);
    }
    for (let r = 0; r < M; r++) {
      const xo = r * K;
      let n = 0;
      // Four independent output rows hide F32 accumulation latency while
      // preserving the exact sequential reduction order within every dot product.
      for (; n + 3 < N; n += 4) {
        const w0 = n * K,
          w1 = w0 + K,
          w2 = w1 + K,
          w3 = w2 + K;
        let s0 = bias?.[n] ?? 0,
          s1 = bias?.[n + 1] ?? 0,
          s2 = bias?.[n + 2] ?? 0,
          s3 = bias?.[n + 3] ?? 0;
        for (let k = 0; k < K; k++) {
          const z = x[xo + k];
          s0 = f(z * v[w0 + k] + s0);
          s1 = f(z * v[w1 + k] + s1);
          s2 = f(z * v[w2 + k] + s2);
          s3 = f(z * v[w3 + k] + s3);
        }
        const out = r * N + n;
        y[out] = s0;
        y[out + 1] = s1;
        y[out + 2] = s2;
        y[out + 3] = s3;
      }
      for (; n < N; n++) {
        const wo = n * K;
        let sum = bias?.[n] ?? 0;
        for (let k = 0; k < K; k++) sum = f(x[xo + k] * v[wo + k] + sum);
        y[r * N + n] = sum;
      }
    }
    return tensor(y, M, N);
  }
  norm(a, name, opts = {}) {
    const D = opts.width ?? a.cols,
      S = opts.stride ?? a.cols,
      O = opts.offset ?? 0,
      w = this.weights[name + ".weight"]?.data,
      b = this.weights[name + ".bias"]?.data,
      y = new Float32Array(a.rows * D);
    const merge = (a, b) => {
      const n = a[0] + b[0],
        c = n ? f(b[0] / n) : 0,
        d = f(b[1] - a[1]);
      return [
        n,
        f(c * d + a[1]),
        f(a[2] + fmaF32(f(f(d * d) * c), a[0], b[2])),
      ];
    };
    const n = D / 4,
      chunks = Math.ceil(n / 16),
      depth = Math.ceil(Math.log2(chunks));
    for (let r = 0; r < a.rows; r++) {
      let total = [0, 0, 0];
      for (let lane = 0; lane < 4; lane++) {
        const stack = Array.from({ length: 8 }, () => [0, 0, 0]);
        for (let chunk = 0; chunk < chunks; chunk++) {
          const size = Math.min(16, n - chunk * 16);
          let m = 0,
            v = 0;
          for (let j = 0; j < size; j++) {
            const z = a.data[r * S + O + (chunk * 16 + j) * 4 + lane],
              d = f(z - m);
            m = f(d * f(1 / (j + 1)) + m);
            v = f(d * f(z - m) + v);
          }
          stack[0] = merge(stack[0], [size, m, v]);
          let mask = chunk + 1,
            level = 1;
          while (level < depth && mask % 2 === 0) {
            stack[level] = merge(stack[level], stack[level - 1]);
            stack[level - 1] = [0, 0, 0];
            mask >>= 1;
            level++;
          }
        }
        for (let j = 1; j < depth; j++) stack[0] = merge(stack[0], stack[j]);
        total = merge(total, stack[0]);
      }
      const mean = total[1],
        inv = f(1 / f(Math.sqrt(f(f(total[2] / D) + f(1e-5)))));
      for (let k = 0; k < D; k++) {
        const z = f(f(a.data[r * S + O + k] - mean) * inv);
        y[r * D + k] = w ? f(z * w[k] + b[k]) : z;
      }
    }
    return tensor(y, a.rows, D);
  }
  swiglu(a) {
    const D = a.cols / 2,
      y = new Float32Array(a.rows * D);
    for (let i = 0; i < y.length; i++) {
      const j = Math.floor(i / D) * a.cols + (i % D),
        g = a.data[j + D];
      y[i] = f(g / f(1 + expF32(-g))) * a.data[j];
    }
    return tensor(y, a.rows, D);
  }
  add(a, b) {
    return tensor(
      Float32Array.from(a.data, (v, i) => v + b.data[i]),
      a.rows,
      a.cols,
    );
  }
  padCondition(a) {
    const y = new Float32Array(1024);
    y.set(a.data, 512);
    return tensor(y, 2, 512);
  }
  registers(a) {
    const y = new Float32Array((a.rows + 10) * 768);
    y.set(this.weights.registers.data);
    y.set(a.data, 7680);
    return tensor(y, a.rows + 10, 768);
  }
  pairFeatures(x) {
    const L = x.rows,
      y = new Float32Array(L * L * 319),
      sparse = [];
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        const sep = Math.max(0, Math.min(126, i - j + 63));
        let d = 0;
        for (let k = 0; k < 3; k++) {
          const z = f(x.data[i * 3 + k] - x.data[j * 3 + k]);
          d = f(d + f(z * z));
        }
        d = f(Math.sqrt(d));
        let bucket = 0;
        while (bucket < 63 && d > constants.distance[bucket]) bucket++;
        const p = i * L + j;
        sparse[p] = [sep, 255 + bucket];
        for (const k of sparse[p]) y[p * 319 + k] = 1;
      }
    return { ...tensor(y, L * L, 319), sparse };
  }
  adaptive(x, g, b, mode) {
    const y = new Float32Array(x.data.length),
      D = x.cols;
    for (let i = 0; i < y.length; i++) {
      const r = Math.floor(i / D),
        j = (mode ? (r < 10 ? 0 : 1) : 0) * D + (i % D);
      y[i] = f(x.data[i] * f(sig(g.data[j]))) + b.data[j];
    }
    return tensor(y, x.rows, D);
  }
  residualGate(x, h, g) {
    const y = new Float32Array(x.data.length),
      D = x.cols;
    for (let i = 0; i < y.length; i++) {
      const j = (Math.floor(i / D) < 10 ? 0 : 1) * D + (i % D);
      y[i] = x.data[i] + f(h.data[i] * f(sig(g.data[j])));
    }
    return tensor(y, x.rows, D);
  }
  padPair(a, L) {
    const N = L + 10,
      y = new Float32Array(N * N * 512);
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++)
        y.set(
          a.data.subarray((i * L + j) * 512, (i * L + j + 1) * 512),
          ((i + 10) * N + j + 10) * 512,
        );
    return tensor(y, N * N, 512);
  }
  attention(q, k, qkv, bias, g, N) {
    const y = new Float32Array(N * 768),
      p = new Float32Array(N);
    for (let h = 0; h < 12; h++)
      for (let i = 0; i < N; i++) {
        let mx = -Infinity;
        for (let j = 0; j < N; j++) {
          let s = 0;
          for (let d = 0; d < 64; d++)
            s = f(
              q.data[i * 768 + h * 64 + d] * k.data[j * 768 + h * 64 + d] + s,
            );
          p[j] = f(s * 0.125) + bias.data[(i * N + j) * 12 + h];
          mx = Math.max(mx, p[j]);
        }
        const sums = new Float32Array(4);
        for (let j = 0; j < N; j++) {
          p[j] = expF32(f(p[j] - mx));
          sums[j % 4] += p[j];
        }
        const den = f(f(sums[0] + sums[2]) + f(sums[1] + sums[3])),
          inv = f(1 / den);
        for (let d = 0; d < 64; d++) {
          let s = 0;
          for (let j = 0; j < N; j++)
            s = f(f(p[j] * inv) * qkv.data[j * 2304 + 1536 + h * 64 + d] + s);
          const off = i * 768 + h * 64 + d;
          y[off] = f(s) * f(sig(g.data[off]));
        }
      }
    return tensor(y, N, 768);
  }
  unpad(a, n) {
    return tensor(a.data.slice(n * a.cols), a.rows - n, a.cols);
  }
  velocity(x, v, t) {
    return tensor(
      Float32Array.from(x.data, (z, i) => z + f(f(1 - t) * v.data[30 + i])),
      x.rows,
      3,
    );
  }
  mix(prev, noise, t, scale) {
    const center = (a) => {
      const L = a.length / 3,
        size = Math.floor(L / 4),
        power = Math.max(
          4,
          Math.floor(Math.ceil(Math.log2(Math.max(size, 1))) / 4),
        ),
        step = 2 ** power,
        mask = step - 1,
        y = new Float32Array(a.length);
      for (let d = 0; d < 3; d++) {
        const acc = new Float32Array(16);
        let i = 0;
        for (; i + step <= size; ) {
          for (let j = 0; j < step; j++)
            for (let k = 0; k < 4; k++) acc[k] += a[((i + j) * 4 + k) * 3 + d];
          i += step;
          for (let level = 1; level < 4; level++) {
            for (let k = 0; k < 4; k++) {
              acc[level * 4 + k] += acc[(level - 1) * 4 + k];
              acc[(level - 1) * 4 + k] = 0;
            }
            if (i & (mask << (level * power))) break;
          }
        }
        for (; i < size; i++)
          for (let k = 0; k < 4; k++) acc[k] += a[(i * 4 + k) * 3 + d];
        for (let level = 1; level < 4; level++)
          for (let k = 0; k < 4; k++) acc[k] += acc[level * 4 + k];
        for (let j = size * 4; j < L; j++) acc[0] += a[j * 3 + d];
        let sum = acc[0];
        for (let k = 1; k < 4; k++) sum = f(sum + acc[k]);
        const mean = f(sum / L);
        for (let j = d; j < a.length; j += 3) y[j] = a[j] - mean;
      }
      return y;
    };
    const z = center(Float32Array.from(center(noise), (v) => v * scale)),
      x = center(prev);
    return Float32Array.from(x, (v, i) => f(f(1 - t) * z[i]) + f(t * v));
  }
}
export async function createSidProteinCpuEngine({
  baseUrl,
  precision = "fp32",
  ...options
} = {}) {
  if (!["fp32", "fp16", "int8"].includes(precision))
    throw new RangeError("Unknown precision");
  const bundle = await loadWeights(
      baseUrl ?? new URL(`../weights/${precision}`, import.meta.url).href,
      options,
    ),
    w = {};
  assertModelManifest(bundle.manifest);
  if (bundle.manifest.precision !== precision)
    throw new Error("Precision mismatch");
  for (const [k, v] of Object.entries(bundle.weights))
    w[k] = tensor(decodeTensor(v), v.shape[0], v.shape[1] ?? 1);
  const b = new CpuBackend(w);
  let busy = false;
  async function forward(coordinates, t, opts = {}) {
    b.check();
    validateInput(coordinates, t);
    if (busy) throw new Error("Engine busy");
    busy = true;
    b.signal = opts.signal;
    try {
      return (
        await forwardNetwork(
          b,
          tensor(coordinates, coordinates.length / 3, 3),
          f(t),
          cathIndices(opts),
          opts.onTrace ? async (k, v) => opts.onTrace(k, v.data) : null,
        )
      ).data;
    } finally {
      busy = false;
      b.signal = null;
    }
  }
  async function generate(opts = {}) {
    const {
      length = 64,
      steps = 16,
      noiseScale = 0.45,
      seed = 5,
      tInit = 0.37,
      onestepGamma = 1,
      signal,
    } = opts;
    if (!Number.isInteger(length) || length < 1)
      throw new RangeError("Invalid length");
    if (
      !Number.isFinite(noiseScale) ||
      noiseScale < 0 ||
      !Number.isFinite(onestepGamma) ||
      onestepGamma < 0
    )
      throw new RangeError("Invalid noise scale");
    const times = schedule(steps, tInit),
      noise = opts.noise ?? gaussianNoise(length, steps, seed);
    if (noise.length !== length * 3 * steps)
      throw new RangeError("Noise shape mismatch");
    let x = new Float32Array(length * 3);
    const trajectory = [];
    for (let i = 0; i < steps; i++) {
      signal?.throwIfAborted();
      x = await forward(
        b.mix(
          x,
          noise.subarray(i * length * 3, (i + 1) * length * 3),
          times[i],
          steps === 1 ? onestepGamma : noiseScale,
        ),
        times[i],
        opts,
      );
      trajectory.push(x);
      await opts.onStep?.({ step: i + 1, steps, coordinates: x });
    }
    return { coordinates: x, trajectory, times, units: "nm", pdb: toPDB(x) };
  }
  return {
    forward,
    generate,
    destroy() {
      b.destroyed = true;
      b.weights = {};
    },
    memoryUsage() {
      return {
        totalBytes: Object.values(b.weights).reduce(
          (s, t) => s + t.data.byteLength,
          0,
        ),
        transientBuffers: 0,
      };
    },
  };
}
