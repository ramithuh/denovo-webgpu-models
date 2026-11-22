const f = Math.fround,
  sigmoid = (x) => f(1 / (1 + Math.exp(-x)));
export class CPUOps {
  constructor(weights, constants) {
    this.weights = weights;
    this.constants = constants;
  }
  tensor(data, rows, cols) {
    return { data, rows, cols };
  }
  scope(fn) {
    return fn();
  }
  retain() {}
  weight(name) {
    const w = this.weights[name];
    if (!w) throw Error(`Missing weight ${name}`);
    return w;
  }
  linear(x, name) {
    const w = this.weight(name + ".weight"),
      [C, K] = w.shape,
      b = this.weights[name + ".bias"]?.data;
    if (K !== x.cols) throw Error("Linear shape mismatch " + name);
    const y = new Float32Array(x.rows * C),
      a = x.data,
      wd = w.data;
    if (x.sparse) {
      for (let r = 0; r < x.rows; r++)
        for (let c = 0; c < C; c++) {
          let s = 0;
          for (const j of x.sparse[r]) s = f(s + wd[c * K + j]);
          y[r * C + c] = s;
        }
    } else
      for (let r = 0; r < x.rows; r++) {
        const base = r * K;
        for (let c = 0; c < C; c++) {
          let s0 = 0,
            s1 = 0,
            s2 = 0,
            s3 = 0,
            k = 0,
            offset = c * K;
          for (; k + 3 < K; k += 4) {
            s0 += a[base + k] * wd[offset + k];
            s1 += a[base + k + 1] * wd[offset + k + 1];
            s2 += a[base + k + 2] * wd[offset + k + 2];
            s3 += a[base + k + 3] * wd[offset + k + 3];
          }
          let s = s0 + s1 + s2 + s3;
          for (; k < K; k++) s += a[base + k] * wd[offset + k];
          y[r * C + c] = s + (b?.[c] ?? 0);
        }
      }
    return this.tensor(y, x.rows, C);
  }
  norm(x, name) {
    const { rows: R, cols: C, data: a } = x,
      w = name ? this.weight(name + ".weight").data : null,
      b = name ? this.weight(name + ".bias").data : null,
      y = new Float32Array(a.length);
    for (let r = 0; r < R; r++) {
      let m = 0;
      for (let c = 0; c < C; c++) m += a[r * C + c];
      m /= C;
      let v = 0;
      for (let c = 0; c < C; c++) v += (a[r * C + c] - m) ** 2;
      const inv = 1 / Math.sqrt(v / C + 1e-5);
      for (let c = 0; c < C; c++)
        y[r * C + c] = (a[r * C + c] - m) * inv * (w?.[c] ?? 1) + (b?.[c] ?? 0);
    }
    return this.tensor(y, R, C);
  }
  slice(x, start, width) {
    const y = new Float32Array(x.rows * width);
    for (let r = 0; r < x.rows; r++)
      y.set(
        x.data.subarray(r * x.cols + start, r * x.cols + start + width),
        r * width,
      );
    return this.tensor(y, x.rows, width);
  }
  concat(a, b) {
    const out = new Float32Array(a.cols + b.cols);
    out.set(a.data);
    out.set(b.data, a.cols);
    return this.tensor(out, 1, out.length);
  }
  add(a, b) {
    return this.tensor(
      Float32Array.from(a.data, (v, i) => f(v + b.data[i])),
      a.rows,
      a.cols,
    );
  }
  adapt(x, g, b) {
    return this.tensor(
      Float32Array.from(x.data, (v, i) =>
        f(
          f(v * sigmoid(g.data[i % g.data.length])) + b.data[i % b.data.length],
        ),
      ),
      x.rows,
      x.cols,
    );
  }
  scaledAdd(x, delta, g) {
    return this.tensor(
      Float32Array.from(x.data, (v, i) =>
        f(v + f(delta.data[i] * sigmoid(g.data[i % g.data.length]))),
      ),
      x.rows,
      x.cols,
    );
  }
  swiglu(x) {
    const C = x.cols / 2,
      y = new Float32Array(x.rows * C);
    for (let r = 0; r < x.rows; r++)
      for (let c = 0; c < C; c++) {
        const g = x.data[r * x.cols + C + c];
        y[r * C + c] = f(f(g * sigmoid(g)) * x.data[r * x.cols + c]);
      }
    return this.tensor(y, x.rows, C);
  }
  time(t) {
    const data = new Float32Array(256);
    for (let i = 0; i < 128; i++) {
      const a = f(f(t * 2000) * this.constants.timeFreq[i]);
      data[i] = Math.sin(a);
      data[i + 128] = Math.cos(a);
    }
    return this.tensor(data, 1, 256);
  }
  fold(indices) {
    const data = new Float32Array(768);
    for (let j = 0; j < 3; j++)
      data.set(
        this.weight(
          `cond_factory.feat_creators.1.embedding_${["C", "A", "T"][j]}.weight`,
        ).data.subarray(indices[j] * 256, (indices[j] + 1) * 256),
        j * 256,
      );
    return this.tensor(data, 1, 768);
  }
  coordinates(input) {
    return this.tensor(input.x, input.length, 3);
  }
  features({ length: L, xSC, residueIndices, chainBreaks }) {
    const data = new Float32Array(L * 132);
    for (let r = 0; r < L; r++) {
      const index = residueIndices
        ? residueIndices[r] - residueIndices[0] + 1
        : r + 1;
      for (let i = 0; i < 64; i++) {
        const a = f(f(index * Math.PI) / this.constants.indexDivisor[i]);
        data[r * 132 + i] = Math.sin(a);
        data[r * 132 + i + 64] = Math.cos(a);
      }
      for (let c = 0; c < 3; c++)
        data[r * 132 + 128 + c] = xSC?.[r * 3 + c] ?? 0;
      data[r * 132 + 131] = chainBreaks?.[r] ?? 0;
    }
    return this.tensor(data, L, 132);
  }
  pairFeatures({ length: L, x, xSC, residueIndices }) {
    const sparse = [],
      bucket = (a, i, j, bins) => {
        let s = 0;
        for (let k = 0; k < 3; k++) {
          const d = f(a[i * 3 + k] - a[j * 3 + k]);
          s = f(s + f(d * d));
        }
        const d = f(Math.sqrt(s));
        let b = 0;
        while (b < bins.length && d > bins[b]) b++;
        return b;
      };
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++) {
        const sep = residueIndices
            ? residueIndices[i] - residueIndices[j]
            : i - j,
          a = [Math.max(0, Math.min(126, Math.ceil(sep + 62.5)))];
        if (xSC) a.push(127 + bucket(xSC, i, j, this.constants.scBins));
        a.push(255 + bucket(x, i, j, this.constants.xtBins));
        sparse.push(a);
      }
    return { sparse, rows: L * L, cols: 319 };
  }
  extendCond(x, L) {
    const y = new Float32Array((L + 10) * 512);
    for (let r = 10; r < L + 10; r++) y.set(x.data, r * 512);
    return this.tensor(y, L + 10, 512);
  }
  extendSequence(x, L) {
    const y = new Float32Array((L + 10) * 768);
    y.set(this.weight("registers").data);
    y.set(x.data, 7680);
    return this.tensor(y, L + 10, 768);
  }
  extendPair(x, L) {
    const N = L + 10,
      y = new Float32Array(N * N * 512);
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++)
        y.set(
          x.data.subarray((i * L + j) * 512, (i * L + j + 1) * 512),
          ((i + 10) * N + j + 10) * 512,
        );
    return this.tensor(y, N * N, 512);
  }
  removeRegisters(x, L) {
    return this.tensor(x.data.slice(7680), L, 768);
  }
  gatedAttention(q, k, v, g, bias, N) {
    const out = new Float32Array(N * 768),
      prob = new Float32Array(N);
    for (let h = 0; h < 12; h++)
      for (let i = 0; i < N; i++) {
        let max = -Infinity;
        for (let j = 0; j < N; j++) {
          let s = 0;
          for (let d = 0; d < 64; d++)
            s += q.data[i * 768 + h * 64 + d] * k.data[j * 768 + h * 64 + d];
          const z = f(f(s * 0.125) + bias.data[(i * N + j) * 12 + h]);
          prob[j] = z;
          max = Math.max(max, z);
        }
        let sum = 0;
        for (let j = 0; j < N; j++) {
          prob[j] = Math.exp(prob[j] - max);
          sum += prob[j];
        }
        for (let j = 0; j < N; j++) prob[j] /= sum;
        for (let d = 0; d < 64; d++) {
          let s = 0;
          for (let j = 0; j < N; j++)
            s += prob[j] * v.data[j * 768 + h * 64 + d];
          const idx = i * 768 + h * 64 + d;
          out[idx] = f(s) * sigmoid(g.data[idx]);
        }
      }
    return this.tensor(out, N, 768);
  }
}
