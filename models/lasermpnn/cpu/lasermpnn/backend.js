// Operations round results into FP32 arrays, with double accumulation in JS.
export const tensor = (data, r, c) => ({
  data: data instanceof Float32Array ? data : Float32Array.from(data),
  r,
  c,
});
export function gelu(x) {
  // Abramowitz-Stegun erf, max absolute error <1.5e-7.
  const z = Math.abs(x) / Math.SQRT2,
    t = 1 / (1 + 0.3275911 * z);
  const e =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-z * z);
  return 0.5 * x * (1 + (x < 0 ? -e : e));
}
export class CpuBackend {
  constructor(weights) {
    this.weights = weights;
  }
  from(data, r, c) {
    return tensor(data, r, c);
  }
  zeros(r, c) {
    return tensor(new Float32Array(r * c), r, c);
  }
  weight(key) {
    const t = this.weights[key];
    if (!t) throw Error(`Missing weight ${key}`);
    return tensor(t.data, t.shape[0], t.data.length / t.shape[0]);
  }
  linear(x, key) {
    return this.matmul(
      x,
      this.weight(key + ".weight"),
      this.weights[key + ".bias"]?.data,
    );
  }
  matmul(x, w, bias) {
    const y = this.zeros(x.r, w.r);
    for (let i = 0; i < x.r; i++)
      for (let j = 0; j < w.r; j++) {
        let s = bias?.[j] ?? 0;
        for (let k = 0; k < x.c; k++)
          s += x.data[i * x.c + k] * w.data[j * x.c + k];
        y.data[i * y.c + j] = s;
      }
    return y;
  }
  act(x, kind) {
    return tensor(
      x.data.map((v) => (kind === "gelu" ? gelu(v) : 1 / (1 + Math.exp(-v)))),
      x.r,
      x.c,
    );
  }
  add(x, y) {
    return tensor(
      x.data.map((v, i) => v + y.data[i]),
      x.r,
      x.c,
    );
  }
  cat(xs) {
    const r = xs[0].r,
      c = xs.reduce((s, x) => s + x.c, 0),
      out = this.zeros(r, c);
    for (let i = 0; i < r; i++) {
      let at = i * c;
      for (const x of xs) {
        out.data.set(x.data.subarray(i * x.c, (i + 1) * x.c), at);
        at += x.c;
      }
    }
    return out;
  }
  gather(x, indices) {
    const out = this.zeros(indices.length, x.c);
    indices.forEach((j, i) =>
      out.data.set(x.data.subarray(j * x.c, (j + 1) * x.c), i * x.c),
    );
    return out;
  }
  assign(x, row, y) {
    x.data.set(y.data, row * x.c);
  }
  norm(x, key) {
    const w = this.weight(key + ".weight").data,
      b = this.weight(key + ".bias").data,
      y = this.zeros(x.r, x.c);
    for (let i = 0; i < x.r; i++) {
      let m = 0,
        v = 0;
      for (let j = 0; j < x.c; j++) m += x.data[i * x.c + j];
      m /= x.c;
      for (let j = 0; j < x.c; j++) v += (x.data[i * x.c + j] - m) ** 2;
      const inv = 1 / Math.sqrt(v / x.c + 1e-5);
      for (let j = 0; j < x.c; j++)
        y.data[i * x.c + j] = (x.data[i * x.c + j] - m) * inv * w[j] + b[j];
    }
    return y;
  }
  vecLinear(x, key) {
    const w = this.weight(key + ".weight"),
      y = this.zeros(x.r, w.r * 3);
    for (let i = 0; i < x.r; i++)
      for (let j = 0; j < w.r; j++)
        for (let d = 0; d < 3; d++) {
          let s = 0;
          for (let k = 0; k < w.c; k++)
            s += x.data[i * x.c + 3 * k + d] * w.data[j * w.c + k];
          y.data[i * y.c + 3 * j + d] = s;
        }
    return y;
  }
  vecLength(x) {
    const y = this.zeros(x.r, x.c / 3);
    for (let i = 0; i < y.data.length; i++)
      y.data[i] = Math.sqrt(
        Math.max(
          1e-8,
          x.data[3 * i] ** 2 + x.data[3 * i + 1] ** 2 + x.data[3 * i + 2] ** 2,
        ),
      );
    return y;
  }
  vecScale(x, s) {
    return tensor(
      x.data.map((v, i) => v * s.data[Math.floor(i / 3)]),
      x.r,
      x.c,
    );
  }
  vecNorm(x) {
    const y = this.zeros(x.r, x.c);
    for (let r = 0; r < x.r; r++) {
      let s = 0;
      for (let j = 0; j < x.c; j += 3) {
        const i = r * x.c + j;
        s += Math.max(
          1e-8,
          x.data[i] ** 2 + x.data[i + 1] ** 2 + x.data[i + 2] ** 2,
        );
      }
      const inv = 1 / Math.sqrt(s / (x.c / 3));
      for (let j = 0; j < x.c; j++)
        y.data[r * x.c + j] = x.data[r * x.c + j] * inv;
    }
    return y;
  }
  rbf(x, key, sigma) {
    const centers = this.weight(key).data,
      y = this.zeros(x.r, x.c * centers.length);
    for (let i = 0; i < x.data.length; i++)
      for (let k = 0; k < centers.length; k++)
        y.data[i * centers.length + k] =
          Math.exp(-(((x.data[i] - centers[k]) / sigma) ** 2)) + 1e-4;
    return y;
  }
  dots(a, b) {
    const va = a.c / 3,
      vb = b.c / 3,
      y = this.zeros(a.r, va * vb);
    for (let r = 0; r < a.r; r++)
      for (let i = 0; i < va; i++)
        for (let j = 0; j < vb; j++) {
          let s = 0,
            aa = 0,
            bb = 0;
          for (let d = 0; d < 3; d++) {
            const x = a.data[r * a.c + 3 * i + d],
              z = b.data[r * b.c + 3 * j + d];
            s += x * z;
            aa += x * x;
            bb += z * z;
          }
          y.data[r * y.c + i * vb + j] =
            s / Math.sqrt(Math.max(aa, 1e-8) * Math.max(bb, 1e-8));
        }
    return y;
  }
  atten(raw, key) {
    const w = this.weights[key],
      heads = w.shape[0],
      dim = w.shape[1],
      y = this.zeros(raw.r, heads);
    for (let r = 0; r < raw.r; r++)
      for (let h = 0; h < heads; h++) {
        let s = 0;
        for (let d = 0; d < dim; d++) {
          const v = raw.data[r * raw.c + h * dim + d];
          s += (v >= 0 ? v : 0.2 * v) * w.data[h * dim + d];
        }
        y.data[r * heads + h] = s;
      }
    return y;
  }
  pool(messages, logits, sinks, rows, heads, dim) {
    const y = this.zeros(rows, dim),
      edges = Array.from({ length: rows }, () => []);
    sinks.forEach((s, i) => edges[s].push(i));
    for (let r = 0; r < rows; r++)
      for (let h = 0; h < heads; h++) {
        let max = -Infinity;
        for (const e of edges[r])
          max = Math.max(max, logits.data[e * heads + h]);
        let sum = 0;
        const probs = edges[r].map((e) => {
          const p = Math.exp(logits.data[e * heads + h] - max);
          sum += p;
          return p;
        });
        for (let d = 0; d < dim; d++) {
          let v = 0;
          for (let j = 0; j < edges[r].length; j++) {
            const e = edges[r][j];
            v +=
              probs[j] *
              messages.data[
                e * messages.c + (messages.c === dim ? 0 : h * dim) + d
              ];
          }
          if (sum) y.data[r * dim + d] += v / sum / heads;
        }
      }
    return y;
  }
  rows(xs) {
    return tensor(
      Float32Array.from(xs.flatMap((x) => Array.from(x.data))),
      xs.reduce((n, x) => n + x.r, 0),
      xs[0].c,
    );
  }
  async read(x) {
    return x.data.slice();
  }
  mark() {
    return 0;
  }
  releaseSince() {}
  destroy() {}
}
