/** FP32 storage and elementwise boundaries; dot products accumulate in JS doubles. */
export class CpuOps {
  tensor(data, shape) {
    return {
      data: data instanceof Float32Array ? data : Float32Array.from(data),
      shape,
    };
  }
  zeros(shape) {
    return this.tensor(
      new Float32Array(shape.reduce((a, b) => a * b, 1)),
      shape,
    );
  }
  linear(x, w, b) {
    const I = w.shape[1],
      O = w.shape[0],
      R = x.data.length / I,
      out = new Float32Array(R * O);
    for (let r = 0; r < R; r++)
      for (let o = 0; o < O; o++) {
        let s = 0;
        const wi = o * I,
          xi = r * I;
        for (let i = 0; i < I; i++) s += x.data[xi + i] * w.data[wi + i];
        out[r * O + o] = s + (b?.data[o] ?? 0);
      }
    return this.tensor(out, [R, O]);
  }
  norm(x, w, b) {
    const D = w.data.length,
      R = x.data.length / D,
      out = new Float32Array(x.data.length);
    for (let r = 0; r < R; r++) {
      let m = 0,
        v = 0;
      for (let j = 0; j < D; j++) m += x.data[r * D + j];
      m /= D;
      for (let j = 0; j < D; j++) v += (x.data[r * D + j] - m) ** 2;
      const s = 1 / Math.sqrt(v / D + 1e-5);
      for (let j = 0; j < D; j++)
        out[r * D + j] = (x.data[r * D + j] - m) * s * w.data[j] + b.data[j];
    }
    return this.tensor(out, [R, D]);
  }
  gelu(x) {
    const out = Float32Array.from(x.data, (v) => {
      const z = Math.abs(v) / Math.SQRT2,
        t = 1 / (1 + 0.3275911 * z),
        erf =
          1 -
          ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
            0.284496736) *
            t +
            0.254829592) *
            t *
            Math.exp(-z * z);
      return 0.5 * v * (1 + Math.sign(v) * erf);
    });
    return this.tensor(out, x.shape);
  }
  add(a, b) {
    return this.tensor(
      Float32Array.from(a.data, (v, i) => v + b.data[i]),
      a.shape,
    );
  }
  mask(a, m) {
    const D = a.shape.at(-1);
    return this.tensor(
      Float32Array.from(a.data, (v, i) => v * m.data[Math.floor(i / D)]),
      a.shape,
    );
  }
  gather(a, idx) {
    const D = a.shape.at(-1),
      out = new Float32Array(idx.length * D);
    for (let r = 0; r < idx.length; r++)
      out.set(a.data.subarray(idx[r] * D, (idx[r] + 1) * D), r * D);
    return this.tensor(out, [idx.length, D]);
  }
  concat(...xs) {
    const R = xs[0].shape[0],
      D = xs.reduce((n, x) => n + x.shape[1], 0),
      out = new Float32Array(R * D);
    for (let r = 0; r < R; r++) {
      let offset = r * D;
      for (const x of xs) {
        out.set(x.data.subarray(r * x.shape[1], (r + 1) * x.shape[1]), offset);
        offset += x.shape[1];
      }
    }
    return this.tensor(out, [R, D]);
  }
  reduce(x, K) {
    const D = x.shape[1],
      R = x.shape[0] / K,
      out = new Float32Array(R * D);
    for (let r = 0; r < R; r++)
      for (let d = 0; d < D; d++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += x.data[(r * K + k) * D + d];
        out[r * D + d] = s / 30;
      }
    return this.tensor(out, [R, D]);
  }
  scatter(target, row, value) {
    target.data.set(value.data, row * target.shape[1]);
  }
  async read(x) {
    return x.data.slice();
  }
  keep(x) {
    return x;
  }
  scope(fn) {
    return fn();
  }
  clear() {}
  destroy() {}
}
