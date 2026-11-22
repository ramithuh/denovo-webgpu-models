import { topk } from "./topk.js";
import { checkAbort } from "../shared/loader.js";
const f = Math.fround;
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  return (
    sign *
    (1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
        t +
        0.254829592) *
        t *
        Math.exp(-x * x))
  );
}
const gelu = (x) => x * 0.5 * (1 + erf(x / Math.SQRT2));
const tensor = (data, rows, cols) => ({ data, rows, cols });
export class CPUOps {
  constructor(weights, options) {
    this.w = weights;
    this.options = options;
    this.traces = {};
  }
  trace(name, t) {
    if (this.options.returnIntermediates) this.traces[name] = t.data.slice();
  }
  free() {}
  async checkpoint(phase) {
    this.options.onProgress?.({ phase });
    checkAbort(this.options.signal);
  }
  zeros(rows, cols) {
    return tensor(new Float32Array(rows * cols), rows, cols);
  }
  geometry(input) {
    const { length: L, X, mask, chain, residue_idx } = input,
      K = Math.min(48, L),
      xyz = new Float32Array(L * 15);
    for (let i = 0; i < L; i++) {
      xyz.set(X.subarray(i * 12, i * 12 + 12), i * 15);
      const b = [],
        c = [];
      for (let d = 0; d < 3; d++) {
        b.push(f(X[i * 12 + 3 + d] - X[i * 12 + d]));
        c.push(f(X[i * 12 + 6 + d] - X[i * 12 + 3 + d]));
      }
      for (let d = 0; d < 3; d++) {
        const d1 = (d + 1) % 3,
          d2 = (d + 2) % 3,
          cross = f(f(b[d1] * c[d2]) - f(b[d2] * c[d1]));
        xyz[i * 15 + 12 + d] = f(
          f(f(f(-0.58273431) * cross) + f(f(0.56802827) * b[d])) -
            f(f(0.54067466) * c[d]),
        );
        xyz[i * 15 + 12 + d] = f(xyz[i * 15 + 12 + d] + X[i * 12 + 3 + d]);
      }
    }
    const dist = (i, j, a, b) => {
      let sum = 0;
      for (let c = 0; c < 3; c++) {
        const d = f(xyz[i * 15 + a * 3 + c] - xyz[j * 15 + b * 3 + c]);
        sum = f(sum + f(d * d));
      }
      return f(Math.sqrt(f(sum + f(1e-6))));
    };
    const idx = new Int32Array(L * K),
      D = new Float32Array(L * K);
    for (let i = 0; i < L; i++) {
      const d = Float32Array.from(
        { length: L },
        (_, j) => mask[i] * mask[j] * dist(i, j, 1, 1),
      );
      let max = 0;
      for (const v of d) max = Math.max(max, v);
      for (let j = 0; j < L; j++)
        d[j] = f(d[j] + (1 - mask[i] * mask[j]) * max);
      const selected = topk(d, K);
      idx.set(selected, i * K);
      for (let k = 0; k < K; k++) D[i * K + k] = d[selected[k]];
    }
    const pairs = [
      [1, 1],
      [0, 0],
      [2, 2],
      [3, 3],
      [4, 4],
      [1, 0],
      [1, 2],
      [1, 3],
      [1, 4],
      [0, 2],
      [0, 3],
      [0, 4],
      [4, 2],
      [4, 3],
      [3, 2],
      [0, 1],
      [2, 1],
      [3, 1],
      [4, 1],
      [2, 0],
      [3, 0],
      [4, 0],
      [2, 4],
      [3, 4],
      [2, 3],
    ];
    const g = this.zeros(L * K, 416),
      pos = this.w["prot_mpnn.features.embeddings.linear.weight"],
      bias = this.w["prot_mpnn.features.embeddings.linear.bias"];
    const step = f(20 / 15),
      mu = Array.from({ length: 16 }, (_, i) =>
        i < 8 ? f(2 + f(i * step)) : f(22 - f((15 - i) * step)),
      );
    for (let i = 0; i < L; i++)
      for (let k = 0; k < K; k++) {
        const r = i * K + k,
          j = idx[r],
          off =
            chain[i] === chain[j]
              ? Math.min(64, Math.max(0, residue_idx[i] - residue_idx[j] + 32))
              : 65;
        for (let c = 0; c < 16; c++)
          g.data[r * 416 + c] = f(pos[c * 66 + off] + bias[c]);
        for (let p = 0; p < 25; p++) {
          const d = p === 0 ? D[r] : dist(i, j, ...pairs[p]);
          for (let c = 0; c < 16; c++) {
            const z = f(f(d - mu[c]) / 1.25);
            g.data[r * 416 + 16 + p * 16 + c] = f(Math.exp(-f(z * z)));
          }
        }
      }
    return { geometry: g, indices: tensor(idx, L, K) };
  }
  dense(x, name, activation = "none", preRelu = false) {
    const w = this.w[name + ".weight"],
      b = this.w[name + ".bias"],
      I = x.cols,
      O = w.length / I,
      y = this.zeros(x.rows, O),
      xd = x.data,
      yd = y.data;
    for (let r = 0; r < x.rows; r++)
      for (let o = 0; o < O; o++) {
        let sum = 0;
        for (let i = 0; i < I; i++) {
          const v = xd[r * I + i];
          sum += (preRelu ? Math.max(v, 0) : v) * w[o * I + i];
        }
        sum = f(sum + (b?.[o] ?? 0));
        yd[r * O + o] = activation === "gelu" ? gelu(sum) : sum;
      }
    return y;
  }
  norm(x, delta, name, input) {
    const y = this.zeros(x.rows, x.cols),
      w = this.w[name + ".weight"],
      b = this.w[name + ".bias"],
      C = x.cols;
    for (let r = 0; r < x.rows; r++) {
      let mean = 0;
      for (let c = 0; c < C; c++) {
        y.data[r * C + c] = f(
          x.data[r * C + c] + (delta?.data[r * C + c] ?? 0),
        );
        mean += y.data[r * C + c];
      }
      mean /= C;
      let v = 0;
      for (let c = 0; c < C; c++) v += (y.data[r * C + c] - mean) ** 2;
      const inv = 1 / Math.sqrt(v / C + 1e-5);
      for (let c = 0; c < C; c++)
        y.data[r * C + c] =
          ((y.data[r * C + c] - mean) * inv * w[c] + b[c]) *
          (input?.mask[r] ?? 1);
    }
    return y;
  }
  concat(nodes, edges, idx, emb, input, decoder) {
    const K = idx.cols,
      L = nodes.rows,
      C = decoder ? 512 : 384,
      y = this.zeros(L * K, C);
    for (let r = 0; r < L * K; r++) {
      const i = Math.floor(r / K),
        j = idx.data[r],
        mask = decoder ? input.mask[i] : 1;
      for (let c = 0; c < 128; c++) {
        y.data[r * C + c] = nodes.data[i * 128 + c];
        y.data[r * C + 128 + c] = edges.data[r * 128 + c] * mask;
        if (decoder) {
          y.data[r * C + 256 + c] = emb.data[j * 128 + c] * mask;
          y.data[r * C + 384 + c] = nodes.data[j * 128 + c] * mask;
        } else y.data[r * C + 256 + c] = nodes.data[j * 128 + c];
      }
    }
    return y;
  }
  reduce(msg, idx, input, encoder) {
    const L = idx.rows,
      K = idx.cols,
      y = this.zeros(L, 128);
    for (let i = 0; i < L; i++)
      for (let c = 0; c < 128; c++) {
        let sum = 0;
        for (let k = 0; k < K; k++)
          sum +=
            msg.data[(i * K + k) * 128 + c] *
            (encoder ? input.mask[i] * input.mask[idx.data[i * K + k]] : 1);
        y.data[i * 128 + c] = sum / 30;
      }
    return y;
  }
  embed(input, name) {
    const y = this.zeros(input.length, 128),
      w = this.w[name];
    for (let i = 0; i < input.length; i++)
      y.data.set(w.subarray(input.S[i] * 128, (input.S[i] + 1) * 128), i * 128);
    return y;
  }
  join(ts) {
    const y = this.zeros(
      ts[0].rows,
      ts.reduce((s, t) => s + t.cols, 0),
    );
    for (let r = 0; r < y.rows; r++) {
      let c = 0;
      for (const t of ts) {
        y.data.set(
          t.data.subarray(r * t.cols, (r + 1) * t.cols),
          r * y.cols + c,
        );
        c += t.cols;
      }
    }
    return y;
  }
  calibrate(x) {
    const y = this.zeros(x.rows, x.cols),
      w = this.w["ddg_out.weight"][0],
      b = this.w["ddg_out.bias"][0];
    for (let i = 0; i < x.data.length; i++) y.data[i] = f(f(x.data[i] * w) + b);
    return y;
  }
  subtract(scores, input) {
    const y = this.zeros(input.length, 20);
    for (let i = 0; i < input.length; i++)
      for (let a = 0; a < 20; a++)
        y.data[i * 20 + a] =
          scores.data[i * 21 + a] - scores.data[i * 21 + input.S[i]];
    return y;
  }
}
