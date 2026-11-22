import { constants as C } from "../shared/constants.js";
const f = Math.fround,
  make = (data, rows, cols) => ({ data, rows, cols });
const norm = (v) => {
  const length = f(Math.sqrt(v.reduce((s, x) => f(s + x * x), 0)));
  return v.map((x) => f(x / Math.max(length, 1e-12)));
};
const sub = (a, b) => a.map((x, i) => f(x - b[i]));
const cross = (a, b) => [
  f(f(a[1] * b[2]) - f(a[2] * b[1])),
  f(f(a[2] * b[0]) - f(a[0] * b[2])),
  f(f(a[0] * b[1]) - f(a[1] * b[0])),
];
// Abramowitz/Stegun's rational form is used only for GELU's scalar erf.
function erf(x) {
  const sign = Math.sign(x),
    a = Math.abs(x),
    t = 1 / (1 + 0.3275911 * a);
  return (
    sign *
    (1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
        t +
        0.254829592) *
        t *
        Math.exp(-a * a))
  );
}
export const graphOps = {
  fromArray(data, rows, cols) {
    return data?.data ? data : make(Float32Array.from(data), rows, cols);
  },
  zeros(rows, cols) {
    return make(new Float32Array(rows * cols), rows, cols);
  },
  repeatRows(x, rows) {
    return make(
      Float32Array.from(
        { length: rows * x.cols },
        (_, i) => x.data[i % x.cols],
      ),
      rows,
      x.cols,
    );
  },
  exp(x) {
    return make(Float32Array.from(x.data, Math.exp), x.rows, x.cols);
  },
  gelu(x) {
    return make(
      Float32Array.from(x.data, (v) => 0.5 * v * (1 + erf(v / Math.SQRT2))),
      x.rows,
      x.cols,
    );
  },
  logSoftmax(x) {
    const out = new Float32Array(x.data.length);
    for (let r = 0; r < x.rows; r++) {
      let largest = -Infinity;
      for (let c = 0; c < x.cols; c++)
        largest = Math.max(largest, x.data[r * x.cols + c]);
      let total = 0;
      for (let c = 0; c < x.cols; c++)
        total += Math.exp(x.data[r * x.cols + c] - largest);
      const log = Math.log(total);
      for (let c = 0; c < x.cols; c++)
        out[r * x.cols + c] = x.data[r * x.cols + c] - largest - log;
    }
    return make(out, x.rows, x.cols);
  },
  neighbors(node, edge, indices, center) {
    const d = node.cols,
      e = edge.cols,
      k = indices.cols,
      cols = e + d + (center ? d : 0),
      out = new Float32Array(edge.rows * cols);
    for (let r = 0; r < edge.rows; r++) {
      let base = r * cols;
      if (center) {
        out.set(
          node.data.subarray(
            Math.floor(r / k) * d,
            (Math.floor(r / k) + 1) * d,
          ),
          base,
        );
        base += d;
      }
      out.set(edge.data.subarray(r * e, (r + 1) * e), base);
      out.set(
        node.data.subarray(indices.data[r] * d, (indices.data[r] + 1) * d),
        base + e,
      );
    }
    return make(out, edge.rows, cols);
  },
  centerConcat(node, edge, indices) {
    const d = node.cols,
      e = edge.cols,
      k = indices.cols,
      cols = d + e,
      out = new Float32Array(edge.rows * cols);
    for (let r = 0; r < edge.rows; r++) {
      out.set(
        node.data.subarray(Math.floor(r / k) * d, (Math.floor(r / k) + 1) * d),
        r * cols,
      );
      out.set(edge.data.subarray(r * e, (r + 1) * e), r * cols + d);
    }
    return make(out, edge.rows, cols);
  },
  maskEdges(x, indices, mask) {
    return make(
      Float32Array.from(
        x.data,
        (v, i) => v * mask[Math.floor(i / x.cols / indices.cols)],
      ),
      x.rows,
      x.cols,
    );
  },
  sumNeighbors(message, indices, mask, attend) {
    const n = indices.rows,
      k = indices.cols,
      d = message.cols,
      out = new Float32Array(n * d);
    for (let r = 0; r < n; r++)
      for (let c = 0; c < d; c++) {
        let sum = 0;
        for (let j = 0; j < k; j++)
          sum +=
            message.data[(r * k + j) * d + c] *
            (attend ? mask[r] * mask[indices.data[r * k + j]] : 1);
        out[r * d + c] = sum / 30;
      }
    return make(out, n, d);
  },
  graphFeatures(input, prefix) {
    const n = input.seqMask.length,
      k = Math.min(32, n),
      coords = input.coords.data ?? input.coords,
      ca = Array.from({ length: n }, (_, i) =>
        Array.from(coords.slice(i * 111 + 3, i * 111 + 6)),
      ),
      indices = new Float32Array(n * k),
      distances = new Float32Array(n * k),
      mask = input.seqMask;
    for (let i = 0; i < n; i++) {
      const distancesAll = ca.map((p, j) =>
        f(
          mask[i] *
            mask[j] *
            f(
              Math.sqrt(
                f(
                  p.reduce((s, x, c) => f(s + f(f(x - ca[i][c]) ** 2)), 0) +
                    1e-6,
                ),
              ),
            ),
        ),
      );
      const max = Math.max(...distancesAll),
        adjust = distancesAll.map((v, j) =>
          f(v + (1 - mask[i] * mask[j]) * max),
        );
      const order = input.neighborIndices
        ? Array.from(input.neighborIndices.slice(i * k, (i + 1) * k))
        : Array.from({ length: n }, (_, i) => i).sort(
            (a, b) => adjust[a] - adjust[b] || a - b,
          );
      for (let j = 0; j < k; j++) {
        indices[i * k + j] = order[j];
        distances[i * k + j] = adjust[order[j]];
      }
    }
    const frames = Array.from({ length: n }, () => [0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const units = ca.slice(1).map((p, i) => {
      const v = sub(p, ca[i]),
        length = f(Math.sqrt(v.reduce((s, x) => f(s + x * x), 0)));
      return norm(v.map((x) => (length > 3.6 && length < 4 ? x : 0)));
    });
    for (let i = 1; i < n - 2; i++) {
      const u2 = units[i - 1],
        u1 = units[i],
        o = norm(sub(u2, u1)),
        normal = norm(cross(u2, u1));
      frames[i] = [...o, ...normal, ...cross(o, normal)];
    }
    const w = this.weight(prefix + "embeddings.linear.weight").data,
      b = this.weight(prefix + "embeddings.linear.bias").data,
      out = new Float32Array(n * k * 167);
    const shifted = (i, shift) => ca[i + shift] ?? [0, 0, 0],
      pairs = [
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
      ];
    for (let i = 0; i < n; i++)
      for (let t = 0; t < k; t++) {
        const j = indices[i * k + t],
          offset =
            mask[i] === mask[j]
              ? Math.max(
                  0,
                  Math.min(
                    64,
                    input.residueIndex[i] - input.residueIndex[j] + 32,
                  ),
                )
              : 65,
          base = (i * k + t) * 167;
        for (let c = 0; c < 16; c++) out[base + c] = w[c * 66 + offset] + b[c];
        for (let p = 0; p < 9; p++) {
          const [s1, s2] = pairs[p],
            a = shifted(i, s1),
            b = shifted(j, s2);
          const distance =
            p === 0
              ? distances[i * k + t]
              : f(
                  Math.sqrt(
                    f(
                      a.reduce((sum, x, c) => f(sum + f(f(x - b[c]) ** 2)), 0) +
                        1e-6,
                    ),
                  ),
                );
          for (let c = 0; c < 16; c++) {
            const mu = f(2 + (20 * c) / 15),
              v = f(f(distance - mu) / 1.25);
            out[base + 16 + p * 16 + c] = Math.exp(-f(v * v));
          }
        }
        const O = frames[i],
          P = frames[j],
          dx = sub(ca[j], ca[i]);
        const local = norm(
          Array.from({ length: 3 }, (_, r) =>
            f(
              f(f(O[r * 3] * dx[0]) + O[r * 3 + 1] * dx[1]) +
                O[r * 3 + 2] * dx[2],
            ),
          ),
        );
        const R = Array.from({ length: 9 }, (_, c) => {
          const row = Math.floor(c / 3),
            col = c % 3;
          return f(
            f(f(O[row] * P[col]) + O[3 + row] * P[3 + col]) +
              O[6 + row] * P[6 + col],
          );
        });
        const diag = [R[0], R[4], R[8]],
          quat = norm([
            Math.sign(R[7] - R[5]) *
              0.5 *
              Math.sqrt(Math.abs(f(1 + f(f(diag[0] - diag[1]) - diag[2])))),
            Math.sign(R[2] - R[6]) *
              0.5 *
              Math.sqrt(Math.abs(f(1 + f(f(-diag[0] + diag[1]) - diag[2])))),
            Math.sign(R[3] - R[1]) *
              0.5 *
              Math.sqrt(Math.abs(f(1 + f(f(-diag[0] - diag[1]) + diag[2])))),
            0.5 *
              Math.sqrt(Math.max(0, f(1 + f(f(diag[0] + diag[1]) + diag[2])))),
          ]);
        out.set([...local, ...quat], base + 160);
      }
    return { features: make(out, n * k, 167), indices: make(indices, n, k) };
  },
};
