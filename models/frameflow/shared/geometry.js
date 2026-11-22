export function quatToMatrix(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - w * z),
    2 * (x * z + w * y),
    2 * (x * y + w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - w * x),
    2 * (x * z - w * y),
    2 * (y * z + w * x),
    1 - 2 * (x * x + y * y),
  ];
}
export function matrixToQuat(r) {
  const d = [
      1 + r[0] + r[4] + r[8],
      1 + r[0] - r[4] - r[8],
      1 - r[0] + r[4] - r[8],
      1 - r[0] - r[4] + r[8],
    ],
    i = d.indexOf(Math.max(...d)),
    v = Math.sqrt(Math.max(0, d[i])) * 2;
  let q;
  if (i === 0)
    q = [v / 4, (r[7] - r[5]) / v, (r[2] - r[6]) / v, (r[3] - r[1]) / v];
  if (i === 1)
    q = [(r[7] - r[5]) / v, v / 4, (r[1] + r[3]) / v, (r[2] + r[6]) / v];
  if (i === 2)
    q = [(r[2] - r[6]) / v, (r[1] + r[3]) / v, v / 4, (r[5] + r[7]) / v];
  if (i === 3)
    q = [(r[3] - r[1]) / v, (r[2] + r[6]) / v, (r[5] + r[7]) / v, v / 4];
  const norm = Math.hypot(...q);
  return q.map((x) => x / norm);
}
export function multiply(a, b) {
  const o = new Float32Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) o[3 * i + j] += a[3 * i + k] * b[3 * k + j];
  return o;
}
export function expSO3(v) {
  const t = Math.hypot(...v),
    a = t > 1e-7 ? Math.sin(t) / t : 1 - (t * t) / 6,
    b = t > 1e-7 ? (1 - Math.cos(t)) / (t * t) : 0.5 - (t * t) / 24;
  const [x, y, z] = v;
  return [
    1 - b * (y * y + z * z),
    b * x * y - a * z,
    b * x * z + a * y,
    b * x * y + a * z,
    1 - b * (x * x + z * z),
    b * y * z - a * x,
    b * x * z - a * y,
    b * y * z + a * x,
    1 - b * (x * x + y * y),
  ];
}
export function logSO3(r) {
  const v = [r[7] - r[5], r[2] - r[6], r[3] - r[1]],
    s = Math.hypot(...v) / 2,
    c = (r[0] + r[4] + r[8] - 1) / 2,
    t = Math.atan2(s, c);
  if (Math.abs(t - Math.PI) < 0.010032) {
    const rows = [0, 1, 2].map((i) =>
      [0, 1, 2].map((j) => (r[i * 3 + j] + Number(i === j)) / 2),
    );
    let m = 0;
    for (let i = 1; i < 3; i++)
      if (Math.hypot(...rows[i]) > Math.hypot(...rows[m])) m = i;
    return [0, 1, 2].map(
      (i) => Math.sqrt(Math.max(0, rows[i][i])) * t * Math.sign(rows[m][i]),
    );
  }
  const f = t < 1e-8 ? 0.5 / (1 - (t * t) / 6) : t / (2 * s);
  return v.map((x) => x * f);
}
export function geodesic(a, b, f) {
  const at = [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
  return multiply(a, expSO3(logSO3(multiply(at, b)).map((x) => x * f)));
}
export function randomFrames(length, random = Math.random) {
  const trans_t = new Float32Array(length * 3),
    rotmats_t = new Float32Array(length * 9),
    mean = [0, 0, 0];
  for (let i = 0; i < length; i++) {
    const u = random(),
      v = random() * 2 * Math.PI,
      w = random() * 2 * Math.PI;
    rotmats_t.set(
      quatToMatrix([
        Math.sqrt(u) * Math.cos(w),
        Math.sqrt(1 - u) * Math.sin(v),
        Math.sqrt(1 - u) * Math.cos(v),
        Math.sqrt(u) * Math.sin(w),
      ]),
      9 * i,
    );
    for (let j = 0; j < 3; j++) {
      const n =
        10 *
        Math.sqrt(-2 * Math.log(Math.max(Number.EPSILON, random()))) *
        Math.cos(2 * Math.PI * random());
      trans_t[3 * i + j] = n;
      mean[j] += n / length;
    }
  }
  for (let i = 0; i < trans_t.length; i++) trans_t[i] -= mean[i % 3];
  return { length, trans_t, rotmats_t };
}
// Upstream uses psi=(0,0): oxygen's transverse displacement collapses to zero.
export const IDEAL_BACKBONE = [
  [-0.525, 1.363, 0],
  [0, 0, 0],
  [1.526, 0, 0],
  [2.153, 0, 0],
  [-0.529, -0.774, -1.205],
];
export function backbone(trans, rots) {
  const L = trans.length / 3,
    out = new Float32Array(L * 15);
  for (let i = 0; i < L; i++)
    for (let a = 0; a < 5; a++)
      for (let c = 0; c < 3; c++) {
        let v = trans[3 * i + c];
        for (let k = 0; k < 3; k++)
          v += rots[9 * i + 3 * c + k] * IDEAL_BACKBONE[a][k];
        out[i * 15 + a * 3 + c] = v;
      }
  return out;
}
export function toPDB(coords) {
  if (coords.length % 15)
    throw new RangeError("Expected N,CA,C,O,CB coordinates");
  const names = ["N", "CA", "C", "O", "CB"];
  const rows = [];
  for (let i = 0; i < coords.length / 15; i++)
    for (let a = 0; a < 5; a++) {
      const v = Array.from(
        coords.slice(15 * i + 3 * a, 15 * i + 3 * a + 3),
        (x) => x.toFixed(3).padStart(8),
      ).join("");
      rows.push(
        `ATOM  ${String(i * 5 + a + 1).padStart(5)} ${(" " + names[a]).padEnd(4)} ALA A${String(i + 1).padStart(4)}    ${v}  1.00  0.00          ${names[a][0].padStart(2)}`,
      );
    }
  return rows.join("\n") + "\nTER\nEND\n";
}
export function adjustOxygen(coords, mask) {
  const out = coords.slice(),
    L = coords.length / 15;
  const vec = (i, a, j, b) =>
    [0, 1, 2].map((k) => out[i * 15 + a * 3 + k] - out[j * 15 + b * 3 + k]);
  const norm = (v) => {
    const n = Math.hypot(...v) + 1e-7;
    return v.map((x) => x / n);
  };
  for (let i = 0; i < L; i++) {
    const u = norm(vec(i, 2, i, 1));
    const v =
      i === L - 1 || (mask && !mask[i + 1])
        ? norm(vec(i, 0, i, 1))
        : norm(vec(i, 2, i + 1, 0));
    const d = norm(u.map((x, k) => x + v[k]));
    for (let k = 0; k < 3; k++)
      out[i * 15 + 9 + k] = out[i * 15 + 6 + k] + 1.23 * d[k];
  }
  return out;
}

// PyTorch CPU linspace: float32 endpoints/step, endpoint symmetry and fused multiply-add.
export function linspaceF32(start, end, count) {
  start = Math.fround(start);
  end = Math.fround(end);
  const step = Math.fround(Math.fround(end - start) / (count - 1));
  return Float32Array.from({ length: count }, (_, i) =>
    i < Math.floor(count / 2) ? start + step * i : end - step * (count - i - 1),
  );
}
