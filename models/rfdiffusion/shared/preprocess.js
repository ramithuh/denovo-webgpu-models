import chemical from "./chemical-constants.js";
const f = Math.fround;
const add = (a, b) => a.map((x, i) => f(x + b[i])),
  sub = (a, b) => a.map((x, i) => f(x - b[i])),
  mul = (a, s) => a.map((x) => f(x * f(s))),
  div = (a, s) => a.map((x) => f(x / s)),
  dot = (a, b) => a.reduce((s, x, i) => f(s + f(x * b[i])), 0),
  dotFMA = (a, b) => a.reduce((s, x, i) => f(x * b[i] + s), 0),
  cross = (a, b) => [
    f(a[1] * b[2] - f(a[2] * b[1])),
    f(a[2] * b[0] - f(a[0] * b[2])),
    f(a[0] * b[1] - f(a[1] * b[0])),
  ],
  norm = (a) => f(Math.sqrt(dotFMA(a, a))),
  unit = (a, eps = 0) => div(a, f(norm(a) + f(eps)));
function rigid(atoms) {
  const T = atoms[1],
    x = unit(sub(atoms[2], T), 1e-8),
    v = sub(atoms[0], T),
    y = unit(sub(v, mul(x, dotFMA(x, v))), 1e-8),
    z = cross(x, y);
  return { R: [x, y, z], T };
}
const transform = ({ R, T }, x) =>
  add(
    T,
    [0, 1, 2].map((i) => dotFMA([R[0][i], R[1][i], R[2][i]], x.map(f))),
  );
function dihedral(a, b, c, d, eps = 0) {
  const N = (x) => div(x, f(Math.sqrt(f(dot(x, x) + f(eps))))),
    ab = N(sub(a, b)),
    bc = N(sub(b, c)),
    cd = N(sub(c, d)),
    n1 = N(cross(ab, bc)),
    n2 = N(cross(bc, cd));
  return [dot(n1, n2), dot(cross(n1, bc), n2)];
}
function templateDihedral(a, b, c, d) {
  const b0 = sub(a, b),
    b1 = unit(sub(c, b)),
    b2 = sub(d, c),
    v = sub(b0, mul(b1, dot(b0, b1))),
    w = sub(b2, mul(b1, dot(b2, b1)));
  return f(Math.atan2(dot(cross(b1, v), w), dot(v, w)));
}
function angle(a, b) {
  const n = (x) => div(x, f(Math.sqrt(f(dot(x, x) + f(1e-8))))),
    c = Math.max(-1, Math.min(1, dot(n(a), n(b))));
  return [c, f(Math.sqrt(f(f(1 - f(c * c)) + f(1e-8))))];
}
function cb(at) {
  const b = sub(at[1], at[0]),
    c = sub(at[2], at[1]);
  return add(
    sub(
      add(mul(cross(b, c), -0.58273431), mul(b, 0.56802827)),
      mul(c, 0.54067466),
    ),
    at[1],
  );
}
export function atomsOf(coords, L) {
  const n = coords.length / L / 3;
  if (![3, 4, 14, 27].includes(n))
    throw Error("Expected 3, 4, 14 or 27 atoms per residue");
  return Array.from({ length: L }, (_, i) =>
    Array.from({ length: 27 }, (_, a) =>
      a < n
        ? Array.from(coords.slice((i * n + a) * 3, (i * n + a + 1) * 3))
        : [NaN, NaN, NaN],
    ),
  );
}
export function template2D(coords, L) {
  const atoms = atomsOf(coords, L),
    CB = atoms.map(cb),
    out = new Float32Array(L * L * 44);
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++) {
      const base = (i * L + j) * 44,
        d = norm(sub(CB[i], CB[j]));
      let distance = Number.isNaN(d) ? 999.9 : d + (i === j ? 999.9 : 0);
      if (distance >= 20) distance = 999.9;
      let bin = 0;
      while (bin < 36 && distance > 2.5 + bin * 0.5) bin++;
      out[base + bin] = 1;
      out[base + 43] = 1;
      if (distance < 20) {
        const omega = templateDihedral(atoms[i][1], CB[i], CB[j], atoms[j][1]),
          theta = templateDihedral(atoms[i][0], atoms[i][1], CB[i], CB[j]),
          v = unit(sub(atoms[i][1], CB[i])),
          w = unit(sub(CB[j], CB[i])),
          phi = f(Math.acos(dot(v, w)));
        const angles = [omega, theta, phi];
        for (let k = 0; k < 3; k++) {
          out[base + 37 + k] = Math.sin(angles[k]) || 0;
          out[base + 40 + k] = Math.cos(angles[k]) || 0;
        }
      }
    }
  return out;
}
export function prepareInput({
  coords,
  length,
  sequence,
  idx,
  t = 50,
  selfCondition = null,
  diffusionMask,
}) {
  const L = length ?? coords.length / 9,
    mask = diffusionMask ?? new Uint8Array(L),
    ids = sequence ?? new Int32Array(L).fill(21),
    atoms = atomsOf(coords, L);
  if (ids.length !== L || mask.length !== L)
    throw Error("Sequence or mask length mismatch");
  const seq = new Float32Array(L * 22),
    msa = new Float32Array(L * 48),
    full = new Float32Array(L * 25),
    t1d = new Float32Array(L * 22),
    alpha = new Float32Array(L * 30),
    xyz = new Float32Array(L * 9);
  for (let i = 0; i < L; i++) {
    const aa = ids[i];
    if (!Number.isInteger(aa) || aa < 0 || aa > 21)
      throw Error("Invalid amino acid");
    seq[i * 22 + aa] = 1;
    msa[i * 48 + aa] = 1;
    msa[i * 48 + 22 + aa] = 1;
    full[i * 25 + aa] = 1;
    t1d[i * 22 + Math.min(20, aa)] = 1;
    t1d[i * 22 + 21] = mask[i] ? 1 : 1 - t / 50;
    for (let a = 0; a < 3; a++) xyz.set(atoms[i][a], i * 9 + a * 3);
    if (!mask[i]) for (let a = 3; a < 27; a++) atoms[i][a] = [NaN, NaN, NaN];
    const frame = rigid(atoms[i]);
    atoms[i][0] = transform(frame, [-0.5272, 1.3593, 0]);
    atoms[i][2] = transform(frame, [1.5233, 0, 0]);
  }
  msa[46] = 1;
  msa[(L - 1) * 48 + 47] = 1;
  full[23] = 1;
  full[(L - 1) * 25 + 24] = 1;
  for (let i = 0; i < L; i++) {
    const at = atoms[i],
      aa = Math.min(ids[i], 20),
      vals = Array.from({ length: 10 }, () => [0, 0]),
      valid = Array(10).fill(true);
    vals[0] =
      i < L - 1
        ? dihedral(at[1], at[2], atoms[i + 1][0], atoms[i + 1][1], 1e-8)
        : [1, 0];
    vals[1] =
      i > 0 ? dihedral(atoms[i - 1][2], at[0], at[1], at[2], 1e-8) : [1, 0];
    valid[0] = i < L - 1;
    valid[1] = i > 0;
    vals[2] = dihedral(at[0], at[1], at[2], at[3], 1e-8).map((x) => -x);
    for (let k = 0; k < 4; k++) {
      const indices = chemical.torsionIndices[aa][k];
      valid[3 + k] = indices[3] > 0;
      vals[3 + k] = dihedral(...indices.map((j) => at[j]), 1e-8);
    }
    const nc = mul(add(at[0], at[2]), 0.5),
      ncca = sub(nc, at[1]),
      ncp = sub(at[2], at[0]),
      ncpp = sub(ncp, mul(ncca, f(dot(ncp, ncca) / dot(ncca, ncca))));
    const bend = [
      angle(sub(at[4], at[1]), ncca),
      angle(sub(at[4], at[1]), ncpp),
      angle(sub(at[5], at[4]), sub(at[1], at[4])),
    ];
    for (let k = 0; k < 3; k++) {
      const ref = chemical.referenceAngles[aa][k],
        v = bend[k];
      vals[7 + k] = [dot(v, ref), f(f(v[0] * ref[1]) - f(v[1] * ref[0]))];
    }
    valid[7] = valid[8] = aa !== 7;
    valid[9] = ![0, 7, 20, 21].includes(aa);
    for (let k = 0; k < 10; k++) {
      alpha[i * 30 + k * 3] = Number.isNaN(vals[k][0]) ? 1 : vals[k][0];
      alpha[i * 30 + k * 3 + 1] = Number.isNaN(vals[k][1]) ? 0 : vals[k][1];
      alpha[i * 30 + k * 3 + 2] = Number(valid[k]);
    }
  }
  const xyz_t = selfCondition
    ? Float32Array.from(
        atomsOf(selfCondition, L).flatMap((a) => a.slice(0, 3).flat()),
      )
    : new Float32Array(L * 9);
  return {
    length: L,
    seq,
    msa_latent: msa,
    msa_full: full,
    xyz,
    idx: idx ?? Int32Array.from({ length: L }, (_, i) => i),
    t,
    t1d,
    t2d: selfCondition ? template2D(xyz_t, L) : new Float32Array(L * L * 44),
    xyz_t,
    alpha_t: alpha,
    motif_mask: mask,
  };
}
export function seededRandom(seed = 0) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function gaussian(random) {
  return (
    Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12))) *
    Math.cos(2 * Math.PI * random())
  );
}
