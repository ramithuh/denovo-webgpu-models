import { PERIODIC_TABLE, SIDE_CHAIN_TYPES } from "./constants.js";
import { ALPHABET } from "./parser.js";
const f = Math.fround;
const sub = (a, b) => a.map((x, i) => f(x - b[i]));
const dot = (a, b) => f(f(f(a[0] * b[0]) + f(a[1] * b[1])) + f(a[2] * b[2]));
const cross = (a, b) => [
  f(f(a[1] * b[2]) - f(a[2] * b[1])),
  f(f(a[2] * b[0]) - f(a[0] * b[2])),
  f(f(a[0] * b[1]) - f(a[1] * b[0])),
];
const norm = (a) => {
  let n = f(Math.sqrt(dot(a, a)));
  return a.map((x) => f(x / Math.max(1e-12, n)));
};
const distance = (a, b, eps = 1e-6) => {
  let d = sub(a, b);
  return f(Math.sqrt(f(dot(d, d) + eps)));
};
export function virtualCB(X, L) {
  const out = new Float32Array(L * 3);
  for (let i = 0; i < L; i++) {
    const n = Array.from(X.slice(i * 12, i * 12 + 3)),
      ca = Array.from(X.slice(i * 12 + 3, i * 12 + 6)),
      c = Array.from(X.slice(i * 12 + 6, i * 12 + 9)),
      b = sub(ca, n),
      d = sub(c, ca),
      a = cross(b, d);
    out.set(
      ca.map((v, j) =>
        f(
          f(
            f(f(-0.58273431 * a[j]) + f(0.56802827 * b[j])) -
              f(0.54067466 * d[j]),
          ) + v,
        ),
      ),
      i * 3,
    );
  }
  return out;
}
const tokens = (x) =>
  typeof x === "string" ? x.split(/[\s,]+/).filter(Boolean) : (x ?? []);
export function constraints(p, opts = {}) {
  const fixed = tokens(opts.fixedPositions),
    redesign = tokens(opts.redesignedResidues),
    chains = tokens(opts.designChains);
  const match = (r, t) => r.token === t || r.chain === t;
  for (const t of [...fixed, ...redesign, ...chains])
    if (!p.residues.some((r) => match(r, t)))
      throw Error("Unknown residue/chain: " + t);
  const chain_mask = Float32Array.from(p.residues, (r) =>
    (!chains.length || chains.includes(r.chain)) &&
    (!redesign.length || redesign.some((t) => match(r, t))) &&
    !fixed.some((t) => match(r, t))
      ? 1
      : 0,
  );
  const bias = new Float32Array(p.L * 21),
    omit = new Set((opts.omitAAs ?? "").replace(/[\s,]/g, ""));
  if ([...omit].some((a) => !ALPHABET.includes(a)))
    throw Error("Unknown omitted amino acid");
  if ([...ALPHABET.slice(0, 20)].every((a) => omit.has(a)))
    throw Error("Cannot omit all amino acids");
  for (let i = 0; i < p.L; i++)
    for (let a = 0; a < 21; a++)
      bias[i * 21 + a] = omit.has(ALPHABET[a])
        ? -1e8
        : Number(opts.biasAA?.[ALPHABET[a]] ?? 0);
  if (!bias.every(Number.isFinite))
    throw Error("Amino acid biases must be finite");
  const seen = new Set();
  const ties = (opts.tiedPositions ?? []).map((group) => {
    const residues = Array.isArray(group) ? group : group.residues;
    const ids = residues.map((t) => {
      const i =
        typeof t === "number" ? t : p.residues.findIndex((r) => r.token === t);
      if (i < 0 || i >= p.L || seen.has(i))
        throw Error("Unknown or overlapping tied residue: " + t);
      seen.add(i);
      return i;
    });
    if (!ids.length) throw Error("Empty tied group");
    if (ids.some((i) => chain_mask[i] !== chain_mask[ids[0]]))
      throw Error("Tied groups must share design/fixed status");
    if (!chain_mask[ids[0]] && ids.some((i) => p.S[i] !== p.S[ids[0]]))
      throw Error("Fixed tied residues have different identities");
    const weights = Array.isArray(group)
      ? ids.map(() => 1 / ids.length)
      : (group.weights ?? ids.map(() => 1 / ids.length));
    if (weights.length !== ids.length || !weights.every(Number.isFinite))
      throw Error("Invalid tied weights");
    return { ids, weights };
  });
  return { chain_mask, bias, ties };
}
export function featurize(p, opts = {}) {
  const { L } = p,
    M = 25,
    K = Math.min(32, L),
    CB = virtualCB(p.X, L),
    c = constraints(p, opts);
  const Y = new Float32Array(L * M * 3),
    Y_t = new Int32Array(L * M),
    Y_m = new Float32Array(L * M),
    mask_XY = new Float32Array(L),
    E_idx = new Int32Array(L * K);
  const coord = (a, i) => Array.from(a.slice(i * 3, i * 3 + 3));
  for (let i = 0; i < L; i++) {
    const ds = Array.from(p.Y_t, (_, j) => ({
      j,
      d:
        p.mask[i] * p.Y_m[j]
          ? dot(
              sub(coord(CB, i), coord(p.Y, j)),
              sub(coord(CB, i), coord(p.Y, j)),
            )
          : 1000,
    })).sort((a, b) => a.d - b.d || a.j - b.j);
    mask_XY[i] = Math.sqrt(ds[0].d) < 8 ? p.mask[i] * p.Y_m[ds[0].j] : 0;
    ds.slice(0, M).forEach(({ j }, k) => {
      Y.set(p.Y.subarray(j * 3, j * 3 + 3), (i * M + k) * 3);
      Y_t[i * M + k] = p.Y_t[j];
      Y_m[i * M + k] = opts.useAtomContext === false ? 0 : p.Y_m[j];
    });
    const ca = Array.from(p.X.slice(i * 12 + 3, i * 12 + 6));
    const dd = Array.from(
      { length: L },
      (_, j) =>
        p.mask[i] *
        p.mask[j] *
        distance(ca, Array.from(p.X.slice(j * 12 + 3, j * 12 + 6))),
    );
    const max = Math.max(...dd);
    const ns = dd
      .map((d, j) => ({ j, d: d + (1 - p.mask[i] * p.mask[j]) * max }))
      .sort((a, b) => a.d - b.d || a.j - b.j);
    ns.slice(0, K).forEach(({ j }, k) => (E_idx[i * K + k] = j));
  }
  const R_idx = Int32Array.from(p.R_idx);
  let count = 0;
  for (let i = 0; i < L; i++) {
    if (i && p.R_idx[i] === p.R_idx[i - 1]) count++;
    R_idx[i] += count;
  }
  const input = { ...p, ...c, R_idx, CB, Y, Y_t, Y_m, mask_XY, E_idx, L, M, K };
  if (opts.useSideChainContext) {
    for (let i = 0; i < L; i++) {
      const candidates = [];
      for (let k = 0; k < Math.min(K, 16); k++) {
        const j = E_idx[i * K + k];
        for (let a = 5; a < 37; a++)
          candidates.push({
            xyz: coord(p.xyz_37, j * 37 + a),
            t: SIDE_CHAIN_TYPES[a - 5],
            m: p.xyz_37_m[j * 37 + a] * (1 - c.chain_mask[j]),
          });
      }
      for (let a = 0; a < M; a++)
        candidates.push({
          xyz: coord(Y, i * M + a),
          t: Y_t[i * M + a],
          m: Y_m[i * M + a],
        });
      candidates.forEach((q, j) => {
        const d = sub(coord(CB, i), q.xyz);
        q.d = p.mask[i] * q.m ? dot(d, d) : 10000;
        q.index = j;
      });
      candidates.sort((a, b) => a.d - b.d || a.index - b.index);
      candidates.slice(0, M).forEach((q, k) => {
        Y.set(q.xyz, (i * M + k) * 3);
        Y_t[i * M + k] = q.t;
        Y_m[i * M + k] = q.m;
      });
    }
  }
  return input;
}
export function rawFeatures(p) {
  const { L, K, M, X, CB, Y, Y_t, E_idx, R_idx, chain_labels, mask } = p;
  const coord = (a, i) => Array.from(a.slice(i * 3, i * 3 + 3));
  const atoms = Array.from({ length: L }, (_, i) =>
    [0, 1, 2, 3].map((j) => coord(X, i * 4 + j)).concat([coord(CB, i)]),
  );
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
  const maxDist = atoms.map((aa, i) =>
    Math.max(
      ...atoms.map((bb, j) => mask[i] * mask[j] * distance(aa[1], bb[1])),
    ),
  );
  const positional = new Float32Array(L * K * 66),
    rbf = new Float32Array(L * K * 400),
    types = new Float32Array(L * M * 147),
    distances = new Float32Array(L * M * 80),
    angles = new Float32Array(L * M * 4),
    yRbf = new Float32Array(L * M * M * 16);
  const rbfWrite = (out, start, d) => {
    for (let k = 0; k < 16; k++) {
      const mu = f(
        k <= 7 ? f(2 + f(k * f(20 / 15))) : f(22 - f((15 - k) * f(20 / 15))),
      );
      const z = f(f(d - mu) / 1.25);
      out[start + k] = f(Math.exp(-f(z * z)));
    }
  };
  for (let i = 0; i < L; i++) {
    for (let k = 0; k < K; k++) {
      const j = E_idx[i * K + k],
        row = i * K + k,
        d =
          chain_labels[i] === chain_labels[j]
            ? Math.max(0, Math.min(64, R_idx[i] - R_idx[j] + 32))
            : 65;
      positional[row * 66 + d] = 1;
      pairs.forEach(([a, b], n) => {
        let dist = distance(atoms[i][a], atoms[j][b]);
        if (n === 0 && mask[i] * mask[j] === 0) dist = maxDist[i];
        rbfWrite(rbf, row * 400 + n * 16, dist);
      });
    }
    const [n, ca, c] = atoms[i],
      e1 = norm(sub(n, ca)),
      v2 = sub(c, ca),
      pr = dot(e1, v2),
      e2 = norm(v2.map((v, j) => f(v - f(e1[j] * pr)))),
      e3 = cross(e1, e2);
    for (let a = 0; a < M; a++) {
      const row = i * M + a,
        t = Y_t[row],
        y = coord(Y, row);
      types[row * 147 + t] = 1;
      types[row * 147 + 120 + PERIODIC_TABLE[1][t]] = 1;
      types[row * 147 + 139 + PERIODIC_TABLE[2][t]] = 1;
      for (let j = 0; j < 5; j++)
        rbfWrite(distances, row * 80 + j * 16, distance(atoms[i][j], y));
      const v = sub(y, ca),
        local = [dot(e1, v), dot(e2, v), dot(e3, v)],
        xy = f(
          Math.sqrt(
            f(f(f(local[0] * local[0]) + f(local[1] * local[1])) + 1e-8),
          ),
        ),
        xyz = f(f(Math.sqrt(dot(local, local))) + 1e-8);
      angles.set(
        [f(local[0] / xy), f(local[1] / xy), f(xy / xyz), f(local[2] / xyz)],
        row * 4,
      );
      for (let b = 0; b < M; b++)
        rbfWrite(yRbf, (row * M + b) * 16, distance(y, coord(Y, i * M + b)));
    }
  }
  return { positional, rbf, types, distances, angles, yRbf };
}
