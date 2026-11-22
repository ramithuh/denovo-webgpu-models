import { constants } from "./constants.js";
export const add = (a, b) => a.map((x, i) => x + b[i]);
export const sub = (a, b) => a.map((x, i) => x - b[i]);
export const mul = (a, s) => a.map((x) => x * s);
export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const unit = (a, eps = 0) => mul(a, 1 / (Math.hypot(...a) + eps));
export const radians = (d) => (d * Math.PI) / 180;
export function dihedral(a, b, c, d) {
  const b0 = sub(a, b),
    b1 = sub(b, c),
    b2 = sub(c, d),
    n1 = cross(b0, b1),
    n2 = cross(b1, b2),
    m1 = cross(n1, unit(b1));
  return (Math.atan2(dot(m1, n2), dot(n1, n2)) * 180) / Math.PI;
}
export function extend([a, b, c], length, angle, chi) {
  const bc = unit(sub(b, c), 1e-6),
    ba = unit(cross(sub(b, a), bc), 1e-6),
    m = cross(ba, bc);
  return add(
    c,
    add(
      mul(bc, length * Math.cos(angle)),
      add(
        mul(m, length * Math.sin(angle) * Math.cos(chi)),
        mul(ba, -length * Math.sin(angle) * Math.sin(chi)),
      ),
    ),
  );
}
/** Exact least-squares proper rigid fit for three noncollinear points.
 * Planar Procrustes is equivalent to upstream's Kabsch SVD for a triad. */
export function align(fixed, mobile) {
  const frame = (pts) => {
    const center = mul(pts.reduce(add, [0, 0, 0]), 1 / 3),
      u = unit(sub(pts[1], pts[0])),
      w = unit(cross(u, sub(pts[2], pts[0]))),
      v = cross(w, u);
    return { center, u, v, w };
  };
  const m = frame(mobile),
    f = frame(fixed);
  let xx = 0,
    xy = 0,
    yx = 0,
    yy = 0;
  for (let i = 0; i < 3; i++) {
    const a = sub(mobile[i], m.center),
      b = sub(fixed[i], f.center),
      x = dot(a, m.u),
      y = dot(a, m.v),
      X = dot(b, f.u),
      Y = dot(b, f.v);
    xx += x * X;
    xy += x * Y;
    yx += y * X;
    yy += y * Y;
  }
  const angle = Math.atan2(xy - yx, xx + yy),
    c = Math.cos(angle),
    s = Math.sin(angle);
  return (p) => {
    const a = sub(p, m.center),
      x = dot(a, m.u),
      y = dot(a, m.v),
      z = dot(a, m.w);
    return add(
      f.center,
      add(mul(f.u, c * x - s * y), add(mul(f.v, s * x + c * y), mul(f.w, z))),
    );
  };
}
export function makeGeometry(weights) {
  const w = (n) => weights["rotamer_builder." + n];
  const lookup = (name, seq) => {
    const t = w(name),
      stride = t.data.length / t.shape[0];
    return Array.from(t.data.slice(seq * stride, (seq + 1) * stride));
  };
  const coords = (name, seq) => {
    const a = lookup(name, seq);
    return Array.from({ length: a.length / 3 }, (_, i) =>
      a.slice(i * 3, i * 3 + 3),
    );
  };
  const idealize = (raw, psi) => {
    const ideal = coords("ideal_prot_aa_coords", 0),
      src = [ideal[0], ideal[1], ideal[2]],
      transform = align(raw, src),
      [N, CA, C] = src.map(transform),
      b = sub(CA, N),
      c = sub(C, CA),
      a = cross(b, c),
      CB = add(
        CA,
        add(mul(a, -0.58273431), add(mul(b, 0.56802827), mul(c, -0.54067466))),
      ),
      O = extend(
        [N, CA, C],
        1.23,
        radians(120.8),
        radians(Number.isFinite(psi) ? psi + 180 : 0),
      );
    return [N, CA, CB, C, O].flat();
  };
  const build = (backbone, sequence, chis, { hydrogens = true } = {}) => {
    const maxAtoms = hydrogens
        ? w("ideal_prot_aa_coords").shape[1]
        : w("ideal_aa_coords").shape[1],
      out = new Float32Array(sequence.length * maxAtoms * 3).fill(NaN);
    for (let r = 0; r < sequence.length; r++) {
      const seq = sequence[r],
        ideal = coords("ideal_aa_coords", seq),
        original = ideal.map((a) => a.slice()),
        indices = lookup("aa_to_chi_angle_atom_index", seq),
        lengths = lookup("ideal_bond_lengths", seq),
        angles = lookup("ideal_bond_angles", seq);
      const place = (i) => {
        const chi = chis[r * 4 + i];
        if (Number.isFinite(chi)) {
          const ids = indices.slice(i * 4, i * 4 + 4);
          ideal[ids[3]] = extend(
            ids.slice(0, 3).map((j) => ideal[j]),
            lengths[i],
            radians(angles[i]),
            radians(chi),
          );
        }
      };
      for (let i = 0; i < 4; i++) place(i);
      const ids = lookup("alignment_indices", seq),
        leftover = lookup("leftover_atom_indices", seq),
        padding = w("ideal_aa_coords").shape[1];
      if (ids.some((i) => i !== padding)) {
        const transform = align(
          ids.map((i) => ideal[i]),
          ids.map((i) => original[i]),
        );
        for (const i of leftover)
          if (i !== padding) ideal[i] = transform(original[i]);
      }
      if (seq === 18) place(2);
      const bb = Array.from({ length: 5 }, (_, i) =>
          Array.from(backbone.slice(r * 15 + i * 3, r * 15 + i * 3 + 3)),
        ),
        gly = seq === 7 || seq === 20,
        mi = gly ? [0, 1, 2] : [0, 1, 4],
        fi = gly ? [0, 1, 3] : [0, 1, 2],
        transform = align(
          fi.map((i) => bb[i]),
          mi.map((i) => ideal[i]),
        );
      let positioned = ideal.map(transform);
      positioned[0] = bb[0];
      positioned[1] = bb[1];
      positioned[2] = bb[3];
      positioned[3] = bb[4];
      while (positioned.length < maxAtoms) positioned.push([NaN, NaN, NaN]);
      if (hydrogens) {
        const prot = coords("ideal_prot_aa_coords", seq),
          tri = lookup("aa_to_hydrogen_alignment_triad_indices", seq),
          hyd = lookup("aa_to_hydrogen_alignment_index", seq),
          groups = tri.length / 3,
          width = hyd.length / groups,
          pad = w("ideal_prot_aa_coords").shape[1] - 1;
        for (let g = 0; g < groups; g++) {
          const hids = hyd
            .slice(g * width, (g + 1) * width)
            .filter((i) => i !== pad);
          if (!hids.length) continue;
          const ids = tri.slice(g * 3, g * 3 + 3),
            t = align(
              ids.map((i) => positioned[i]),
              ids.map((i) => prot[i]),
            );
          for (const i of hids) positioned[i] = t(prot[i]);
        }
      }
      out.set(positioned.flat().slice(0, maxAtoms * 3), r * maxAtoms * 3);
    }
    return { coords: out, atomsPerResidue: maxAtoms };
  };
  // Match upstream CLI postprocessing: disulfide HG removal and epsilon-HIS.
  const cleanup = (built, sequence) => {
    const cys = Array.from(sequence, (s, i) => (s === 4 ? i : -1)).filter(
        (i) => i >= 0,
      ),
      names = constants.atomNames;
    const point = (r, a) =>
      Array.from(
        built.coords.slice(
          (r * built.atomsPerResidue + a) * 3,
          (r * built.atomsPerResidue + a + 1) * 3,
        ),
      );
    const remove = (r, a) =>
      built.coords.fill(
        NaN,
        (r * built.atomsPerResidue + a) * 3,
        (r * built.atomsPerResidue + a + 1) * 3,
      );
    for (const i of cys)
      if (
        cys.some(
          (j) =>
            j !== i &&
            Math.hypot(
              ...sub(
                point(i, names[4].indexOf("SG")),
                point(j, names[4].indexOf("SG")),
              ),
            ) < 2.5,
        )
      )
        remove(i, names[4].indexOf("HG"));
    sequence.forEach((s, i) => {
      if (s === 8) remove(i, names[8].indexOf("HD1"));
    });
    return built;
  };
  return { idealize, build, lookup, cleanup };
}
