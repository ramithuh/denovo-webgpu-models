// Pure-JS CPU reference featurizers used ONLY by CPUAlgebra (cpu/ backend).
// WebGPU executes the corresponding WGSL compute kernels in webgpu/kernels.js directly on GPU (zero CPU fallback).
import motifConstants from './motif-constants.js';

const f = Math.fround;
const ANGLE_BINS = Float32Array.from(motifConstants.angle_bins);
const BB_DIST_BINS = Float32Array.from(motifConstants.bb_dist_bins);

function bucketize20(val, bins) {
  let b = 0;
  for (let k = 0; k < 20; k++) if (val > bins[k]) b++;
  return b;
}

function norm3(v) {
  const n = Math.max(Math.hypot(v[0], v[1], v[2]), 1e-8);
  return [f(v[0] / n), f(v[1] / n), f(v[2] / n)];
}

function cross3(a, b) {
  return [
    f(a[1] * b[2] - a[2] * b[1]),
    f(a[2] * b[0] - a[0] * b[2]),
    f(a[0] * b[1] - a[1] * b[0]),
  ];
}

function dot3(a, b) {
  return f(f(a[0] * b[0]) + f(f(a[1] * b[1]) + f(a[2] * b[2])));
}

function safeAtan2(y, x) {
  if (y === 0 && x === 0) return 0;
  return f(Math.atan2(y, x));
}

function bondAngle(a, b, c) {
  const b0 = norm3([f(b[0] - a[0]), f(b[1] - a[1]), f(b[2] - a[2])]);
  const b1 = norm3([f(c[0] - a[0]), f(c[1] - a[1]), f(c[2] - a[2])]);
  const cosA = dot3(b0, b1);
  const cr = cross3(b0, b1);
  const sinA = f(Math.hypot(cr[0], cr[1], cr[2]));
  return safeAtan2(sinA, cosA);
}

function signedDihedral(a, b, c, d) {
  const b0 = [f(b[0] - a[0]), f(b[1] - a[1]), f(b[2] - a[2])];
  const b1 = [f(c[0] - b[0]), f(c[1] - b[1]), f(c[2] - b[2])];
  const b2 = [f(d[0] - c[0]), f(d[1] - c[1]), f(d[2] - c[2])];
  const n1 = norm3(cross3(b0, b1));
  const n2 = norm3(cross3(b1, b2));
  const cosA = dot3(n1, n2);
  const nc = cross3(n1, n2);
  const sinMag = f(Math.hypot(nc[0], nc[1], nc[2]));
  const vd = dot3(nc, b1);
  const sgn = vd > 0 ? 1 : vd < 0 ? -1 : 0;
  return safeAtan2(f(sgn * sinMag), cosA);
}

function getAtomScaled(coords, r, a, scale) {
  const base = (r * 37 + a) * 3;
  return [f(coords[base] * scale), f(coords[base + 1] * scale), f(coords[base + 2] * scale)];
}

function writeCoorsAbs(out, off, coordsNm, mask, r) {
  for (let a = 0; a < 37; a++) {
    const m = mask[r * 37 + a] > 0 ? 1 : 0;
    const base = (r * 37 + a) * 3;
    out[off + a * 3] = m ? coordsNm[base] : 0;
    out[off + a * 3 + 1] = m ? coordsNm[base + 1] : 0;
    out[off + a * 3 + 2] = m ? coordsNm[base + 2] : 0;
  }
  for (let a = 0; a < 37; a++) out[off + 111 + a] = mask[r * 37 + a] > 0 ? 1 : 0;
}

function writeCoorsRel(out, off, coordsNm, mask, r) {
  const mCa = mask[r * 37 + 1] > 0 ? 1 : 0;
  const caBase = (r * 37 + 1) * 3;
  const cax = mCa ? coordsNm[caBase] : 0;
  const cay = mCa ? coordsNm[caBase + 1] : 0;
  const caz = mCa ? coordsNm[caBase + 2] : 0;
  for (let a = 0; a < 37; a++) {
    const m = mask[r * 37 + a] > 0 ? 1 : 0;
    const base = (r * 37 + a) * 3;
    out[off + a * 3] = m ? f(coordsNm[base] - cax) : 0;
    out[off + a * 3 + 1] = m ? f(coordsNm[base + 1] - cay) : 0;
    out[off + a * 3 + 2] = m ? f(coordsNm[base + 2] - caz) : 0;
  }
  for (let a = 0; a < 37; a++) out[off + 111 + a] = mask[r * 37 + a] > 0 ? 1 : 0;
}

function writeBbTorsions(out, off, coordsNm, L, r, scale) {
  let psi = 0, omega = 0, phi = 0;
  if (r < L - 1) {
    const N0 = getAtomScaled(coordsNm, r, 0, scale);
    const CA0 = getAtomScaled(coordsNm, r, 1, scale);
    const C0 = getAtomScaled(coordsNm, r, 2, scale);
    const N1 = getAtomScaled(coordsNm, r + 1, 0, scale);
    const CA1 = getAtomScaled(coordsNm, r + 1, 1, scale);
    const C1 = getAtomScaled(coordsNm, r + 1, 2, scale);
    psi = signedDihedral(N0, CA0, C0, N1);
    omega = signedDihedral(CA0, C0, N1, CA1);
    phi = signedDihedral(C0, N1, CA1, C1);
  }
  const angles = [psi, omega, phi];
  for (let t = 0; t < 3; t++) {
    const bin = bucketize20(angles[t], ANGLE_BINS);
    out[off + t * 21 + bin] = 1;
  }
}

function writeScAngles(out, off, coordsNm, mask, rtRaw, r, scale) {
  const rt = Math.max(0, Math.min(20, Math.round(rtRaw)));
  const chiAtoms = motifConstants.chi_atom_indices[rt];
  const chiMaskTbl = motifConstants.chi_angles_mask[rt];
  for (let chi = 0; chi < 4; chi++) {
    const [a0, a1, a2, a3] = chiAtoms[chi];
    const mChi =
      chiMaskTbl[chi] > 0 &&
      mask[r * 37 + a0] > 0 &&
      mask[r * 37 + a1] > 0 &&
      mask[r * 37 + a2] > 0 &&
      mask[r * 37 + a3] > 0
        ? 1
        : 0;
    if (mChi) {
      const p0 = getAtomScaled(coordsNm, r, a0, scale);
      const p1 = getAtomScaled(coordsNm, r, a1, scale);
      const p2 = getAtomScaled(coordsNm, r, a2, scale);
      const p3 = getAtomScaled(coordsNm, r, a3, scale);
      let e0 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
      const d0 = Math.sqrt(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2] + 1e-8);
      e0 = [e0[0] / d0, e0[1] / d0, e0[2] / d0];
      let e1 = [p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]];
      const dt = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2];
      e1 = [e1[0] - e0[0] * dt, e1[1] - e0[1] * dt, e1[2] - e0[2] * dt];
      const d1 = Math.sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2] + 1e-8);
      e1 = [e1[0] / d1, e1[1] / d1, e1[2] / d1];
      const e2 = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
      ];
      const rel = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];
      const yRel = e1[0] * rel[0] + e1[1] * rel[1] + e1[2] * rel[2];
      const zRel = e2[0] * rel[0] + e2[1] * rel[1] + e2[2] * rel[2];
      const dn = Math.sqrt(zRel * zRel + yRel * yRel + 1e-8);
      let sn = zRel / dn;
      let cs = yRel / dn;
      const nrm = Math.hypot(sn, cs) + 1e-10;
      sn /= nrm;
      cs /= nrm;
      const ang = safeAtan2(f(sn), f(cs));
      const bin = bucketize20(ang, ANGLE_BINS);
      out[off + chi * 21 + bin] = 1;
    }
    out[off + 84 + chi] = mChi;
  }
}

export function structSeqFeaturesCPU(o, coordsNmT, coordMaskT, resTypeT, extraT, mode, coordScale = 1.0) {
  const L = resTypeT.shape[0];
  const coordsNm = coordsNmT.data;
  const mask = coordMaskT.data;
  const resType = resTypeT.data;
  const extra = extraT?.data;
  const rows = mode === 3 ? extraT.shape[0] : L;
  const width = mode === 0 ? 468 : mode === 1 ? 504 : 205;
  const out = new Float32Array(rows * width);

  for (let row = 0; row < rows; row++) {
    const r = mode === 3 ? Math.round(extra[row]) : row;
    const base = row * width;
    if (mode === 0) {
      // AE3 encoder: [chainBreak(1), seq(20), abs(148), rel(148), bbAngles(63), scAngles(88)]
      out[base] = extra ? extra[r] : 0;
      const rt = Math.round(resType[r]);
      if (rt >= 0 && rt < 20) out[base + 1 + rt] = 1;
      writeCoorsAbs(out, base + 21, coordsNm, mask, r);
      writeCoorsRel(out, base + 169, coordsNm, mask, r);
      writeBbTorsions(out, base + 317, coordsNm, L, r, coordScale);
      writeScAngles(out, base + 380, coordsNm, mask, resType[r], r, coordScale);
    } else if (mode === 1) {
      // LD4 BulkAllAtomXmotifSeqFeat: [abs(148), rel(148), seq(20), scAngles(88), bbAngles(63), mask(37)] * seqMask
      let seqMask = 0;
      for (let a = 0; a < 37; a++) if (mask[r * 37 + a] > 0) { seqMask = 1; break; }
      if (!seqMask) continue;
      writeCoorsAbs(out, base, coordsNm, mask, r);
      writeCoorsRel(out, base + 148, coordsNm, mask, r);
      const rt = Math.round(resType[r]);
      if (rt >= 0 && rt < 20) out[base + 296 + rt] = 1;
      writeScAngles(out, base + 316, coordsNm, mask, resType[r], r, coordScale);
      writeBbTorsions(out, base + 404, coordsNm, L, r, coordScale);
      for (let a = 0; a < 37; a++) out[base + 467 + a] = mask[r * 37 + a] > 0 ? 1 : 0;
    } else {
      // LD5 (mode 2) or LD6/LD7 (mode 3): [abs(148), seq(20), mask(37)] * seqMask
      let seqMask = 0;
      for (let a = 0; a < 37; a++) if (mask[r * 37 + a] > 0) { seqMask = 1; break; }
      if (!seqMask) continue;
      writeCoorsAbs(out, base, coordsNm, mask, r);
      const rt = Math.round(resType[r]);
      if (rt >= 0 && rt < 20) out[base + 148 + rt] = 1;
      for (let a = 0; a < 37; a++) out[base + 168 + a] = mask[r * 37 + a] > 0 ? 1 : 0;
    }
  }
  return o.upload([rows, width], out);
}

export function ae3PairFeaturesCPU(o, coordsNmT, coordMaskT) {
  const L = coordMaskT.shape[0];
  const coordsNm = coordsNmT.data;
  const mask = coordMaskT.data;
  const W = 316;
  const out = new Float32Array(L * L * W);

  for (let i = 0; i < L; i++) {
    const mCaI = mask[i * 37 + 1] > 0;
    const hasCbI = mask[i * 37 + 3] > 0;
    const Ni = getAtomScaled(coordsNm, i, 0, 10);
    const CAi_ang = getAtomScaled(coordsNm, i, 1, 10);
    const CBi_ang = getAtomScaled(coordsNm, i, 3, 10);
    const CAi_nm = getAtomScaled(coordsNm, i, 1, 1);

    for (let j = 0; j < L; j++) {
      const base = (i * L + j) * W;
      const sep = Math.max(0, Math.min(126, i - j + 63));
      out[base + sep] = 1;

      const mCaJ = mask[j * 37 + 1] > 0;
      if (mCaI && mCaJ) {
        const hasCbJ = mask[j * 37 + 3] > 0 ? 1 : 0;
        const atomsJ = [0, 1, 2, 3];
        for (let t = 0; t < 4; t++) {
          const Pj = getAtomScaled(coordsNm, j, atomsJ[t], 1);
          let dist = f(Math.hypot(f(CAi_nm[0] - Pj[0]), f(CAi_nm[1] - Pj[1]), f(CAi_nm[2] - Pj[2])));
          if (t === 3) dist = f(dist * hasCbJ);
          const bin = bucketize20(dist, BB_DIST_BINS);
          out[base + 127 + t * 21 + bin] = 1;
        }
        // Match feature_factory.py line 1507: beta_carbon_pair_mask = has_cb[:, :, None] * has_cb[:, :, None]
        if (hasCbI) {
          const Nj = getAtomScaled(coordsNm, j, 0, 10);
          const CAj_ang = getAtomScaled(coordsNm, j, 1, 10);
          const CBj_ang = getAtomScaled(coordsNm, j, 3, 10);
          const angles = [
            signedDihedral(Ni, CAi_ang, CBi_ang, CBj_ang),
            signedDihedral(Nj, CAj_ang, CBj_ang, CBi_ang),
            bondAngle(CAi_ang, CBi_ang, CBj_ang),
            bondAngle(CAj_ang, CBj_ang, CBi_ang),
            signedDihedral(CAi_ang, CBi_ang, CBj_ang, CAj_ang),
          ];
          for (let t = 0; t < 5; t++) {
            const bin = bucketize20(angles[t], ANGLE_BINS);
            out[base + 211 + t * 21 + bin] = 1;
          }
        }
      }
    }
  }
  return o.upload([L, L, W], out);
}

export function ld4MotifPairFeaturesCPU(o, xMotifT, motifMaskT) {
  const L = motifMaskT.shape[0];
  const coordsNm = xMotifT.data;
  const mask = motifMaskT.data;
  const out = new Float32Array(L * L * 84);
  for (let i = 0; i < L; i++) {
    if (!(mask[i * 37 + 1] > 0)) continue;
    const CAi = getAtomScaled(coordsNm, i, 1, 1);
    for (let j = 0; j < L; j++) {
      if (!(mask[j * 37 + 1] > 0)) continue;
      const base = (i * L + j) * 84;
      const hasCbJ = mask[j * 37 + 3] > 0 ? 1 : 0;
      const atomsJ = [0, 1, 2, 3];
      for (let t = 0; t < 4; t++) {
        const Pj = getAtomScaled(coordsNm, j, atomsJ[t], 1);
        let dist = f(Math.hypot(f(CAi[0] - Pj[0]), f(CAi[1] - Pj[1]), f(CAi[2] - Pj[2])));
        if (t === 3) dist = f(dist * hasCbJ);
        const bin = bucketize20(dist, BB_DIST_BINS);
        out[base + t * 21 + bin] = 1;
      }
    }
  }
  return o.upload([L, L, 84], out);
}

export function padPairCPU(o, pairT, L, N) {
  const C = pairT.shape[2];
  const src = pairT.data;
  const out = new Float32Array(N * N * C);
  for (let r = 0; r < L; r++) {
    for (let c = 0; c < L; c++) {
      const sOff = (r * L + c) * C;
      const dOff = (r * N + c) * C;
      out.set(src.subarray(sOff, sOff + C), dOff);
    }
  }
  return o.upload([N, N, C], out);
}

export function expandSplitRowsCPU(o, twoRowT, L, N) {
  const C = twoRowT.shape[1];
  const src = twoRowT.data;
  const row0 = src.subarray(0, C);
  const row1 = src.subarray(C, 2 * C);
  const out = new Float32Array(N * C);
  for (let r = 0; r < N; r++) {
    out.set(r < L ? row0 : row1, r * C);
  }
  return o.upload([N, C], out);
}
