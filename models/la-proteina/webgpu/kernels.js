// Dynamic uniform-driven WGSL compute kernels for La-Proteina.
// All dimensions and step scalars pass through uniform buffers: zero shape/step shader specialization.
// Preserves bias-first FMA accumulation order, ARM64 PyTorch 4-lane cascade Welford moments,
// SLEEF transcendental polynomials, and strict Firefox wgpu/Naga compatibility.

import { numericWGSL } from '../shared/numeric.js';

export const MAX_WG_X = 65535;

export function fold2(groups) {
  const g = Math.max(1, groups | 0);
  return g <= MAX_WG_X ? [g, 1] : [MAX_WG_X, Math.ceil(g / MAX_WG_X)];
}

export const ACCURATE_DIV_SQRT_WGSL = `
fn rfdiv(a: f32, b: f32) -> f32 {
  let signBit = (bitcast<u32>(a) ^ bitcast<u32>(b)) & 0x80000000u;
  let x = abs(a);
  let y = abs(b);
  var bits = bitcast<u32>(x / y);
  if (bits == 0u || bits >= 0x7f800000u) { return a / b; }
  for (var k = 0u; k < 4u; k++) {
    let q = bitcast<f32>(bits);
    let e = fma(-q, y, x);
    let up = e > 0.0;
    let nextBits = select(bits - 1u, bits + 1u, up);
    let halfGap = abs(bitcast<f32>(nextBits) - q) * 0.5;
    let boundary = halfGap * y;
    if (abs(e) > boundary || (abs(e) == boundary && (bits & 1u) == 1u)) {
      bits = nextBits;
    } else {
      break;
    }
  }
  return bitcast<f32>(bits | signBit);
}
fn rfSqrt(x: f32) -> f32 {
  let y = sqrt(x);
  if (x == 0.0) { return 0.0; }
  return y + fma(-y, y, x) / (2.0 * y);
}
fn fmul(a: f32, b: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(a * b));
}
`;

/**
 * Register-tiled batched/strided GEMM:
 *   C = (hasAlpha ? alpha * : 1*) (bias + A * B^T)
 * Supports BM = 32 or BM = 64, BN = 128, BK = 16, @workgroup_size(256).
 * Weight modes: 'F32', 'F16' (packed u32), 'F16_NATIVE' (shader-f16), 'I8' (packed u32 + trailing f32 scales).
 */
export function gemmSource({ tileM = 32, wMode = 'F32' } = {}) {
  const rowsPerThread = tileM === 64 ? 4 : 2;
  const taStride = tileM === 64 ? 17 : 9;
  const taCount = 16 * taStride;
  const useNativeF16 = wMode === 'F16_NATIVE';
  const EB = useNativeF16 ? 'f16' : 'f32';
  const bArrayType = useNativeF16 ? 'f16' : (wMode === 'F32' ? 'f32' : 'u32');

  let loadBBlock;
  if (wMode === 'F32' || useNativeF16) {
    loadBBlock = `
    for (var e = 0u; e < 2u; e += 1u) {
      let slot = li + e * 256u;
      let kk = slot % 16u;
      let r4 = slot / 16u;
      let k = k0 + kk;
      var v = vec4<${EB}>(0.0);
      if (inBatch && k < p.k) {
        let row0 = n0 + r4 * 4u;
        let bAddr0 = baseB + row0 * p.sBn + k * p.sBk;
        if (row0 + 3u < p.n) {
          v = vec4<${EB}>(b[bAddr0], b[bAddr0 + p.sBn], b[bAddr0 + 2u * p.sBn], b[bAddr0 + 3u * p.sBn]);
        } else {
          if (row0 < p.n) { v.x = b[bAddr0]; }
          if (row0 + 1u < p.n) { v.y = b[bAddr0 + p.sBn]; }
          if (row0 + 2u < p.n) { v.z = b[bAddr0 + 2u * p.sBn]; }
          if (row0 + 3u < p.n) { v.w = b[bAddr0 + 3u * p.sBn]; }
        }
      }
      tb[kk * 33u + r4] = v;
    }`;
  } else if (wMode === 'F16') {
    loadBBlock = `
    if ((p.sBn & 1u) == 0u && (baseB & 1u) == 0u && p.sBk == 1u) {
      let kkB = (li % 8u) * 2u;
      let r4B = li / 8u;
      let rowB0 = n0 + r4B * 4u;
      let sBnHalf = p.sBn >> 1u;
      let bOff = ((baseB + rowB0 * p.sBn + k0 + kkB) >> 1u);
      var lo = vec4<f32>(0.0);
      var hi = vec4<f32>(0.0);
      if (inBatch && k0 + kkB < p.k) {
        let hasHi = k0 + kkB + 1u < p.k;
        if (rowB0 + 3u < p.n) {
          let p0 = unpack2x16float(b[bOff]);
          let p1 = unpack2x16float(b[bOff + sBnHalf]);
          let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]);
          let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]);
          lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
          if (hasHi) { hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y); }
        } else {
          if (rowB0 < p.n) { let p0 = unpack2x16float(b[bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
          if (rowB0 + 1u < p.n) { let p1 = unpack2x16float(b[bOff + sBnHalf]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
          if (rowB0 + 2u < p.n) { let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
          if (rowB0 + 3u < p.n) { let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }
        }
      }
      tb[kkB * 33u + r4B] = lo;
      tb[(kkB + 1u) * 33u + r4B] = hi;
    } else {
      for (var e = 0u; e < 2u; e += 1u) {
        let slot = li + e * 256u;
        let kk = slot % 16u;
        let r4 = slot / 16u;
        let k = k0 + kk;
        var v = vec4<f32>(0.0);
        if (inBatch && k < p.k) {
          let row0 = n0 + r4 * 4u;
          for (var u = 0u; u < 4u; u += 1u) {
            if (row0 + u < p.n) {
              let idx = baseB + (row0 + u) * p.sBn + k * p.sBk;
              v[u] = unpack2x16float(b[idx >> 1u])[idx & 1u];
            }
          }
        }
        tb[kk * 33u + r4] = v;
      }
    }`;
  } else {
    // I8 with trailing f32 scales at p.scaleOff
    loadBBlock = `
    if ((p.sBn & 3u) == 0u && baseB == 0u && p.sBk == 1u) {
      if (li < 128u) {
        let kk4 = (li % 4u) * 4u;
        let r4B = li / 4u;
        let rowB0 = n0 + r4B * 4u;
        let sBnWord = p.sBn >> 2u;
        let wOff = rowB0 * sBnWord + ((k0 + kk4) >> 2u);
        var v0 = vec4<f32>(0.0);
        var v1 = vec4<f32>(0.0);
        var v2 = vec4<f32>(0.0);
        var v3 = vec4<f32>(0.0);
        if (inBatch && k0 + kk4 < p.k) {
          for (var u = 0u; u < 4u; u += 1u) {
            if (rowB0 + u < p.n) {
              let sc = bitcast<f32>(b[p.scaleOff + rowB0 + u]);
              let packed = b[wOff + u * sBnWord];
              v0[u] = f32(i32(packed << 24u) >> 24) * sc;
              if (k0 + kk4 + 1u < p.k) { v1[u] = f32(i32((packed >> 8u) << 24u) >> 24) * sc; }
              if (k0 + kk4 + 2u < p.k) { v2[u] = f32(i32((packed >> 16u) << 24u) >> 24) * sc; }
              if (k0 + kk4 + 3u < p.k) { v3[u] = f32(i32(packed) >> 24) * sc; }
            }
          }
        }
        tb[kk4 * 33u + r4B] = v0;
        tb[(kk4 + 1u) * 33u + r4B] = v1;
        tb[(kk4 + 2u) * 33u + r4B] = v2;
        tb[(kk4 + 3u) * 33u + r4B] = v3;
      }
    } else {
      for (var e = 0u; e < 2u; e += 1u) {
        let slot = li + e * 256u;
        let kk = slot % 16u;
        let r4 = slot / 16u;
        let k = k0 + kk;
        var v = vec4<f32>(0.0);
        if (inBatch && k < p.k) {
          let row0 = n0 + r4 * 4u;
          for (var u = 0u; u < 4u; u += 1u) {
            let r = row0 + u;
            if (r < p.n) {
              let idx = baseB + r * p.sBn + k * p.sBk;
              let q = i32((b[idx >> 2u] >> ((idx & 3u) * 8u)) << 24u) >> 24;
              v[u] = f32(q) * bitcast<f32>(b[p.scaleOff + r]);
            }
          }
        }
        tb[kk * 33u + r4] = v;
      }
    }`;
  }

  const loadABlock = tileM === 64 ? `
    {
      let kkA = li % 16u;
      let r4A = li / 16u;
      let rowA0 = m0 + r4A * 4u;
      var v = vec4<f32>(0.0);
      if (inBatch && k0 + kkA < p.k) {
        let aOff = baseA + rowA0 * p.sAm + (k0 + kkA) * p.sAk;
        if (rowA0 + 3u < p.m) {
          v = vec4<f32>(a[aOff], a[aOff + p.sAm], a[aOff + 2u * p.sAm], a[aOff + 3u * p.sAm]);
        } else {
          if (rowA0 < p.m) { v.x = a[aOff]; }
          if (rowA0 + 1u < p.m) { v.y = a[aOff + p.sAm]; }
          if (rowA0 + 2u < p.m) { v.z = a[aOff + 2u * p.sAm]; }
          if (rowA0 + 3u < p.m) { v.w = a[aOff + 3u * p.sAm]; }
        }
      }
      ta[kkA * 17u + r4A] = v;
    }` : `
    if (li < 128u) {
      let kkA = li % 16u;
      let r4A = li / 16u;
      let rowA0 = m0 + r4A * 4u;
      var v = vec4<f32>(0.0);
      if (inBatch && k0 + kkA < p.k) {
        let aOff = baseA + rowA0 * p.sAm + (k0 + kkA) * p.sAk;
        if (rowA0 + 3u < p.m) {
          v = vec4<f32>(a[aOff], a[aOff + p.sAm], a[aOff + 2u * p.sAm], a[aOff + 3u * p.sAm]);
        } else {
          if (rowA0 < p.m) { v.x = a[aOff]; }
          if (rowA0 + 1u < p.m) { v.y = a[aOff + p.sAm]; }
          if (rowA0 + 2u < p.m) { v.z = a[aOff + 2u * p.sAm]; }
          if (rowA0 + 3u < p.n + p.m && rowA0 + 3u < p.m) { v.w = a[aOff + 3u * p.sAm]; }
        }
      }
      ta[kkA * 9u + r4A] = v;
    }`;

  const stepBody = (kkExpr) => tileM === 64 ? `{
      let av = ta[(${kkExpr}) * 17u + tm4];
      let bv0 = vec4<f32>(tb[(${kkExpr}) * 33u + tx]);
      let bv1 = vec4<f32>(tb[(${kkExpr}) * 33u + 16u + tx]);
      acc_0_0 = fma(vec4<f32>(av.x), bv0, acc_0_0); acc_0_1 = fma(vec4<f32>(av.x), bv1, acc_0_1);
      acc_1_0 = fma(vec4<f32>(av.y), bv0, acc_1_0); acc_1_1 = fma(vec4<f32>(av.y), bv1, acc_1_1);
      acc_2_0 = fma(vec4<f32>(av.z), bv0, acc_2_0); acc_2_1 = fma(vec4<f32>(av.z), bv1, acc_2_1);
      acc_3_0 = fma(vec4<f32>(av.w), bv0, acc_3_0); acc_3_1 = fma(vec4<f32>(av.w), bv1, acc_3_1);
    }` : `{
      let aq = ta[(${kkExpr}) * 9u + (tm4 >> 1u)];
      let a0 = select(aq.x, aq.z, (tm4 & 1u) != 0u);
      let a1 = select(aq.y, aq.w, (tm4 & 1u) != 0u);
      let bv0 = vec4<f32>(tb[(${kkExpr}) * 33u + tx]);
      let bv1 = vec4<f32>(tb[(${kkExpr}) * 33u + 16u + tx]);
      acc_0_0 = fma(vec4<f32>(a0), bv0, acc_0_0); acc_0_1 = fma(vec4<f32>(a0), bv1, acc_0_1);
      acc_1_0 = fma(vec4<f32>(a1), bv0, acc_1_0); acc_1_1 = fma(vec4<f32>(a1), bv1, acc_1_1);
    }`;

  const rowIdxs = Array.from({ length: rowsPerThread }, (_, i) => i);

  return `${useNativeF16 ? 'enable f16;\n' : ''}
struct P {
  m: u32, n: u32, k: u32, b1: u32,
  offA: u32, sA0: u32, sA1: u32, sAm: u32,
  sAk: u32, offB: u32, sB0: u32, sB1: u32,
  sBn: u32, sBk: u32, offC: u32, sC0: u32,
  sC1: u32, sCm: u32, sCn: u32, hasBias: u32,
  hasAlpha: u32, alpha: f32, batches: u32, scaleOff: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<${bArrayType}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;

var<workgroup> ta: array<vec4<f32>, ${taCount}>;
var<workgroup> tb: array<vec4<${EB}>, 528>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = bias[0];
  let tilesN = (p.n + 127u) / 128u;
  let tilesM = (p.m + ${tileM - 1}u) / ${tileM}u;
  let flat = wid.x + wid.y * 65535u;
  let tilesPerBatch = tilesN * tilesM;
  let batch = flat / tilesPerBatch;
  let inBatch = batch < p.batches;
  let bat = min(batch, max(p.batches, 1u) - 1u);
  let tile = flat % tilesPerBatch;
  let m0 = (tile / tilesN) * ${tileM}u;
  let n0 = (tile % tilesN) * 128u;
  let b0 = bat / max(p.b1, 1u);
  let b1 = bat % max(p.b1, 1u);
  let baseA = p.offA + b0 * p.sA0 + b1 * p.sA1;
  let baseB = p.offB + b0 * p.sB0 + b1 * p.sB1;

  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * ${rowsPerThread}u;
  let tn0 = tx * 4u;
  let tn1 = tn0 + 64u;

  var bVec0 = vec4<f32>(0.0);
  var bVec1 = vec4<f32>(0.0);
  if (p.hasBias != 0u) {
    let c0 = n0 + tn0;
    if (c0 < p.n) { bVec0.x = bias[c0]; }
    if (c0 + 1u < p.n) { bVec0.y = bias[c0 + 1u]; }
    if (c0 + 2u < p.n) { bVec0.z = bias[c0 + 2u]; }
    if (c0 + 3u < p.n) { bVec0.w = bias[c0 + 3u]; }
    let c1 = n0 + tn1;
    if (c1 < p.n) { bVec1.x = bias[c1]; }
    if (c1 + 1u < p.n) { bVec1.y = bias[c1 + 1u]; }
    if (c1 + 2u < p.n) { bVec1.z = bias[c1 + 2u]; }
    if (c1 + 3u < p.n) { bVec1.w = bias[c1 + 3u]; }
  }
  ${rowIdxs.map((i) => `var acc_${i}_0 = bVec0; var acc_${i}_1 = bVec1;`).join('\n  ')}

  for (var k0 = 0u; k0 < p.k; k0 += 16u) {
    ${loadABlock}
    ${loadBBlock}
    workgroupBarrier();
    if (k0 + 16u <= p.k) {
      ${Array.from({ length: 16 }, (_, kk) => stepBody(`${kk}u`)).join('\n      ')}
    } else {
      let rem = p.k - k0;
      for (var kk = 0u; kk < rem; kk += 1u) {
        ${stepBody('kk')}
      }
    }
    workgroupBarrier();
  }

  if (!inBatch) { return; }
  let baseC = p.offC + b0 * p.sC0 + b1 * p.sC1;
  ${rowIdxs.map((i) => `{
    let m = m0 + tm + ${i}u;
    if (m < p.m) {
      let r0 = select(acc_${i}_0, acc_${i}_0 * p.alpha, p.hasAlpha != 0u);
      let r1 = select(acc_${i}_1, acc_${i}_1 * p.alpha, p.hasAlpha != 0u);
      let rowC = baseC + m * p.sCm;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((j) => {
        const nExpr = j < 4 ? `n0 + tn0 + ${j}u` : `n0 + tn1 + ${j - 4}u`;
        const valExpr = j < 4 ? `r0.${'xyzw'[j]}` : `r1.${'xyzw'[j - 4]}`;
        return `{ let n = ${nExpr}; if (n < p.n) { c[rowC + n * p.sCn] = ${valExpr}; } }`;
      }).join('\n      ')}
    }
  }`).join('\n  ')}
}`;
}

/**
 * Fast zero-barrier GEMV / tiny-M linear (M <= 4, e.g. conditioning projections [1, 256] -> [1, 768]).
 * 1 thread per (row, col), sequential k=0..K-1 bias-first FMA: 100% bit-identical to PyTorch addmm.
 */
export function gemvSource(wMode = 'F32') {
  const useNativeF16 = wMode === 'F16_NATIVE';
  const bArrayType = useNativeF16 ? 'f16' : (wMode === 'F32' ? 'f32' : 'u32');
  const readW = wMode === 'F32'
    ? 'b[idx]'
    : useNativeF16
      ? 'f32(b[idx])'
      : wMode === 'F16'
        ? 'unpack2x16float(b[idx >> 1u])[idx & 1u]'
        : 'f32(i32((b[idx >> 2u] >> ((idx & 3u) * 8u)) << 24u) >> 24) * sc';
  return `${useNativeF16 ? 'enable f16;\n' : ''}
struct P { m: u32, n: u32, k: u32, hasBias: u32, scaleOff: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<${bArrayType}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = bias[0];
  let tid = (wid.x + wid.y * 65535u) * 64u + li;
  if (tid >= p.m * p.n) { return; }
  let row = tid / p.n;
  let col = tid % p.n;
  var acc = 0.0;
  if (p.hasBias != 0u) { acc = bias[col]; }
  ${wMode === 'I8' ? 'let sc = bitcast<f32>(b[p.scaleOff + col]);' : ''}
  let aBase = row * p.k;
  let bBase = col * p.k;
  ${wMode === 'I8' ? `
  if ((p.k & 3u) == 0u) {
    let wBase = bBase >> 2u;
    let words = p.k >> 2u;
    for (var w = 0u; w < words; w += 1u) {
      let packed = b[wBase + w];
      let kk = aBase + (w << 2u);
      acc = fma(a[kk], f32(i32(packed << 24u) >> 24) * sc, acc);
      acc = fma(a[kk + 1u], f32(i32((packed >> 8u) << 24u) >> 24) * sc, acc);
      acc = fma(a[kk + 2u], f32(i32((packed >> 16u) << 24u) >> 24) * sc, acc);
      acc = fma(a[kk + 3u], f32(i32(packed) >> 24) * sc, acc);
    }
    c[tid] = acc;
    return;
  }` : wMode === 'F16' ? `
  if ((p.k & 1u) == 0u) {
    let wBase = bBase >> 1u;
    let words = p.k >> 1u;
    for (var w = 0u; w < words; w += 1u) {
      let pair = unpack2x16float(b[wBase + w]);
      let kk = aBase + (w << 1u);
      acc = fma(a[kk], pair.x, acc);
      acc = fma(a[kk + 1u], pair.y, acc);
    }
    c[tid] = acc;
    return;
  }` : ''}
  for (var k = 0u; k < p.k; k += 1u) {
    let idx = bBase + k;
    acc = fma(a[aBase + k], ${readW}, acc);
  }
  c[tid] = acc;
}`;
}

/**
 * Fused single-pass ARM64 PyTorch 4-lane cascade-Welford LayerNorm (+ optional adaptive modulation).
 * mode = 0: standard LayerNorm (hasW ? fma((x-mu)*inv, w, b) : (x-mu)*inv)
 * mode = 1: adaptive LayerNorm ((x-mu)*inv * sigmoid(w[c]) + b[c])
 * mode = 2: stats-only output [rows, 2] = (mu, inv)
 */
export const CASCADE_NORM_SOURCE = `
${numericWGSL}
${ACCURATE_DIV_SQRT_WGSL}
struct P {
  rows: u32, width: u32, xStride: u32, xOff: u32,
  hasW: u32, mode: u32, eps: f32, pad0: u32,
  splitRow: u32, pad1: u32, pad2: u32, pad3: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

var<workgroup> means: array<f32, 128>;
var<workgroup> vars: array<f32, 128>;
var<workgroup> counts: array<u32, 32>;
var<workgroup> rowMean: f32;
var<workgroup> rowInv: f32;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  _ = w[0]; _ = b[0];
  let rawRow = wid.x + wid.y * 65535u;
  let row = min(rawRow, max(p.rows, 1u) - 1u);
  let chunks = (p.width + 63u) / 64u;
  let slots = chunks * 4u;

  var RECIP = array<f32, 16>(
    1.0, 0.5, 0.3333333432674408, 0.25,
    0.20000000298023224, 0.1666666716337204, 0.1428571492433548, 0.125,
    0.1111111119389534, 0.10000000149011612, 0.09090909361839294, 0.0833333358168602,
    0.07692307978868484, 0.0714285746216774, 0.06666667014360428, 0.0625
  );

  if (tid < slots) {
    let ch = tid / 4u;
    let lane = tid % 4u;
    let nVec = p.width / 4u;
    let count = min(16u, nVec - ch * 16u);
    var mean = 0.0;
    var v = 0.0;
    let base = p.xOff + row * p.xStride + ch * 64u + lane;
    for (var j = 0u; j < count; j += 1u) {
      let z = x[base + j * 4u];
      let d = z - mean;
      mean = fma(d, RECIP[j], mean);
      v = fma(d, z - mean, v);
    }
    means[tid] = mean;
    vars[tid] = v;
    if (lane == 0u) { counts[ch] = count; }
  }
  workgroupBarrier();

  if (tid < 4u) {
    let lane = tid;
    var locCounts: array<u32, 32>;
    for (var ch = 0u; ch < chunks; ch += 1u) {
      locCounts[ch] = counts[ch];
    }
    for (var stride = 1u; stride < chunks; stride *= 2u) {
      for (var a = 0u; a + stride * 2u <= chunks; a += stride * 2u) {
        let bi = a + stride;
        let n0 = locCounts[a];
        let n1 = locCounts[bi];
        let c = rfdiv(f32(n1), f32(n0 + n1));
        let pi = a * 4u + lane;
        let qi = bi * 4u + lane;
        let d = means[qi] - means[pi];
        means[pi] = fma(c, d, means[pi]);
        vars[pi] = vars[pi] + fma((d * d) * c, f32(n0), vars[qi]);
        locCounts[a] = n0 + n1;
      }
    }
    var mMean = 0.0;
    var mVar = 0.0;
    var mCount = 0u;
    for (var end = chunks; end > 0u;) {
      var width = 1u;
      while (end % (width * 2u) == 0u) { width *= 2u; }
      end -= width;
      let n1 = locCounts[end];
      let c = rfdiv(f32(n1), f32(mCount + n1));
      let d = means[end * 4u + lane] - mMean;
      mMean = fma(c, d, mMean);
      mVar = mVar + fma((d * d) * c, f32(mCount), vars[end * 4u + lane]);
      mCount += n1;
    }
    means[lane] = mMean;
    vars[lane] = mVar;
    if (lane == 0u) { counts[0] = mCount; }
  }
  workgroupBarrier();

  if (tid == 0u) {
    let n1 = counts[0];
    var count = 0u;
    var mean = 0.0;
    var v = 0.0;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let c = rfdiv(f32(n1), f32(count + n1));
      let d = means[lane] - mean;
      mean = fma(c, d, mean);
      v = v + fma((d * d) * c, f32(count), vars[lane]);
      count += n1;
    }
    rowMean = mean;
    let variance = rfdiv(v, f32(p.width));
    rowInv = rfdiv(1.0, rfSqrt(variance + p.eps));
  }
  workgroupBarrier();

  if (rawRow >= p.rows) { return; }
  if (p.mode == 2u) {
    if (tid == 0u) {
      y[row * 2u] = rowMean;
      y[row * 2u + 1u] = rowInv;
    }
    return;
  }
  let mu = rowMean;
  let inv = rowInv;
  let xb = p.xOff + row * p.xStride;
  let yb = row * p.width;
  let condOff = select(0u, p.width, p.splitRow > 0u && row >= p.splitRow);
  for (var c = tid; c < p.width; c += 128u) {
    var val = (x[xb + c] - mu) * inv;
    if (p.mode == 1u) {
      let normed = bitcast<f32>(bitcast<u32>(val) ^ p.pad0);
      let g = rfdiv(1.0, 1.0 + rfExp(-w[condOff + c]));
      let prod = bitcast<f32>(bitcast<u32>(normed * g) ^ p.pad0);
      val = prod + b[condOff + c];
    } else if (p.hasW != 0u) {
      val = fma(val, w[c], b[c]);
    }
    y[yb + c] = val;
  }
}
`;

/**
 * Workgroup row-wise Softmax along the last axis (inner == 1), preserving exact 4-lane sum order.
 * Evaluates rfExp ONCE per element across 64 threads and caches in shared memory for len <= 512.
 */
export const ROW_SOFTMAX_SOURCE = `
${numericWGSL}
${ACCURATE_DIV_SQRT_WGSL}
struct P { rows: u32, len: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, 64>;
var<workgroup> rowExp: array<f32, 512>;
var<workgroup> maxVal: f32;
var<workgroup> invSum: f32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let rawRow = wid.x + wid.y * 65535u;
  let row = min(rawRow, max(p.rows, 1u) - 1u);
  let base = row * p.len;

  var mx = -3.402823e38;
  for (var j = tid; j < p.len; j += 64u) {
    mx = max(mx, x[base + j]);
  }
  red[tid] = mx;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
    workgroupBarrier();
  }
  if (tid == 0u) { maxVal = red[0]; }
  workgroupBarrier();
  let m = maxVal;

  if (p.len <= 512u) {
    for (var j = tid; j < p.len; j += 64u) {
      rowExp[j] = rfExp(x[base + j] - m);
    }
    workgroupBarrier();
    if (tid < 4u) {
      var ls = 0.0;
      for (var j = tid; j < p.len; j += 4u) {
        ls = ls + rowExp[j];
      }
      red[tid] = ls;
    }
    workgroupBarrier();
    if (tid == 0u) {
      let s = select((red[0] + red[2]) + (red[1] + red[3]), (red[0] + red[1]) + red[2], p.len < 4u);
      invSum = rfdiv(1.0, s);
    }
    workgroupBarrier();
    if (rawRow < p.rows) {
      let inv = invSum;
      for (var j = tid; j < p.len; j += 64u) {
        y[base + j] = rowExp[j] * inv;
      }
    }
  } else {
    if (tid < 4u) {
      var ls = 0.0;
      for (var j = tid; j < p.len; j += 4u) {
        ls = ls + rfExp(x[base + j] - m);
      }
      red[tid] = ls;
    }
    workgroupBarrier();
    if (tid == 0u) {
      let s = (red[0] + red[2]) + (red[1] + red[3]);
      invSum = rfdiv(1.0, s);
    }
    workgroupBarrier();
    if (rawRow < p.rows) {
      let inv = invSum;
      for (var j = tid; j < p.len; j += 64u) {
        y[base + j] = rfExp(x[base + j] - m) * inv;
      }
    }
  }
}
`;

/**
 * Compute pair feature sparse indices [L*L, 4] (sep, 127+binCa, hasSc ? 157+binSc : 0xffffffffu, 0).
 */
export const PAIR_BINS_SOURCE = `
${ACCURATE_DIV_SQRT_WGSL}
struct P { L: u32, hasSc: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ca: array<f32>;
@group(0) @binding(2) var<storage, read> scCa: array<f32>;
@group(0) @binding(3) var<storage, read> bins: array<f32>;
@group(0) @binding(4) var<storage, read_write> outIdx: array<u32>;

fn distBin(r: u32, c: u32, useSc: bool) -> u32 {
  var s = 0.0;
  for (var k = 0u; k < 3u; k += 1u) {
    let d = select(ca[r * 3u + k] - ca[c * 3u + k], scCa[r * 3u + k] - scCa[c * 3u + k], useSc);
    s = s + d * d;
  }
  let dist = rfSqrt(s);
  var bin = 0u;
  for (var k = 0u; k < 29u; k += 1u) {
    if (dist > bins[k]) { bin += 1u; }
  }
  return bin;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = scCa[0];
  let idx = (wid.x + wid.y * 65535u) * 64u + li;
  let total = p.L * p.L;
  if (idx >= total) { return; }
  let r = idx / p.L;
  let c = idx % p.L;
  let sep = u32(clamp(i32(r) - i32(c) + 63, 0, 126));
  let b0 = 127u + distBin(r, c, false);
  var b1 = 0xffffffffu;
  if (p.hasSc != 0u) {
    b1 = 157u + distBin(r, c, true);
  }
  let base = idx * 4u;
  outIdx[base] = sep;
  outIdx[base + 1u] = b0;
  outIdx[base + 2u] = b1;
  outIdx[base + 3u] = 0u;
}
`;

export const GEOM_HELPERS_WGSL = `
fn rfAtan2(y: f32, x: f32) -> f32 {
  if (y == 0.0 && x == 0.0) { return 0.0; }
  return atan2(y, x);
}
fn norm3(v: vec3<f32>) -> vec3<f32> {
  let n = max(rfSqrt(v.x * v.x + v.y * v.y + v.z * v.z), 1e-8);
  return vec3<f32>(v.x / n, v.y / n, v.z / n);
}
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
fn bondAngle3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  let b0 = norm3(b - a);
  let b1 = norm3(c - a);
  let cosA = dot3(b0, b1);
  let cr = cross3(b0, b1);
  let sinA = rfSqrt(dot3(cr, cr));
  return rfAtan2(sinA, cosA);
}
fn signedDihedral4(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
  let b0 = b - a;
  let b1 = c - b;
  let b2 = d - c;
  let n1 = norm3(cross3(b0, b1));
  let n2 = norm3(cross3(b1, b2));
  let cosA = dot3(n1, n2);
  let nc = cross3(n1, n2);
  let sinMag = rfSqrt(dot3(nc, nc));
  let vd = dot3(nc, b1);
  var sgn = 0.0;
  if (vd > 0.0) { sgn = 1.0; } else if (vd < 0.0) { sgn = -1.0; }
  return rfAtan2(sgn * sinMag, cosA);
}
`;

/**
 * All-atom structural sequence featurizer on GPU (zero CPU fallback, Naga-safe storage lookup tables).
 * mode = 0: AE3 encoder [L, 468] (coordScale = 10.0)
 * mode = 1: LD4 BulkAllAtomXmotifSeqFeat [L, 504] (coordScale = 1.0)
 * mode = 2: LD5 XmotifBulkTipSeqFeat [L, 205]
 * mode = 3: LD6/LD7 FeatureFactoryUidxMotif [M, 205] (extra = activeResidueIndices)
 */
export const STRUCT_SEQ_FEATS_SOURCE = `
${ACCURATE_DIV_SQRT_WGSL}
${GEOM_HELPERS_WGSL}
struct P {
  rows: u32, L: u32, width: u32, mode: u32,
  coordScale: f32, pad0: u32, pad1: u32, pad2: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> coordsNm: array<f32>;
@group(0) @binding(2) var<storage, read> coordMask: array<f32>;
@group(0) @binding(3) var<storage, read> resType: array<f32>;
@group(0) @binding(4) var<storage, read> extra: array<f32>;
@group(0) @binding(5) var<storage, read> motifAux: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

fn angleBin(val: f32) -> u32 {
  var b = 0u;
  for (var k = 0u; k < 20u; k += 1u) {
    if (val > bitcast<f32>(motifAux[420u + k])) { b += 1u; }
  }
  return b;
}

fn atomScaled(r: u32, a: u32, scale: f32) -> vec3<f32> {
  let base = (r * 37u + a) * 3u;
  return vec3<f32>(coordsNm[base] * scale, coordsNm[base + 1u] * scale, coordsNm[base + 2u] * scale);
}

fn writeCoorsAbs(off: u32, r: u32) {
  for (var a = 0u; a < 37u; a += 1u) {
    if (coordMask[r * 37u + a] > 0.0) {
      let base = (r * 37u + a) * 3u;
      out[off + a * 3u] = coordsNm[base];
      out[off + a * 3u + 1u] = coordsNm[base + 1u];
      out[off + a * 3u + 2u] = coordsNm[base + 2u];
      out[off + 111u + a] = 1.0;
    }
  }
}

fn writeCoorsRel(off: u32, r: u32) {
  var cax = 0.0; var cay = 0.0; var caz = 0.0;
  if (coordMask[r * 37u + 1u] > 0.0) {
    let caBase = (r * 37u + 1u) * 3u;
    cax = coordsNm[caBase];
    cay = coordsNm[caBase + 1u];
    caz = coordsNm[caBase + 2u];
  }
  for (var a = 0u; a < 37u; a += 1u) {
    if (coordMask[r * 37u + a] > 0.0) {
      let base = (r * 37u + a) * 3u;
      out[off + a * 3u] = coordsNm[base] - cax;
      out[off + a * 3u + 1u] = coordsNm[base + 1u] - cay;
      out[off + a * 3u + 2u] = coordsNm[base + 2u] - caz;
      out[off + 111u + a] = 1.0;
    }
  }
}

fn writeBbTorsions(off: u32, r: u32) {
  var psi = 0.0; var omega = 0.0; var phi = 0.0;
  if (r + 1u < p.L) {
    let n0 = atomScaled(r, 0u, p.coordScale);
    let ca0 = atomScaled(r, 1u, p.coordScale);
    let c0 = atomScaled(r, 2u, p.coordScale);
    let n1 = atomScaled(r + 1u, 0u, p.coordScale);
    let ca1 = atomScaled(r + 1u, 1u, p.coordScale);
    let c1 = atomScaled(r + 1u, 2u, p.coordScale);
    psi = signedDihedral4(n0, ca0, c0, n1);
    omega = signedDihedral4(ca0, c0, n1, ca1);
    phi = signedDihedral4(c0, n1, ca1, c1);
  }
  out[off + angleBin(psi)] = 1.0;
  out[off + 21u + angleBin(omega)] = 1.0;
  out[off + 42u + angleBin(phi)] = 1.0;
}

fn writeScAngles(off: u32, r: u32) {
  let rt = u32(clamp(i32(round(resType[r])), 0, 20));
  for (var chi = 0u; chi < 4u; chi += 1u) {
    let idxBase = rt * 16u + chi * 4u;
    let a0 = motifAux[idxBase];
    let a1 = motifAux[idxBase + 1u];
    let a2 = motifAux[idxBase + 2u];
    let a3 = motifAux[idxBase + 3u];
    let mChi = (motifAux[336u + rt * 4u + chi] != 0u) &&
      (coordMask[r * 37u + a0] > 0.0) &&
      (coordMask[r * 37u + a1] > 0.0) &&
      (coordMask[r * 37u + a2] > 0.0) &&
      (coordMask[r * 37u + a3] > 0.0);
    if (mChi) {
      let p0 = atomScaled(r, a0, p.coordScale);
      let p1 = atomScaled(r, a1, p.coordScale);
      let p2 = atomScaled(r, a2, p.coordScale);
      let p3 = atomScaled(r, a3, p.coordScale);
      let e0raw = p2 - p1;
      let d0 = rfSqrt(dot3(e0raw, e0raw) + 1e-8);
      let e0 = e0raw / d0;
      let e1raw = p0 - p2;
      let dt = dot3(e0, e1raw);
      let e1ort = e1raw - e0 * dt;
      let d1 = rfSqrt(dot3(e1ort, e1ort) + 1e-8);
      let e1 = e1ort / d1;
      let e2 = cross3(e0, e1);
      let rel = p3 - p2;
      let yRel = dot3(e1, rel);
      let zRel = dot3(e2, rel);
      let dn = rfSqrt(zRel * zRel + yRel * yRel + 1e-8);
      var sn = zRel / dn;
      var cs = yRel / dn;
      let nrm = rfSqrt(sn * sn + cs * cs) + 1e-10;
      sn = sn / nrm;
      cs = cs / nrm;
      let ang = rfAtan2(sn, cs);
      out[off + chi * 21u + angleBin(ang)] = 1.0;
      out[off + 84u + chi] = 1.0;
    }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = extra[0];
  let row = (wid.x + wid.y * 65535u) * 64u + li;
  if (row >= p.rows) { return; }
  let base = row * p.width;
  for (var k = 0u; k < p.width; k += 1u) {
    out[base + k] = 0.0;
  }
  let r = select(row, u32(round(extra[row])), p.mode == 3u);
  if (p.mode == 0u) {
    out[base] = extra[r];
    let rtI = i32(round(resType[r]));
    if (rtI >= 0 && rtI < 20) { out[base + 1u + u32(rtI)] = 1.0; }
    writeCoorsAbs(base + 21u, r);
    writeCoorsRel(base + 169u, r);
    writeBbTorsions(base + 317u, r);
    writeScAngles(base + 380u, r);
  } else {
    var seqMask = false;
    for (var a = 0u; a < 37u; a += 1u) {
      if (coordMask[r * 37u + a] > 0.0) { seqMask = true; break; }
    }
    if (!seqMask) { return; }
    if (p.mode == 1u) {
      writeCoorsAbs(base, r);
      writeCoorsRel(base + 148u, r);
      let rtI = i32(round(resType[r]));
      if (rtI >= 0 && rtI < 20) { out[base + 296u + u32(rtI)] = 1.0; }
      writeScAngles(base + 316u, r);
      writeBbTorsions(base + 404u, r);
      for (var a = 0u; a < 37u; a += 1u) {
        if (coordMask[r * 37u + a] > 0.0) { out[base + 467u + a] = 1.0; }
      }
    } else {
      writeCoorsAbs(base, r);
      let rtI = i32(round(resType[r]));
      if (rtI >= 0 && rtI < 20) { out[base + 148u + u32(rtI)] = 1.0; }
      for (var a = 0u; a < 37u; a += 1u) {
        if (coordMask[r * 37u + a] > 0.0) { out[base + 168u + a] = 1.0; }
      }
    }
  }
}
`;

/**
 * Sparse 10-slot pair bin indices [L*L, 12] for AE3 encoder ([L, L, 316]).
 * Matches feature_factory.py line 1507 (has_cb masked by residue i).
 */
export const AE3_PAIR_BINS_SOURCE = `
${ACCURATE_DIV_SQRT_WGSL}
${GEOM_HELPERS_WGSL}
struct P { L: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> coordsNm: array<f32>;
@group(0) @binding(2) var<storage, read> coordMask: array<f32>;
@group(0) @binding(3) var<storage, read> motifAux: array<u32>;
@group(0) @binding(4) var<storage, read_write> outIdx: array<u32>;

fn angleBin(val: f32) -> u32 {
  var b = 0u;
  for (var k = 0u; k < 20u; k += 1u) {
    if (val > bitcast<f32>(motifAux[420u + k])) { b += 1u; }
  }
  return b;
}

fn bbDistBin(val: f32) -> u32 {
  var b = 0u;
  for (var k = 0u; k < 20u; k += 1u) {
    if (val > bitcast<f32>(motifAux[440u + k])) { b += 1u; }
  }
  return b;
}

fn atomVec(r: u32, a: u32, scale: f32) -> vec3<f32> {
  let base = (r * 37u + a) * 3u;
  return vec3<f32>(coordsNm[base] * scale, coordsNm[base + 1u] * scale, coordsNm[base + 2u] * scale);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 64u + li;
  let total = p.L * p.L;
  if (idx >= total) { return; }
  let r = idx / p.L;
  let c = idx % p.L;
  let base = idx * 12u;
  outIdx[base] = u32(clamp(i32(r) - i32(c) + 63, 0, 126));
  for (var s = 1u; s < 12u; s += 1u) {
    outIdx[base + s] = 0xffffffffu;
  }
  let mCaI = coordMask[r * 37u + 1u] > 0.0;
  let mCaJ = coordMask[c * 37u + 1u] > 0.0;
  if (mCaI && mCaJ) {
    let caiNm = atomVec(r, 1u, 1.0);
    let njNm = atomVec(c, 0u, 1.0);
    let cajNm = atomVec(c, 1u, 1.0);
    let cjNm = atomVec(c, 2u, 1.0);
    let cbjNm = atomVec(c, 3u, 1.0);
    let hasCbJ = select(0.0, 1.0, coordMask[c * 37u + 3u] > 0.0);
    let d0 = caiNm - njNm;
    let d1 = caiNm - cajNm;
    let d2 = caiNm - cjNm;
    let d3 = caiNm - cbjNm;
    outIdx[base + 1u] = 127u + bbDistBin(rfSqrt(dot3(d0, d0)));
    outIdx[base + 2u] = 148u + bbDistBin(rfSqrt(dot3(d1, d1)));
    outIdx[base + 3u] = 169u + bbDistBin(rfSqrt(dot3(d2, d2)));
    outIdx[base + 4u] = 190u + bbDistBin(rfSqrt(dot3(d3, d3)) * hasCbJ);
    if (coordMask[r * 37u + 3u] > 0.0) {
      let niAng = atomVec(r, 0u, 10.0);
      let caiAng = atomVec(r, 1u, 10.0);
      let cbiAng = atomVec(r, 3u, 10.0);
      let njAng = atomVec(c, 0u, 10.0);
      let cajAng = atomVec(c, 1u, 10.0);
      let cbjAng = atomVec(c, 3u, 10.0);
      outIdx[base + 5u] = 211u + angleBin(signedDihedral4(niAng, caiAng, cbiAng, cbjAng));
      outIdx[base + 6u] = 232u + angleBin(signedDihedral4(njAng, cajAng, cbjAng, cbiAng));
      outIdx[base + 7u] = 253u + angleBin(bondAngle3(caiAng, cbiAng, cbjAng));
      outIdx[base + 8u] = 274u + angleBin(bondAngle3(cajAng, cbjAng, cbiAng));
      outIdx[base + 9u] = 295u + angleBin(signedDihedral4(caiAng, cbiAng, cbjAng, cajAng));
    }
  }
}
`;

/**
 * Step-invariant motif pair distance bin indices [L*L, 4] for LD4 ([217..300] of [L, L, 301]).
 */
export const LD4_MOTIF_BINS_SOURCE = `
${ACCURATE_DIV_SQRT_WGSL}
${GEOM_HELPERS_WGSL}
struct P { L: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> xMotif: array<f32>;
@group(0) @binding(2) var<storage, read> motifMask: array<f32>;
@group(0) @binding(3) var<storage, read> motifAux: array<u32>;
@group(0) @binding(4) var<storage, read_write> outBins: array<u32>;

fn bbDistBin(val: f32) -> u32 {
  var b = 0u;
  for (var k = 0u; k < 20u; k += 1u) {
    if (val > bitcast<f32>(motifAux[440u + k])) { b += 1u; }
  }
  return b;
}

fn atomNm(r: u32, a: u32) -> vec3<f32> {
  let base = (r * 37u + a) * 3u;
  return vec3<f32>(xMotif[base], xMotif[base + 1u], xMotif[base + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 64u + li;
  let total = p.L * p.L;
  if (idx >= total) { return; }
  let r = idx / p.L;
  let c = idx % p.L;
  let base = idx * 4u;
  if (motifMask[r * 37u + 1u] > 0.0 && motifMask[c * 37u + 1u] > 0.0) {
    let cai = atomNm(r, 1u);
    let d0 = cai - atomNm(c, 0u);
    let d1 = cai - atomNm(c, 1u);
    let d2 = cai - atomNm(c, 2u);
    let d3 = cai - atomNm(c, 3u);
    let hasCbJ = select(0.0, 1.0, motifMask[c * 37u + 3u] > 0.0);
    outBins[base] = 217u + bbDistBin(rfSqrt(dot3(d0, d0)));
    outBins[base + 1u] = 238u + bbDistBin(rfSqrt(dot3(d1, d1)));
    outBins[base + 2u] = 259u + bbDistBin(rfSqrt(dot3(d2, d2)));
    outBins[base + 3u] = 280u + bbDistBin(rfSqrt(dot3(d3, d3)) * hasCbJ);
  } else {
    outBins[base] = 0xffffffffu;
    outBins[base + 1u] = 0xffffffffu;
    outBins[base + 2u] = 0xffffffffu;
    outBins[base + 3u] = 0xffffffffu;
  }
}
`;

/**
 * Dynamic 7-slot pair bin indices [L*L, 8] for LD4 ([L, L, 301]), combining dynamic CA/scCA bins with static motifBins.
 */
export const LD4_PAIR_BINS_SOURCE = `
${ACCURATE_DIV_SQRT_WGSL}
struct P { L: u32, hasSc: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ca: array<f32>;
@group(0) @binding(2) var<storage, read> scCa: array<f32>;
@group(0) @binding(3) var<storage, read> bins: array<f32>;
@group(0) @binding(4) var<storage, read> motifBins: array<u32>;
@group(0) @binding(5) var<storage, read_write> outIdx: array<u32>;

fn distBin(r: u32, c: u32, useSc: bool) -> u32 {
  var s = 0.0;
  for (var k = 0u; k < 3u; k += 1u) {
    let d = select(ca[r * 3u + k] - ca[c * 3u + k], scCa[r * 3u + k] - scCa[c * 3u + k], useSc);
    s = s + d * d;
  }
  let dist = rfSqrt(s);
  var bin = 0u;
  for (var k = 0u; k < 29u; k += 1u) {
    if (dist > bins[k]) { bin += 1u; }
  }
  return bin;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = scCa[0];
  let idx = (wid.x + wid.y * 65535u) * 64u + li;
  let total = p.L * p.L;
  if (idx >= total) { return; }
  let r = idx / p.L;
  let c = idx % p.L;
  let sep = u32(clamp(i32(r) - i32(c) + 63, 0, 126));
  let b0 = 127u + distBin(r, c, false);
  var b1 = 0xffffffffu;
  if (p.hasSc != 0u) {
    b1 = 157u + distBin(r, c, true);
  }
  let mb = idx * 4u;
  let base = idx * 8u;
  outIdx[base] = sep;
  outIdx[base + 1u] = b0;
  outIdx[base + 2u] = b1;
  outIdx[base + 3u] = motifBins[mb];
  outIdx[base + 4u] = motifBins[mb + 1u];
  outIdx[base + 5u] = motifBins[mb + 2u];
  outIdx[base + 6u] = motifBins[mb + 3u];
  outIdx[base + 7u] = 0xffffffffu;
}
`;

/**
 * Zero-pad pair representation [L, L, C] -> [N, N, C] for unindexed motif models (LD6/LD7).
 */
export const PAD_PAIR_SOURCE = `
struct P { L: u32, N: u32, C: u32, total: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.total) { return; }
  let col = i % p.C;
  let pair = i / p.C;
  let r = pair / p.N;
  let c = pair % p.N;
  if (r < p.L && c < p.L) {
    dst[i] = src[(r * p.L + c) * p.C + col];
  } else {
    dst[i] = 0.0;
  }
}
`;

/**
 * Sparse multi-hot linear projection for initial pair features:
 *   Y[pair, col] = bias[col] + sum_{active s} W[col, indices[pair * stride + s]]
 * Eliminates the [L, L, 217/157/301/316] dense one-hot allocation and GEMM.
 */
export function sparsePairLinearSource(wMode = 'F32') {
  const useNativeF16 = wMode === 'F16_NATIVE';
  const bArrayType = useNativeF16 ? 'f16' : (wMode === 'F32' ? 'f32' : 'u32');
  const fetchW = wMode === 'F32'
    ? 'w[idx]'
    : useNativeF16
      ? 'f32(w[idx])'
      : wMode === 'F16'
        ? 'unpack2x16float(w[idx >> 1u])[idx & 1u]'
        : 'f32(i32((w[idx >> 2u] >> ((idx & 3u) * 8u)) << 24u) >> 24) * sc';
  return `${useNativeF16 ? 'enable f16;\n' : ''}
struct P { pairs: u32, n: u32, k: u32, hasBias: u32, scaleOff: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<${bArrayType}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = bias[0];
  let tid = (wid.x + wid.y * 65535u) * 256u + li;
  if (tid >= p.pairs * p.n) { return; }
  let pair = tid / p.n;
  let col = tid % p.n;
  ${wMode === 'I8' ? 'let sc = bitcast<f32>(w[p.scaleOff + col]);' : ''}
  var acc = 0.0;
  if (p.hasBias != 0u) { acc = bias[col]; }
  let rowW = col * p.k;
  if (p.pad0 == 0u) {
    let ib = pair * 4u;
    { let idx = rowW + indices[ib]; acc = fma(1.0, ${fetchW}, acc); }
    { let idx = rowW + indices[ib + 1u]; acc = fma(1.0, ${fetchW}, acc); }
    let i2 = indices[ib + 2u];
    if (i2 != 0xffffffffu) {
      let idx = rowW + i2;
      acc = fma(1.0, ${fetchW}, acc);
    }
  } else {
    let ib = pair * p.pad1;
    for (var s = 0u; s < p.pad0; s += 1u) {
      let colIdx = indices[ib + s];
      if (colIdx != 0xffffffffu) {
        let idx = rowW + colIdx;
        acc = fma(1.0, ${fetchW}, acc);
      }
    }
  }
  y[tid] = acc;
}`;
}

/**
 * Fused tiled pair_norm (using precomputed [L*L, 2] Welford stats) + to_bias linear + [2,0,1] permute -> [H, L, L].
 * Workgroup 16x16 (256 threads): 16 pair rows x 16 head cols (active for col < H=12), tiled over K=256 in steps of 16.
 */
export function pairBiasPermuteSource(wMode = 'F32') {
  const useNativeF16 = wMode === 'F16_NATIVE';
  const bArrayType = useNativeF16 ? 'f16' : (wMode === 'F32' ? 'f32' : 'u32');
  const fetchW = wMode === 'F32'
    ? 'bw[idx]'
    : useNativeF16
      ? 'f32(bw[idx])'
      : wMode === 'F16'
        ? 'unpack2x16float(bw[idx >> 1u])[idx & 1u]'
        : 'f32(i32((bw[idx >> 2u] >> ((idx & 3u) * 8u)) << 24u) >> 24) * bitcast<f32>(bw[p.scaleOff + col])';
  return `${useNativeF16 ? 'enable f16;\n' : ''}
struct P { pairs: u32, k: u32, heads: u32, hasBias: u32, scaleOff: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pair: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> nw: array<f32>;
@group(0) @binding(4) var<storage, read> nb: array<f32>;
@group(0) @binding(5) var<storage, read> bw: array<${bArrayType}>;
@group(0) @binding(6) var<storage, read> bb: array<f32>;
@group(0) @binding(7) var<storage, read_write> outBias: array<f32>;

var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = bb[0];
  let tilePair = wid.x + wid.y * 65535u;
  let rLocal = li / 16u;
  let cLocal = li % 16u;
  let row = tilePair * 16u + rLocal;
  let col = cLocal;

  var mu = 0.0;
  var inv = 0.0;
  if (row < p.pairs) {
    mu = stats[row * 2u];
    inv = stats[row * 2u + 1u];
  }
  var acc = 0.0;
  if (p.hasBias != 0u && col < p.heads) { acc = bb[col]; }

  for (var k0 = 0u; k0 < p.k; k0 += 16u) {
    var aVal = 0.0;
    let kkA = k0 + cLocal;
    if (row < p.pairs && kkA < p.k) {
      let normed = (pair[row * p.k + kkA] - mu) * inv;
      aVal = fma(normed, nw[kkA], nb[kkA]);
    }
    ta[rLocal * 16u + cLocal] = aVal;

    var bVal = 0.0;
    let kkB = k0 + rLocal;
    if (col < p.heads && kkB < p.k) {
      let idx = col * p.k + kkB;
      bVal = ${fetchW};
    }
    tb[rLocal * 16u + cLocal] = bVal;
    workgroupBarrier();

    for (var kk = 0u; kk < 16u; kk += 1u) {
      acc = fma(ta[rLocal * 16u + kk], tb[kk * 16u + cLocal], acc);
    }
    workgroupBarrier();
  }

  if (row < p.pairs && col < p.heads) {
    outBias[col * p.pairs + row] = acc;
  }
}`;
}

/**
 * Tiled Multi-Head Attention Logits:
 *   out[h, i, j] = (sum_{d=0..D-1} Q[i, h, d] * K[j, h, d]) * scale + pairBias[h, i, j]
 * 32x32 output tile per (h, iTile, jTile), @workgroup_size(256) (16x16 threads, 2x2 outputs/thread), BK = 16.
 */
export const ATTN_LOGITS_SOURCE = `
struct P { L: u32, H: u32, D: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> pairBias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> ta: array<f32, 512>;
var<workgroup> tb: array<f32, 512>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesL = (p.L + 31u) / 32u;
  let flat = wid.x + wid.y * 65535u;
  let tilesPerHead = tilesL * tilesL;
  let h = flat / tilesPerHead;
  let inHead = h < p.H;
  let head = min(h, max(p.H, 1u) - 1u);
  let rem = flat % tilesPerHead;
  let i0 = (rem / tilesL) * 32u;
  let j0 = (rem % tilesL) * 32u;

  let ty = li / 16u;
  let tx = li % 16u;
  let stride = p.H * p.D;

  var acc00 = 0.0; var acc01 = 0.0;
  var acc10 = 0.0; var acc11 = 0.0;

  for (var d0 = 0u; d0 < p.D; d0 += 16u) {
    for (var e = 0u; e < 2u; e += 1u) {
      let slot = li + e * 256u;
      let r = slot / 16u;
      let d = slot % 16u;
      var qv = 0.0;
      var kv = 0.0;
      if (inHead && d0 + d < p.D) {
        if (i0 + r < p.L) { qv = q[(i0 + r) * stride + head * p.D + d0 + d]; }
        if (j0 + r < p.L) { kv = k[(j0 + r) * stride + head * p.D + d0 + d]; }
      }
      ta[r * 16u + d] = qv;
      tb[r * 16u + d] = kv;
    }
    workgroupBarrier();
    let r0 = ty * 2u;
    let c0 = tx * 2u;
    let dEnd = min(16u, p.D - d0);
    for (var d = 0u; d < dEnd; d += 1u) {
      let qa0 = ta[r0 * 16u + d];
      let qa1 = ta[(r0 + 1u) * 16u + d];
      let kb0 = tb[c0 * 16u + d];
      let kb1 = tb[(c0 + 1u) * 16u + d];
      acc00 = fma(qa0, kb0, acc00);
      acc01 = fma(qa0, kb1, acc01);
      acc10 = fma(qa1, kb0, acc10);
      acc11 = fma(qa1, kb1, acc11);
    }
    workgroupBarrier();
  }

  if (!inHead) { return; }
  let r0 = i0 + ty * 2u;
  let c0 = j0 + tx * 2u;
  let hOff = head * p.L * p.L;
  let zMask = p.H >> 31u;
  if (r0 < p.L && c0 < p.L) { let idx = hOff + r0 * p.L + c0; out[idx] = bitcast<f32>(bitcast<u32>(acc00 * p.scale) ^ zMask) + pairBias[idx]; }
  if (r0 < p.L && c0 + 1u < p.L) { let idx = hOff + r0 * p.L + c0 + 1u; out[idx] = bitcast<f32>(bitcast<u32>(acc01 * p.scale) ^ zMask) + pairBias[idx]; }
  if (r0 + 1u < p.L && c0 < p.L) { let idx = hOff + (r0 + 1u) * p.L + c0; out[idx] = bitcast<f32>(bitcast<u32>(acc10 * p.scale) ^ zMask) + pairBias[idx]; }
  if (r0 + 1u < p.L && c0 + 1u < p.L) { let idx = hOff + (r0 + 1u) * p.L + c0 + 1u; out[idx] = bitcast<f32>(bitcast<u32>(acc11 * p.scale) ^ zMask) + pairBias[idx]; }
}
`;

/**
 * Tiled Multi-Head Attention Context + Sigmoid Gate (reads V directly from QKV at vOffset):
 *   out[i, h*D + d] = (sum_{j=0..L-1} probs[h, i, j] * QKV[j, vOffset + h*D + d]) * sigmoid(gate[i, h*D + d])
 * Tile: 32 rows (i) x 64 cols (d = whole head D=64!) per (h, iTile), @workgroup_size(256) (16x16 threads, 2x4 outputs/thread).
 */
export const ATTN_CONTEXT_GATED_SOURCE = `
${numericWGSL}
${ACCURATE_DIV_SQRT_WGSL}
struct P { L: u32, H: u32, D: u32, qkvStride: u32, vOffset: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> qkv: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> ta: array<f32, 512>;
var<workgroup> tb: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesI = (p.L + 31u) / 32u;
  let flat = wid.x + wid.y * 65535u;
  let h = flat / tilesI;
  let inHead = h < p.H;
  let head = min(h, max(p.H, 1u) - 1u);
  let i0 = (flat % tilesI) * 32u;

  let ty = li / 16u;
  let tx = li % 16u;
  let row0 = i0 + ty * 2u;
  let col0 = tx * 4u;
  let outStride = p.H * p.D;

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);

  for (var j0 = 0u; j0 < p.L; j0 += 16u) {
    for (var e = 0u; e < 2u; e += 1u) {
      let slot = li + e * 256u;
      let r = slot / 16u;
      let kj = slot % 16u;
      var pv = 0.0;
      if (inHead && i0 + r < p.L && j0 + kj < p.L) {
        pv = probs[(head * p.L + (i0 + r)) * p.L + (j0 + kj)];
      }
      ta[r * 16u + kj] = pv;
    }
    for (var e = 0u; e < 4u; e += 1u) {
      let slot = li + e * 256u;
      let kj = slot / 64u;
      let d = slot % 64u;
      var vv = 0.0;
      if (inHead && j0 + kj < p.L && d < p.D) {
        vv = qkv[(j0 + kj) * p.qkvStride + p.vOffset + head * p.D + d];
      }
      tb[kj * 64u + d] = vv;
    }
    workgroupBarrier();
    let rem = min(16u, p.L - j0);
    for (var kj = 0u; kj < rem; kj += 1u) {
      let p0 = ta[(ty * 2u) * 16u + kj];
      let p1 = ta[(ty * 2u + 1u) * 16u + kj];
      let vb = kj * 64u + col0;
      let vVec = vec4<f32>(tb[vb], tb[vb + 1u], tb[vb + 2u], tb[vb + 3u]);
      acc0 = fma(vec4<f32>(p0), vVec, acc0);
      acc1 = fma(vec4<f32>(p1), vVec, acc1);
    }
    workgroupBarrier();
  }

  if (!inHead) { return; }
  for (var r = 0u; r < 2u; r += 1u) {
    let row = row0 + r;
    if (row < p.L) {
      let vals = select(acc0, acc1, r == 1u);
      for (var u = 0u; u < 4u; u += 1u) {
        let d = col0 + u;
        if (d < p.D) {
          let idx = row * outStride + head * p.D + d;
          let g = rfdiv(1.0, 1.0 + rfExp(-gate[idx]));
          out[idx] = vals[u] * g;
        }
      }
    }
  }
}
`;

/**
 * Fused SwiGLU: out[r, c] = h[r, c] * (h[r, width+c] / (1 + exp(-h[r, width+c])))
 * and Fused Residual Scale Add: out[r, c] = x[r, c] + h[r, c] * sigmoid(gamma[c])
 */
export const FUSED_OPS_SOURCE = `
${numericWGSL}
${ACCURATE_DIV_SQRT_WGSL}
struct P {
  total: u32, width: u32, mode: u32, pad0: u32,
  splitRow: u32, pad1: u32, pad2: u32, pad3: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = b[0]; _ = c[0];
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.total) { return; }
  let col = i % p.width;
  let row = i / p.width;
  let condOff = select(0u, p.width, p.splitRow > 0u && row >= p.splitRow);
  if (p.mode == 0u) {
    // SwiGLU: a is [rows, 2*width]
    let base = row * (p.width * 2u);
    let gate = a[base + p.width + col];
    y[i] = a[base + col] * rfdiv(gate, 1.0 + rfExp(-gate));
  } else if (p.mode == 1u) {
    // Residual + scale: y = a + b * sigmoid(c[condOff + col])
    let g = rfdiv(1.0, 1.0 + rfExp(-c[condOff + col]));
    let prod = bitcast<f32>(bitcast<u32>(b[i] * g) ^ p.pad0);
    y[i] = a[i] + prod;
  } else if (p.mode == 3u) {
    // Constant fill: y = bitcast<f32>(p.pad0)
    y[i] = bitcast<f32>(p.pad0);
  } else {
    // Scale only: y = a * sigmoid(b[condOff + col])
    let g = rfdiv(1.0, 1.0 + rfExp(-b[condOff + col]));
    y[i] = a[i] * g;
  }
}
`;

/**
 * Dynamic uniform-parameterized elementwise binary/unary kernel:
 * Eliminates baking step scalars (t, 1-t, dt, g, std) and shapes into WGSL strings!
 */
export function dynamicElementSource(expr, numInputs) {
  const bindings = Array.from({ length: numInputs }, (_, j) =>
    `@group(0) @binding(${j + 1}) var<storage, read> in${j}: array<f32>;`
  ).join('\n');
  return `
${numericWGSL}
${ACCURATE_DIV_SQRT_WGSL}
struct P {
  total: u32, rank: u32, pad0: u32, pad1: u32,
  outStrides: vec4<u32>,
  outDims: vec4<u32>,
  in0Strides: vec4<u32>,
  in1Strides: vec4<u32>,
  scalars: vec4<f32>,
};
@group(0) @binding(0) var<uniform> p: P;
${bindings}
@group(0) @binding(${numInputs + 1}) var<storage, read_write> o: array<f32>;

fn idxOf(coords: vec4<u32>, st: vec4<u32>) -> u32 {
  return coords.x * st.x + coords.y * st.y + coords.z * st.z + coords.w * st.w;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.total) { return; }
  let coords = vec4<u32>(
    (i / p.outStrides.x) % p.outDims.x,
    (i / p.outStrides.y) % p.outDims.y,
    (i / p.outStrides.z) % p.outDims.z,
    (i / p.outStrides.w) % p.outDims.w,
  );
  ${numInputs >= 1 ? 'let v0 = in0[idxOf(coords, p.in0Strides)];' : ''}
  ${numInputs >= 2 ? 'let v1 = in1[idxOf(coords, p.in1Strides)];' : ''}
  o[i] = ${expr};
}
`;
}
