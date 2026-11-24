// Dynamic uniform-driven WGSL compute kernels for FoldFlow 2 (ESM-2 650M + ReFT trunk + IPA).
// Zero shape- or step-specialized shader strings. Strict Firefox wgpu/Naga compatibility.

import { numericWGSL, erfWGSL, trigWGSL } from '../shared/numeric.js';

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
`;

/**
 * Register-tiled batched/strided GEMM supporting tileM in {32, 64} and tileN in {32, 64, 128}, BK = 16.
 * Weight modes: 'F32', 'F16' (packed u32), 'F16_NATIVE' (shader-f16), 'I8' (packed u32 + trailing f32 scales).
 * Preserves bias-first sequential k=0..K-1 FMA accumulation order.
 */
export function gemmSource({ tileM = 32, tileN = 128, wMode = 'F32' } = {}) {
  const rowsPerThread = tileM === 64 ? 4 : 2;
  const colsPerThread = tileN / 16; // 2, 4, or 8 (for tileN = 32, 64, 128)
  const taStride = tileM === 64 ? 17 : 9;
  const taCount = 16 * taStride;
  const tbVecsPerK = tileN / 4; // 8, 16, or 32
  const tbStride = tbVecsPerK + 1; // 9, 17, or 33 (bank-conflict padded)
  const tbCount = 16 * tbStride;
  const useNativeF16 = wMode === 'F16_NATIVE';
  const EB = useNativeF16 ? 'f16' : 'f32';
  const bArrayType = useNativeF16 ? 'f16' : (wMode === 'F32' ? 'f32' : 'u32');

  const totalTbVecs = 16 * tbVecsPerK; // 128, 256, or 512
  let loadBBlock;
  if (wMode === 'F32' || useNativeF16) {
    const iters = Math.ceil(totalTbVecs / 256);
    loadBBlock = `
    for (var e = 0u; e < ${iters}u; e += 1u) {
      let slot = li + e * 256u;
      if (slot < ${totalTbVecs}u) {
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
        tb[kk * ${tbStride}u + r4] = v;
      }
    }`;
  } else if (wMode === 'F16') {
    const activeThreads = tbVecsPerK * 8; // 64, 128, or 256
    const iters = Math.ceil(totalTbVecs / 256);
    loadBBlock = `
    if ((p.sBn & 1u) == 0u && (baseB & 1u) == 0u && p.sBk == 1u) {
      if (li < ${activeThreads}u) {
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
        tb[kkB * ${tbStride}u + r4B] = lo;
        tb[(kkB + 1u) * ${tbStride}u + r4B] = hi;
      }
    } else {
      for (var e = 0u; e < ${iters}u; e += 1u) {
        let slot = li + e * 256u;
        if (slot < ${totalTbVecs}u) {
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
          tb[kk * ${tbStride}u + r4] = v;
        }
      }
    }`;
  } else {
    const activeThreads = tbVecsPerK * 4; // 32, 64, or 128
    const iters = Math.ceil(totalTbVecs / 256);
    loadBBlock = `
    if ((p.sBn & 3u) == 0u && baseB == 0u && p.sBk == 1u) {
      if (li < ${activeThreads}u) {
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
        tb[kk4 * ${tbStride}u + r4B] = v0;
        tb[(kk4 + 1u) * ${tbStride}u + r4B] = v1;
        tb[(kk4 + 2u) * ${tbStride}u + r4B] = v2;
        tb[(kk4 + 3u) * ${tbStride}u + r4B] = v3;
      }
    } else {
      for (var e = 0u; e < ${iters}u; e += 1u) {
        let slot = li + e * 256u;
        if (slot < ${totalTbVecs}u) {
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
          tb[kk * ${tbStride}u + r4] = v;
        }
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
          if (rowA0 + 3u < p.m) { v.w = a[aOff + 3u * p.sAm]; }
        }
      }
      ta[kkA * 9u + r4A] = v;
    }`;

  const rowIdxs = Array.from({ length: rowsPerThread }, (_, i) => i);
  const accType = colsPerThread === 2 ? 'vec2<f32>' : 'vec4<f32>';

  const stepBody = (kkExpr) => {
    const loadAExpr = tileM === 64
      ? `let av = ta[(${kkExpr}) * 17u + tm4]; let a0 = av.x; let a1 = av.y; let a2 = av.z; let a3 = av.w;`
      : `let aq = ta[(${kkExpr}) * 9u + (tm4 >> 1u)]; let a0 = select(aq.x, aq.z, (tm4 & 1u) != 0u); let a1 = select(aq.y, aq.w, (tm4 & 1u) != 0u);`;
    if (colsPerThread === 2) {
      return `{
        ${loadAExpr}
        let bq = vec4<f32>(tb[(${kkExpr}) * ${tbStride}u + (tx >> 1u)]);
        let bv0 = select(bq.xy, bq.zw, (tx & 1u) != 0u);
        ${rowIdxs.map((i) => `acc_${i}_0 = fma(vec2<f32>(a${i}), bv0, acc_${i}_0);`).join(' ')}
      }`;
    }
    if (colsPerThread === 4) {
      return `{
        ${loadAExpr}
        let bv0 = vec4<f32>(tb[(${kkExpr}) * ${tbStride}u + tx]);
        ${rowIdxs.map((i) => `acc_${i}_0 = fma(vec4<f32>(a${i}), bv0, acc_${i}_0);`).join(' ')}
      }`;
    }
    return `{
      ${loadAExpr}
      let bv0 = vec4<f32>(tb[(${kkExpr}) * ${tbStride}u + tx]);
      let bv1 = vec4<f32>(tb[(${kkExpr}) * ${tbStride}u + 16u + tx]);
      ${rowIdxs.map((i) => `acc_${i}_0 = fma(vec4<f32>(a${i}), bv0, acc_${i}_0); acc_${i}_1 = fma(vec4<f32>(a${i}), bv1, acc_${i}_1);`).join(' ')}
    }`;
  };

  const biasInit = colsPerThread === 2 ? `
  var bVec0 = vec2<f32>(0.0);
  if (p.hasBias != 0u) {
    let c0 = n0 + tx * 2u;
    if (c0 < p.n) { bVec0.x = bias[c0]; }
    if (c0 + 1u < p.n) { bVec0.y = bias[c0 + 1u]; }
  }
  ${rowIdxs.map((i) => `var acc_${i}_0 = bVec0;`).join('\n  ')}` : colsPerThread === 4 ? `
  var bVec0 = vec4<f32>(0.0);
  if (p.hasBias != 0u) {
    let c0 = n0 + tx * 4u;
    if (c0 < p.n) { bVec0.x = bias[c0]; }
    if (c0 + 1u < p.n) { bVec0.y = bias[c0 + 1u]; }
    if (c0 + 2u < p.n) { bVec0.z = bias[c0 + 2u]; }
    if (c0 + 3u < p.n) { bVec0.w = bias[c0 + 3u]; }
  }
  ${rowIdxs.map((i) => `var acc_${i}_0 = bVec0;`).join('\n  ')}` : `
  var bVec0 = vec4<f32>(0.0);
  var bVec1 = vec4<f32>(0.0);
  if (p.hasBias != 0u) {
    let c0 = n0 + tx * 4u;
    if (c0 < p.n) { bVec0.x = bias[c0]; }
    if (c0 + 1u < p.n) { bVec0.y = bias[c0 + 1u]; }
    if (c0 + 2u < p.n) { bVec0.z = bias[c0 + 2u]; }
    if (c0 + 3u < p.n) { bVec0.w = bias[c0 + 3u]; }
    let c1 = c0 + 64u;
    if (c1 < p.n) { bVec1.x = bias[c1]; }
    if (c1 + 1u < p.n) { bVec1.y = bias[c1 + 1u]; }
    if (c1 + 2u < p.n) { bVec1.z = bias[c1 + 2u]; }
    if (c1 + 3u < p.n) { bVec1.w = bias[c1 + 3u]; }
  }
  ${rowIdxs.map((i) => `var acc_${i}_0 = bVec0; var acc_${i}_1 = bVec1;`).join('\n  ')}`;

  const storeBlock = rowIdxs.map((i) => {
    if (colsPerThread === 2) {
      return `{
      let m = m0 + tm + ${i}u;
      if (m < p.m) {
        let r0 = select(acc_${i}_0, acc_${i}_0 * p.alpha, p.hasAlpha != 0u);
        let rowC = baseC + m * p.sCm;
        let c0 = n0 + tx * 2u;
        if (c0 < p.n) { c[rowC + c0 * p.sCn] = r0.x; }
        if (c0 + 1u < p.n) { c[rowC + (c0 + 1u) * p.sCn] = r0.y; }
      }
    }`;
    }
    if (colsPerThread === 4) {
      return `{
      let m = m0 + tm + ${i}u;
      if (m < p.m) {
        let r0 = select(acc_${i}_0, acc_${i}_0 * p.alpha, p.hasAlpha != 0u);
        let rowC = baseC + m * p.sCm;
        let c0 = n0 + tx * 4u;
        ${[0, 1, 2, 3].map((j) => `if (c0 + ${j}u < p.n) { c[rowC + (c0 + ${j}u) * p.sCn] = r0.${'xyzw'[j]}; }`).join('\n        ')}
      }
    }`;
    }
    return `{
    let m = m0 + tm + ${i}u;
    if (m < p.m) {
      let r0 = select(acc_${i}_0, acc_${i}_0 * p.alpha, p.hasAlpha != 0u);
      let r1 = select(acc_${i}_1, acc_${i}_1 * p.alpha, p.hasAlpha != 0u);
      let rowC = baseC + m * p.sCm;
      let c0 = n0 + tx * 4u;
      let c1 = c0 + 64u;
      ${[0, 1, 2, 3].map((j) => `if (c0 + ${j}u < p.n) { c[rowC + (c0 + ${j}u) * p.sCn] = r0.${'xyzw'[j]}; }`).join('\n      ')}
      ${[0, 1, 2, 3].map((j) => `if (c1 + ${j}u < p.n) { c[rowC + (c1 + ${j}u) * p.sCn] = r1.${'xyzw'[j]}; }`).join('\n      ')}
    }
  }`;
  }).join('\n  ');

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
var<workgroup> tb: array<vec4<${EB}>, ${tbCount}>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = bias[0];
  let tilesN = (p.n + ${tileN - 1}u) / ${tileN}u;
  let tilesM = (p.m + ${tileM - 1}u) / ${tileM}u;
  let flat = wid.x + wid.y * 65535u;
  let tilesPerBatch = tilesN * tilesM;
  let batch = flat / tilesPerBatch;
  let inBatch = batch < p.batches;
  let bat = min(batch, max(p.batches, 1u) - 1u);
  let tile = flat % tilesPerBatch;
  let m0 = (tile / tilesN) * ${tileM}u;
  let n0 = (tile % tilesN) * ${tileN}u;
  let b0 = bat / max(p.b1, 1u);
  let b1 = bat % max(p.b1, 1u);
  let baseA = p.offA + b0 * p.sA0 + b1 * p.sA1;
  let baseB = p.offB + b0 * p.sB0 + b1 * p.sB1;

  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * ${rowsPerThread}u;

  ${biasInit}

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
  ${storeBlock}
}`;
}

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
  for (var k = 0u; k < p.k; k += 1u) {
    let idx = bBase + k;
    acc = fma(a[aBase + k], ${readW}, acc);
  }
  c[tid] = acc;
}`;
}

/**
 * FoldFlow 2 single-pass ARM64 PyTorch 4-lane cascade-Welford LayerNorm.
 * Exact match to foldflow2/shared/moments.js (a + stride < chunks merge, v / f32(p.width)).
 */
export const FF2_CASCADE_NORM_SOURCE = `
${ACCURATE_DIV_SQRT_WGSL}
struct P {
  rows: u32, width: u32, xStride: u32, xOff: u32,
  hasW: u32, pad0: u32, eps: f32, pad1: u32,
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
      for (var a = 0u; a + stride < chunks; a += stride * 2u) {
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
    if (lane == 0u) { counts[0] = locCounts[0]; }
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
    let variance = v / f32(p.width);
    rowInv = rfdiv(1.0, rfSqrt(variance + p.eps));
  }
  workgroupBarrier();

  if (rawRow >= p.rows) { return; }
  let mu = rowMean;
  let inv = rowInv;
  let xb = p.xOff + row * p.xStride;
  let yb = row * p.width;
  for (var c = tid; c < p.width; c += 128u) {
    var val = (x[xb + c] - mu) * inv;
    if (p.hasW != 0u) {
      val = fma(val, w[c], b[c]);
    }
    y[yb + c] = val;
  }
}
`;

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
 * Pure-GPU node & edge embedding feature builder for FoldFlow 2:
 * Replaces the 980,000-element JS idx.flatMap allocation + 3.92 MB writeBuffer + cat on EVERY denoiser step!
 * mode = 0: single [L, 65]
 * mode = 1: edge [L, L, 120] (fusing pairBase [L, L, 98] + self-cond distance RBFs [L, L, 22])
 */
export const FF2_EMBED_FEATURES_SOURCE = `
${trigWGSL}
${ACCURATE_DIV_SQRT_WGSL}
struct P {
  L: u32, mode: u32, total: u32, pad0: u32,
  te: array<vec4<f32>, 8>,
  denom: array<vec4<f32>, 4>,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> seqIdx: array<f32>;
@group(0) @binding(2) var<storage, read> sc: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn getTe(c: u32) -> f32 {
  return p.te[c >> 2u][c & 3u];
}
fn getDenom(c: u32) -> f32 {
  return p.denom[c >> 2u][c & 3u];
}
fn posVal(indexVal: f32, c: u32) -> f32 {
  let a = (indexVal * 3.1415927410125732) / getDenom(c % 16u);
  let scv = rfSinCos(a);
  return select(scv.y, scv.x, c < 16u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = sc[0];
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  if (idx >= p.total) { return; }
  if (p.mode == 0u) {
    let r = idx / 65u;
    let c = idx % 65u;
    if (c < 32u) { out[idx] = getTe(c); }
    else if (c == 32u) { out[idx] = 0.0; }
    else { out[idx] = posVal(seqIdx[r], c - 33u); }
    return;
  }
  let pair = idx / 120u;
  let c = idx % 120u;
  let r = pair / p.L;
  let col = pair % p.L;
  if (c < 32u) { out[idx] = getTe(c); return; }
  if (c == 32u) { out[idx] = 0.0; return; }
  if (c < 65u) { out[idx] = getTe(c - 33u); return; }
  if (c == 65u) { out[idx] = 0.0; return; }
  if (c < 98u) { out[idx] = posVal(seqIdx[r] - seqIdx[col], c - 66u); return; }
  let k = c - 98u;
  let aBase = r * 3u;
  let bBase = col * 3u;
  var d = 0.0;
  for (var j = 0u; j < 3u; j += 1u) {
    let diff = sc[aBase + j] - sc[bBase + j];
    d += diff * diff;
  }
  d = rfSqrt(d);
  let lo = 0.00001 + f32(k) * (20.0 - 0.00001) / 21.0;
  var hi = 1e8;
  if (k < 21u) { hi = 0.00001 + f32(k + 1u) * (20.0 - 0.00001) / 21.0; }
  out[idx] = select(0.0, 1.0, d > lo && d < hi);
}
`;

export const FF2_GELU_FILL_SOURCE = `
${numericWGSL}
${erfWGSL}
${ACCURATE_DIV_SQRT_WGSL}
struct P { total: u32, mode: u32, val: f32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = x[0];
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.total) { return; }
  if (p.mode == 0u) {
    let z = x[i];
    y[i] = (z * 0.5) * (1.0 + rfErf(rfdiv(z, 1.4142135623730951)));
  } else {
    y[i] = p.val;
  }
}
`;

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

/**
 * Dynamic SO(3) quaternion step kernel (replaces per-step alpha literal shader compilation in geometry.js).
 */
export const FF2_SO3_DQ_SOURCE = `
${trigWGSL}
${ACCURATE_DIV_SQRT_WGSL}
struct P { L: u32, alpha: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> rel: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 64u + li;
  if (i >= p.L * 4u) { return; }
  let r = (i / 4u) * 4u;
  let k = i % 4u;
  let n = rfSqrt(rel[r + 1u] * rel[r + 1u] + rel[r + 2u] * rel[r + 2u] + rel[r + 3u] * rel[r + 3u]);
  let theta = 2.0 * atan2(n, rel[r]);
  let theta2 = theta * theta;
  var scale = 2.0 + theta2 / 12.0 + 7.0 * theta2 * theta2 / 2880.0;
  if (theta > 1e-3) { scale = theta / rfSin(theta / 2.0 + 1e-6); }
  let angle = 0.5 * n / rfSqrt(n * n + rel[r] * rel[r]) * scale * p.alpha;
  var s = p.alpha;
  if (n > 1e-8) { s = rfSin(angle) / n; }
  if (k == 0u) { out[i] = rfCos(angle); }
  else { out[i] = rel[i] * s; }
}
`;
