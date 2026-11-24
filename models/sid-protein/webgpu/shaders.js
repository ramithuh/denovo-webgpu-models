import { numericWGSL, divisionWGSL } from "../shared/numeric.js";

/** All shape and scalar parameters arrive via @binding(0) uniform buffer; zero shape-specialized WGSL. */
export const MAX_WG_X = 65535;

export function fold2(groups) {
  const g = Math.max(1, groups | 0);
  return g <= MAX_WG_X ? [g, 1] : [MAX_WG_X, Math.ceil(g / MAX_WG_X)];
}

/**
 * Register-tiled GEMM (BM = 32 or 64, BN = 128, BK = 16, @workgroup_size(256)).
 * Preserves bias-first sequential k = 0..K-1 FMA accumulation order.
 */
export function matmul(dtype, hasBias, native = false, tileM = 32) {
  const useNativeF16 = dtype === "F16" && native;
  const EB = useNativeF16 ? "f16" : "f32";
  const wType = dtype === "F32" ? "f32" : useNativeF16 ? "f16" : "u32";
  const rowsPerThread = tileM === 64 ? 4 : 2;
  const taStride = tileM === 64 ? 17 : 9;
  const taCount = 16 * taStride;

  let loadBBlock;
  if (dtype === "F32" || useNativeF16) {
    loadBBlock = `
    for (var e = 0u; e < 2u; e += 1u) {
      let slot = li + e * 256u;
      let kk = slot % 16u;
      let r4 = slot / 16u;
      let k = k0 + kk;
      var v = vec4<${EB}>(0.0);
      if (k < p.K) {
        let row0 = n0 + r4 * 4u;
        let bAddr0 = row0 * p.K + k;
        if (row0 + 3u < p.N) {
          v = vec4<${EB}>(w[bAddr0], w[bAddr0 + p.K], w[bAddr0 + 2u * p.K], w[bAddr0 + 3u * p.K]);
        } else {
          if (row0 < p.N) { v.x = w[bAddr0]; }
          if (row0 + 1u < p.N) { v.y = w[bAddr0 + p.K]; }
          if (row0 + 2u < p.N) { v.z = w[bAddr0 + 2u * p.K]; }
          if (row0 + 3u < p.N) { v.w = w[bAddr0 + 3u * p.K]; }
        }
      }
      tb[kk * 33u + r4] = v;
    }`;
  } else if (dtype === "F16") {
    loadBBlock = `
    if ((p.K & 1u) == 0u) {
      let kkB = (li % 8u) * 2u;
      let r4B = li / 8u;
      let rowB0 = n0 + r4B * 4u;
      let kHalf = p.K >> 1u;
      let bOff = rowB0 * kHalf + ((k0 + kkB) >> 1u);
      var lo = vec4<f32>(0.0);
      var hi = vec4<f32>(0.0);
      if (k0 + kkB < p.K) {
        let hasHi = k0 + kkB + 1u < p.K;
        if (rowB0 + 3u < p.N) {
          let p0 = unpack2x16float(w[bOff]);
          let p1 = unpack2x16float(w[bOff + kHalf]);
          let p2 = unpack2x16float(w[bOff + 2u * kHalf]);
          let p3 = unpack2x16float(w[bOff + 3u * kHalf]);
          lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
          if (hasHi) { hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y); }
        } else {
          if (rowB0 < p.N) { let p0 = unpack2x16float(w[bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
          if (rowB0 + 1u < p.N) { let p1 = unpack2x16float(w[bOff + kHalf]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
          if (rowB0 + 2u < p.N) { let p2 = unpack2x16float(w[bOff + 2u * kHalf]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
          if (rowB0 + 3u < p.N) { let p3 = unpack2x16float(w[bOff + 3u * kHalf]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }
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
        if (k < p.K) {
          let row0 = n0 + r4 * 4u;
          for (var u = 0u; u < 4u; u += 1u) {
            if (row0 + u < p.N) {
              let idx = (row0 + u) * p.K + k;
              v[u] = unpack2x16float(w[idx >> 1u])[idx & 1u];
            }
          }
        }
        tb[kk * 33u + r4] = v;
      }
    }`;
  } else {
    loadBBlock = `
    if ((p.K & 3u) == 0u) {
      if (li < 128u) {
        let kk4 = (li % 4u) * 4u;
        let r4B = li / 4u;
        let rowB0 = n0 + r4B * 4u;
        let kWord = p.K >> 2u;
        let wOff = rowB0 * kWord + ((k0 + kk4) >> 2u);
        var v0 = vec4<f32>(0.0);
        var v1 = vec4<f32>(0.0);
        var v2 = vec4<f32>(0.0);
        var v3 = vec4<f32>(0.0);
        if (k0 + kk4 < p.K) {
          for (var u = 0u; u < 4u; u += 1u) {
            if (rowB0 + u < p.N) {
              let s = sc[rowB0 + u];
              let packed = w[wOff + u * kWord];
              v0[u] = f32(i32(packed << 24u) >> 24) * s;
              if (k0 + kk4 + 1u < p.K) { v1[u] = f32(i32((packed >> 8u) << 24u) >> 24) * s; }
              if (k0 + kk4 + 2u < p.K) { v2[u] = f32(i32((packed >> 16u) << 24u) >> 24) * s; }
              if (k0 + kk4 + 3u < p.K) { v3[u] = f32(i32(packed) >> 24) * s; }
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
        if (k < p.K) {
          let row0 = n0 + r4 * 4u;
          for (var u = 0u; u < 4u; u += 1u) {
            let r = row0 + u;
            if (r < p.N) {
              let idx = r * p.K + k;
              let q = i32((w[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u);
              v[u] = f32(select(q, q - 256, q >= 128)) * sc[r];
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
      if (k0 + kkA < p.K) {
        let aOff = rowA0 * p.K + k0 + kkA;
        if (rowA0 + 3u < p.M) {
          v = vec4<f32>(a[aOff], a[aOff + p.K], a[aOff + 2u * p.K], a[aOff + 3u * p.K]);
        } else {
          if (rowA0 < p.M) { v.x = a[aOff]; }
          if (rowA0 + 1u < p.M) { v.y = a[aOff + p.K]; }
          if (rowA0 + 2u < p.M) { v.z = a[aOff + 2u * p.K]; }
          if (rowA0 + 3u < p.M) { v.w = a[aOff + 3u * p.K]; }
        }
      }
      ta[kkA * 17u + r4A] = v;
    }` : `
    if (li < 128u) {
      let kkA = li % 16u;
      let r4A = li / 16u;
      let rowA0 = m0 + r4A * 4u;
      var v = vec4<f32>(0.0);
      if (k0 + kkA < p.K) {
        let aOff = rowA0 * p.K + k0 + kkA;
        if (rowA0 + 3u < p.M) {
          v = vec4<f32>(a[aOff], a[aOff + p.K], a[aOff + 2u * p.K], a[aOff + 3u * p.K]);
        } else {
          if (rowA0 < p.M) { v.x = a[aOff]; }
          if (rowA0 + 1u < p.M) { v.y = a[aOff + p.K]; }
          if (rowA0 + 2u < p.M) { v.z = a[aOff + 2u * p.K]; }
          if (rowA0 + 3u < p.M) { v.w = a[aOff + 3u * p.K]; }
        }
      }
      ta[kkA * 9u + r4A] = v;
    }`;

  const rowIdxs = Array.from({ length: rowsPerThread }, (_, i) => i);
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

  return `${useNativeF16 ? "enable f16;\n" : ""}
struct P { M: u32, N: u32, K: u32, hasBias: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${wType}>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read> sc: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

var<workgroup> ta: array<vec4<f32>, ${taCount}>;
var<workgroup> tb: array<vec4<${EB}>, 528>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = b[0]; _ = sc[0];
  let tilesN = (p.N + 127u) / 128u;
  let flat = wid.x + wid.y * 65535u;
  let m0 = (flat / tilesN) * ${tileM}u;
  let n0 = (flat % tilesN) * 128u;
  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * ${rowsPerThread}u;
  let tn0 = tx * 4u;
  let tn1 = tn0 + 64u;

  var bVec0 = vec4<f32>(0.0);
  var bVec1 = vec4<f32>(0.0);
  ${hasBias ? `
  {
    let c0 = n0 + tn0;
    if (c0 < p.N) { bVec0.x = b[c0]; }
    if (c0 + 1u < p.N) { bVec0.y = b[c0 + 1u]; }
    if (c0 + 2u < p.N) { bVec0.z = b[c0 + 2u]; }
    if (c0 + 3u < p.N) { bVec0.w = b[c0 + 3u]; }
    let c1 = n0 + tn1;
    if (c1 < p.N) { bVec1.x = b[c1]; }
    if (c1 + 1u < p.N) { bVec1.y = b[c1 + 1u]; }
    if (c1 + 2u < p.N) { bVec1.z = b[c1 + 2u]; }
    if (c1 + 3u < p.N) { bVec1.w = b[c1 + 3u]; }
  }` : ""}
  ${rowIdxs.map((i) => `var acc_${i}_0 = bVec0; var acc_${i}_1 = bVec1;`).join("\n  ")}

  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    ${loadABlock}
    ${loadBBlock}
    workgroupBarrier();
    if (k0 + 16u <= p.K) {
      ${Array.from({ length: 16 }, (_, kk) => stepBody(`${kk}u`)).join("\n      ")}
    } else {
      let rem = p.K - k0;
      for (var kk = 0u; kk < rem; kk += 1u) {
        ${stepBody("kk")}
      }
    }
    workgroupBarrier();
  }

  ${rowIdxs.map((i) => `{
    let r = m0 + tm + ${i}u;
    if (r < p.M) {
      let r0 = acc_${i}_0;
      let r1 = acc_${i}_1;
      let rowY = r * p.N;
      ${[0, 1, 2, 3].map((j) => `if (n0 + tn0 + ${j}u < p.N) { y[rowY + n0 + tn0 + ${j}u] = r0.${"xyzw"[j]}; }`).join("\n      ")}
      ${[0, 1, 2, 3].map((j) => `if (n0 + tn1 + ${j}u < p.N) { y[rowY + n0 + tn1 + ${j}u] = r1.${"xyzw"[j]}; }`).join("\n      ")}
    }
  }`).join("\n  ")}
}`;
}

export function gemv(dtype, hasBias, native = false) {
  const useNativeF16 = dtype === "F16" && native;
  const wType = dtype === "F32" ? "f32" : useNativeF16 ? "f16" : "u32";
  const readW = dtype === "F32"
    ? "w[idx]"
    : useNativeF16
      ? "f32(w[idx])"
      : dtype === "F16"
        ? "unpack2x16float(w[idx >> 1u])[idx & 1u]"
        : "f32(select(i32((w[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u), i32((w[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u) - 256, ((w[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u) >= 128u)) * s";
  return `${useNativeF16 ? "enable f16;\n" : ""}
struct P { M: u32, N: u32, K: u32, hasBias: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${wType}>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read> sc: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = b[0]; _ = sc[0];
  let tid = (wid.x + wid.y * 65535u) * 64u + li;
  if (tid >= p.M * p.N) { return; }
  let r = tid / p.N;
  let c = tid % p.N;
  var acc = 0.0;
  ${hasBias ? "acc = b[c];" : ""}
  ${dtype === "I8" ? "let s = sc[c];" : ""}
  let aBase = r * p.K;
  let bBase = c * p.K;
  ${dtype === "I8" ? `
  if ((p.K & 3u) == 0u) {
    let wBase = bBase >> 2u;
    let words = p.K >> 2u;
    for (var u = 0u; u < words; u += 1u) {
      let packed = w[wBase + u];
      let kk = aBase + (u << 2u);
      acc = fma(a[kk], f32(i32(packed << 24u) >> 24) * s, acc);
      acc = fma(a[kk + 1u], f32(i32((packed >> 8u) << 24u) >> 24) * s, acc);
      acc = fma(a[kk + 2u], f32(i32((packed >> 16u) << 24u) >> 24) * s, acc);
      acc = fma(a[kk + 3u], f32(i32(packed) >> 24) * s, acc);
    }
    y[tid] = acc;
    return;
  }` : dtype === "F16" && !useNativeF16 ? `
  if ((p.K & 1u) == 0u) {
    let wBase = bBase >> 1u;
    let words = p.K >> 1u;
    for (var u = 0u; u < words; u += 1u) {
      let pair = unpack2x16float(w[wBase + u]);
      let kk = aBase + (u << 1u);
      acc = fma(a[kk], pair.x, acc);
      acc = fma(a[kk + 1u], pair.y, acc);
    }
    y[tid] = acc;
    return;
  }` : ""}
  for (var k = 0u; k < p.K; k += 1u) {
    let idx = bBase + k;
    acc = fma(a[aBase + k], ${readW}, acc);
  }
  y[tid] = acc;
}`;
}

/**
 * Parallelized 4-lane binary-carry cascade Welford LayerNorm matching SiD-Protein's ARM64 PyTorch oracle.
 * mode = 0: standard LayerNorm
 * mode = 1: fused adaptive LayerNorm ((x - mu)*inv * sigmoid(w[condRow*D + j]) + b[condRow*D + j]), condMode = p.condMode
 * mode = 2: stats-only output [rows, 2] = (mu, inv)
 */
export const norm = `${numericWGSL}${divisionWGSL}
struct P {
  D: u32, stride: u32, offset: u32, hasW: u32,
  rows: u32, mode: u32, condMode: u32, pad0: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

var<workgroup> chunkMeans: array<f32, 128>;
var<workgroup> chunkVars: array<f32, 128>;
var<workgroup> means: array<f32, 4>;
var<workgroup> variances: array<f32, 4>;
var<workgroup> mu: f32;
var<workgroup> inv: f32;

fn divide(x: f32, z: f32) -> f32 {
  if (z == 0.0) { return 0.0; }
  return rfdiv(x, z);
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) gid: vec3<u32>) {
  _ = w[0]; _ = b[0];
  let rawRow = gid.x + gid.y * 65535u;
  let row = min(rawRow, max(p.rows, 1u) - 1u);
  let D = p.D;
  let n = D / 4u;
  let chunks = (n + 15u) / 16u;
  let slots = chunks * 4u;

  var RECIP = array<f32, 16>(
    1.0, 0.5, 0.3333333432674408, 0.25,
    0.20000000298023224, 0.1666666716337204, 0.1428571492433548, 0.125,
    0.1111111119389534, 0.10000000149011612, 0.09090909361839294, 0.0833333358168602,
    0.07692307978868484, 0.0714285746216774, 0.06666667014360428, 0.0625
  );

  if (tid < slots) {
    let chunk = tid / 4u;
    let lane = tid % 4u;
    let size = min(16u, n - chunk * 16u);
    var m = 0.0;
    var v = 0.0;
    let base = row * p.stride + p.offset + chunk * 64u + lane;
    for (var j = 0u; j < size; j += 1u) {
      let x = a[base + j * 4u];
      let delta = x - m;
      m = fma(delta, RECIP[j], m);
      v = fma(delta, x - m, v);
    }
    chunkMeans[tid] = m;
    chunkVars[tid] = v;
  }
  workgroupBarrier();

  if (tid < 4u) {
    let lane = tid;
    var count: array<u32, 8>;
    var ms: array<f32, 8>;
    var vs: array<f32, 8>;
    var depth = 0u;
    var z = chunks - 1u;
    while (z > 0u) { depth += 1u; z = z >> 1u; }
    for (var chunk = 0u; chunk < chunks; chunk += 1u) {
      let size = min(16u, n - chunk * 16u);
      let m = chunkMeans[chunk * 4u + lane];
      let v = chunkVars[chunk * 4u + lane];
      var total = count[0] + size;
      var c = divide(f32(size), f32(total));
      var delta = m - ms[0];
      ms[0] = fma(c, delta, ms[0]);
      vs[0] = vs[0] + fma((delta * delta) * c, f32(count[0]), v);
      count[0] = total;
      var mask = chunk + 1u;
      var level = 1u;
      while (level < depth && (mask & 1u) == 0u) {
        total = count[level] + count[level - 1u];
        c = divide(f32(count[level - 1u]), f32(total));
        delta = ms[level - 1u] - ms[level];
        ms[level] = fma(c, delta, ms[level]);
        vs[level] = vs[level] + fma((delta * delta) * c, f32(count[level]), vs[level - 1u]);
        count[level] = total;
        count[level - 1u] = 0u;
        ms[level - 1u] = 0.0;
        vs[level - 1u] = 0.0;
        mask = mask >> 1u;
        level += 1u;
      }
    }
    for (var level = 1u; level < depth; level += 1u) {
      let total = count[0] + count[level];
      let c = divide(f32(count[level]), f32(total));
      let delta = ms[level] - ms[0];
      ms[0] = fma(c, delta, ms[0]);
      vs[0] = vs[0] + fma((delta * delta) * c, f32(count[0]), vs[level]);
      count[0] = total;
    }
    means[lane] = ms[0];
    variances[lane] = vs[0];
  }
  workgroupBarrier();

  if (tid == 0u) {
    var count = 0u;
    var m = 0.0;
    var v = 0.0;
    for (var k = 0u; k < 4u; k += 1u) {
      let next = count + D / 4u;
      let c = divide(f32(D / 4u), f32(next));
      let delta = means[k] - m;
      m = fma(c, delta, m);
      v = v + fma((delta * delta) * c, f32(count), variances[k]);
      count = next;
    }
    mu = m;
    let variance = divide(v, f32(D)) + 0.00001;
    var root = sqrt(variance);
    root = root + fma(-root, root, variance) / (2.0 * root);
    inv = divide(1.0, root);
  }
  workgroupBarrier();

  if (rawRow >= p.rows) { return; }
  if (p.mode == 2u) {
    if (tid == 0u) {
      y[row * 2u] = mu;
      y[row * 2u + 1u] = inv;
    }
    return;
  }
  let condRow = select(0u, select(0u, 1u, row >= 10u), p.condMode != 0u);
  for (var j = tid; j < D; j += 128u) {
    var z = (a[row * p.stride + p.offset + j] - mu) * inv;
    if (p.mode == 1u) {
      let cj = condRow * D + j;
      let g = rfdiv(1.0, 1.0 + rfExp(-w[cj]));
      z = z * g + b[cj];
    } else if (p.hasW != 0u) {
      z = fma(z, w[j], b[j]);
    }
    y[row * D + j] = z;
  }
}`;

export const pairBins = `
struct P { L: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> bounds: array<f32>;
@group(0) @binding(3) var<storage, read_write> outIdx: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 64u + li;
  let total = p.L * p.L;
  if (idx >= total) { return; }
  let row = idx / p.L;
  let col = idx % p.L;
  let sep = u32(clamp(i32(row) - i32(col) + 63, 0, 126));
  var bucket = 0u;
  let dx = x[row * 3u] - x[col * 3u];
  let dy = x[row * 3u + 1u] - x[col * 3u + 1u];
  let dz = x[row * 3u + 2u] - x[col * 3u + 2u];
  let dist = sqrt((dx * dx + dy * dy) + dz * dz);
  for (var j = 0u; j < 63u; j += 1u) {
    if (dist > bounds[j]) { bucket += 1u; }
  }
  outIdx[idx * 2u] = sep;
  outIdx[idx * 2u + 1u] = 255u + bucket;
}`;

export function sparsePairLinear(dtype, hasBias, native = false) {
  const useNativeF16 = dtype === "F16" && native;
  const wType = dtype === "F32" ? "f32" : useNativeF16 ? "f16" : "u32";
  const fetchW = dtype === "F32"
    ? "w[idx]"
    : useNativeF16
      ? "f32(w[idx])"
      : dtype === "F16"
        ? "unpack2x16float(w[idx >> 1u])[idx & 1u]"
        : "f32(select(i32((w[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u), i32((w[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u) - 256, ((w[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u) >= 128u)) * s";
  return `${useNativeF16 ? "enable f16;\n" : ""}
struct P { pairs: u32, N: u32, K: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<${wType}>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read> sc: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = b[0]; _ = sc[0];
  let tid = (wid.x + wid.y * 65535u) * 256u + li;
  if (tid >= p.pairs * p.N) { return; }
  let pair = tid / p.N;
  let col = tid % p.N;
  ${dtype === "I8" ? "let s = sc[col];" : ""}
  var acc = 0.0;
  ${hasBias ? "acc = b[col];" : ""}
  let rowW = col * p.K;
  let i0 = indices[pair * 2u];
  let i1 = indices[pair * 2u + 1u];
  { let idx = rowW + i0; acc = fma(1.0, ${fetchW}, acc); }
  { let idx = rowW + i1; acc = fma(1.0, ${fetchW}, acc); }
  y[tid] = acc;
}`;
}

/**
 * Fused 16x16 tiled pair_norm (from precomputed pairStats [N*N, 2]) + to_bias [N*N, 12].
 * Eliminates 14 redundant 25 MB Welford reductions and 15 intermediate 25 MB pair_norm writes/reads!
 */
export function pairBias(dtype, hasBias, native = false) {
  const useNativeF16 = dtype === "F16" && native;
  const wType = dtype === "F32" ? "f32" : useNativeF16 ? "f16" : "u32";
  const fetchW = dtype === "F32"
    ? "bw[idx]"
    : useNativeF16
      ? "f32(bw[idx])"
      : dtype === "F16"
        ? "unpack2x16float(bw[idx >> 1u])[idx & 1u]"
        : "f32(select(i32((bw[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u), i32((bw[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u) - 256, ((bw[idx >> 2u] >> ((idx & 3u) * 8u)) & 255u) >= 128u)) * sc[col]";
  return `${useNativeF16 ? "enable f16;\n" : ""}
struct P { pairs: u32, K: u32, heads: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pair: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> nw: array<f32>;
@group(0) @binding(4) var<storage, read> nb: array<f32>;
@group(0) @binding(5) var<storage, read> bw: array<${wType}>;
@group(0) @binding(6) var<storage, read> bb: array<f32>;
@group(0) @binding(7) var<storage, read> sc: array<f32>;
@group(0) @binding(8) var<storage, read_write> outBias: array<f32>;

var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = bb[0]; _ = sc[0];
  let tilePair = wid.x + wid.y * 65535u;
  let rLocal = li / 16u;
  let cLocal = li % 16u;
  let row = tilePair * 16u + rLocal;
  let col = cLocal;

  var m = 0.0;
  var iv = 0.0;
  if (row < p.pairs) {
    m = stats[row * 2u];
    iv = stats[row * 2u + 1u];
  }
  var acc = 0.0;
  ${hasBias ? "if (col < p.heads) { acc = bb[col]; }" : ""}

  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    var aVal = 0.0;
    let kkA = k0 + cLocal;
    if (row < p.pairs && kkA < p.K) {
      let normed = (pair[row * p.K + kkA] - m) * iv;
      aVal = fma(normed, nw[kkA], nb[kkA]);
    }
    ta[rLocal * 16u + cLocal] = aVal;

    var bVal = 0.0;
    let kkB = k0 + rLocal;
    if (col < p.heads && kkB < p.K) {
      let idx = col * p.K + kkB;
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
    outBias[row * p.heads + col] = acc;
  }
}`;
}

/**
 * Tiled 32x32 Attention Scores (H=12, D=64) + [N*N, 12] pair bias.
 * Sequential d=0..63 accumulation preserves exact non-FMA/FMA rounding of original scores kernel!
 */
export const scores = `
struct P { N: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

var<workgroup> ta: array<f32, 512>;
var<workgroup> tb: array<f32, 512>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesN = (p.N + 31u) / 32u;
  let flat = wid.x + wid.y * 65535u;
  let tilesPerHead = tilesN * tilesN;
  let h = flat / tilesPerHead;
  let inHead = h < 12u;
  let head = min(h, 11u);
  let rem = flat % tilesPerHead;
  let i0 = (rem / tilesN) * 32u;
  let j0 = (rem % tilesN) * 32u;
  let ty = li / 16u;
  let tx = li % 16u;

  var acc00 = 0.0; var acc01 = 0.0;
  var acc10 = 0.0; var acc11 = 0.0;

  for (var d0 = 0u; d0 < 64u; d0 += 16u) {
    for (var e = 0u; e < 2u; e += 1u) {
      let slot = li + e * 256u;
      let r = slot / 16u;
      let d = slot % 16u;
      var qv = 0.0;
      var kv = 0.0;
      if (inHead) {
        if (i0 + r < p.N) { qv = q[(i0 + r) * 768u + head * 64u + d0 + d]; }
        if (j0 + r < p.N) { kv = k[(j0 + r) * 768u + head * 64u + d0 + d]; }
      }
      ta[r * 16u + d] = qv;
      tb[r * 16u + d] = kv;
    }
    workgroupBarrier();
    let r0 = ty * 2u;
    let c0 = tx * 2u;
    for (var d = 0u; d < 16u; d += 1u) {
      let qa0 = ta[r0 * 16u + d];
      let qa1 = ta[(r0 + 1u) * 16u + d];
      let kb0 = tb[c0 * 16u + d];
      let kb1 = tb[(c0 + 1u) * 16u + d];
      acc00 = acc00 + qa0 * kb0;
      acc01 = acc01 + qa0 * kb1;
      acc10 = acc10 + qa1 * kb0;
      acc11 = acc11 + qa1 * kb1;
    }
    workgroupBarrier();
  }

  if (!inHead) { return; }
  let r0 = i0 + ty * 2u;
  let c0 = j0 + tx * 2u;
  let hOff = head * p.N * p.N;
  if (r0 < p.N && c0 < p.N) { y[hOff + r0 * p.N + c0] = acc00 * 0.125 + b[(r0 * p.N + c0) * 12u + head]; }
  if (r0 < p.N && c0 + 1u < p.N) { y[hOff + r0 * p.N + c0 + 1u] = acc01 * 0.125 + b[(r0 * p.N + c0 + 1u) * 12u + head]; }
  if (r0 + 1u < p.N && c0 < p.N) { y[hOff + (r0 + 1u) * p.N + c0] = acc10 * 0.125 + b[((r0 + 1u) * p.N + c0) * 12u + head]; }
  if (r0 + 1u < p.N && c0 + 1u < p.N) { y[hOff + (r0 + 1u) * p.N + c0 + 1u] = acc11 * 0.125 + b[((r0 + 1u) * p.N + c0 + 1u) * 12u + head]; }
}`;

export const softmax = `${numericWGSL}${divisionWGSL}
struct P { N: u32, rows: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

var<workgroup> tmp: array<f32, 64>;
var<workgroup> rowExp: array<f32, 512>;
var<workgroup> maxVal: f32;
var<workgroup> inverse: f32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let rawRow = g.x + g.y * 65535u;
  let row = min(rawRow, max(p.rows, 1u) - 1u);
  let N = p.N;
  let off = row * N;
  var m = -3.402823e38;
  for (var k = lane; k < N; k += 64u) { m = max(m, x[off + k]); }
  tmp[lane] = m;
  workgroupBarrier();
  for (var n = 32u; n > 0u; n >>= 1u) {
    if (lane < n) { tmp[lane] = max(tmp[lane], tmp[lane + n]); }
    workgroupBarrier();
  }
  if (lane == 0u) { maxVal = tmp[0]; }
  workgroupBarrier();
  m = maxVal;

  if (N <= 512u) {
    for (var k = lane; k < N; k += 64u) { rowExp[k] = rfExp(x[off + k] - m); }
    workgroupBarrier();
    if (lane < 4u) {
      var s = 0.0;
      for (var k = lane; k < N; k += 4u) { s += rowExp[k]; }
      tmp[lane] = s;
    }
    workgroupBarrier();
    if (lane == 0u) {
      let den = (tmp[0] + tmp[2]) + (tmp[1] + tmp[3]);
      inverse = rfdiv(1.0, den);
    }
    workgroupBarrier();
    if (rawRow < p.rows) {
      let inv = inverse;
      for (var k = lane; k < N; k += 64u) { y[off + k] = rowExp[k] * inv; }
    }
  } else {
    if (lane < 4u) {
      var s = 0.0;
      for (var k = lane; k < N; k += 4u) { s += rfExp(x[off + k] - m); }
      tmp[lane] = s;
    }
    workgroupBarrier();
    if (lane == 0u) {
      let den = (tmp[0] + tmp[2]) + (tmp[1] + tmp[3]);
      inverse = rfdiv(1.0, den);
    }
    workgroupBarrier();
    if (rawRow < p.rows) {
      let inv = inverse;
      for (var k = lane; k < N; k += 64u) { y[off + k] = rfExp(x[off + k] - m) * inv; }
    }
  }
}`;

export const context = `${numericWGSL}${divisionWGSL}
struct P { N: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> qkv: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

var<workgroup> ta: array<f32, 512>;
var<workgroup> tb: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesI = (p.N + 31u) / 32u;
  let flat = wid.x + wid.y * 65535u;
  let h = flat / tilesI;
  let inHead = h < 12u;
  let head = min(h, 11u);
  let i0 = (flat % tilesI) * 32u;

  let ty = li / 16u;
  let tx = li % 16u;
  let row0 = i0 + ty * 2u;
  let col0 = tx * 4u;

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);

  for (var j0 = 0u; j0 < p.N; j0 += 16u) {
    for (var e = 0u; e < 2u; e += 1u) {
      let slot = li + e * 256u;
      let r = slot / 16u;
      let kj = slot % 16u;
      var pv = 0.0;
      if (inHead && i0 + r < p.N && j0 + kj < p.N) {
        pv = a[(head * p.N + i0 + r) * p.N + j0 + kj];
      }
      ta[r * 16u + kj] = pv;
    }
    for (var e = 0u; e < 4u; e += 1u) {
      let slot = li + e * 256u;
      let kj = slot / 64u;
      let d = slot % 64u;
      var vv = 0.0;
      if (inHead && j0 + kj < p.N) {
        vv = qkv[(j0 + kj) * 2304u + 1536u + head * 64u + d];
      }
      tb[kj * 64u + d] = vv;
    }
    workgroupBarrier();
    let rem = min(16u, p.N - j0);
    for (var kj = 0u; kj < rem; kj += 1u) {
      let p0 = ta[(ty * 2u) * 16u + kj];
      let p1 = ta[(ty * 2u + 1u) * 16u + kj];
      let vb = kj * 64u + col0;
      let vVec = vec4<f32>(tb[vb], tb[vb + 1u], tb[vb + 2u], tb[vb + 3u]);
      acc0 = acc0 + vec4<f32>(p0) * vVec;
      acc1 = acc1 + vec4<f32>(p1) * vVec;
    }
    workgroupBarrier();
  }

  if (!inHead) { return; }
  for (var r = 0u; r < 2u; r += 1u) {
    let row = row0 + r;
    if (row < p.N) {
      let vals = select(acc0, acc1, r == 1u);
      for (var u = 0u; u < 4u; u += 1u) {
        let idx = row * 768u + head * 64u + col0 + u;
        y[idx] = vals[u] * rfdiv(1.0, 1.0 + rfExp(-gate[idx]));
      }
    }
  }
}`;

export function element(body) {
  return `
struct P { n: u32, D: u32, ex0: u32, ex1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${numericWGSL}${divisionWGSL}
fn sigmoid(x: f32) -> f32 { return rfdiv(1.0, 1.0 + rfExp(-x)); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = a[0]; _ = b[0]; _ = c[0];
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.n) { return; }
  let D = p.D;
  ${body}
}`;
}

export const elements = {
  add: element("y[i] = a[i] + b[i];"),
  swiglu: element(
    "let j = (i / D) * D * 2u + i % D; let v = a[j + D]; y[i] = rfdiv(v, 1.0 + rfExp(-v)) * a[j];",
  ),
  residualGateAdd: element(
    "let j = select(0u, D, i / D >= 10u) + i % D; y[i] = a[i] + b[i] * sigmoid(c[j]);",
  ),
  padCondition: element("y[i] = 0.0; if (i >= 512u) { y[i] = a[i - 512u]; }"),
  padPair: element(
    "let N = p.ex0; let row = (i / D) / N; let col = (i / D) % N; y[i] = 0.0; if (row >= 10u && col >= 10u) { y[i] = a[((row - 10u) * (N - 10u) + col - 10u) * D + i % D]; }",
  ),
  unpad: element("y[i] = a[i + p.ex0 * D];"),
  velocityAdd: element("let t = bitcast<f32>(p.ex0); y[i] = a[i] + (1.0 - t) * b[30u + i];"),
  time: element(
    "let k = i % 128u; let v = bitcast<f32>(p.ex0) * 2000.0 * a[k]; y[i] = select(rfSin(v), rfCos(v), i >= 128u);",
  ),
  position: element(
    "let row = i / 132u; let k = i % 132u; y[i] = 0.0; if (k < 128u) { let numerator = f32(row + 1u) * 3.1415927410125732; let z = rfdiv(numerator, a[k % 64u]); let large = rfdiv(numerator, a[((k % 64u) / 4u) * 4u]) >= 125.0; let sc = rfSinCos(z, large); y[i] = select(sc.x, sc.y, k >= 64u); }",
  ),
  center: element("y[i] = a[i] - b[i % 3u];"),
  scale: element("y[i] = a[i] * bitcast<f32>(p.ex0);"),
};

export function gather(dtype, native = false) {
  const useNativeF16 = dtype === "F16" && native;
  const wType = dtype === "F32" ? "f32" : useNativeF16 ? "f16" : "u32";
  const weightExpr = dtype === "F32"
    ? "w[wIdx]"
    : useNativeF16
      ? "f32(w[wIdx])"
      : dtype === "F16"
        ? "unpack2x16float(w[wIdx >> 1u])[wIdx & 1u]"
        : "f32(select(i32((w[wIdx >> 2u] >> ((wIdx & 3u) * 8u)) & 255u), i32((w[wIdx >> 2u] >> ((wIdx & 3u) * 8u)) & 255u) - 256, ((w[wIdx >> 2u] >> ((wIdx & 3u) * 8u)) & 255u) >= 128u)) * sc[wRow]";
  return `${useNativeF16 ? "enable f16;\n" : ""}
struct P { n: u32, mode: u32, rowId: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${wType}>;
@group(0) @binding(3) var<storage, read> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = a[0]; _ = sc[0];
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.n) { return; }
  if (p.mode == 0u) {
    let wRow = p.rowId;
    let wIdx = wRow * 256u + i;
    y[i] = ${weightExpr};
  } else {
    if (i < 7680u) {
      let wRow = i / 768u;
      let wIdx = i;
      y[i] = ${weightExpr};
    } else {
      y[i] = a[i - 7680u];
    }
  }
}`;
}

export const coordinateMean = `${divisionWGSL}
struct P { L: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = g.x;
  if (d >= 3u) { return; }
  let L = p.L;
  let size = L / 4u;
  var levelPower = 4u;
  var lg = 0u;
  var q = max(size, 1u) - 1u;
  while (q > 0u) { lg += 1u; q = q >> 1u; }
  levelPower = max(levelPower, lg / 4u);
  let step = 1u << levelPower;
  let mask = step - 1u;
  var acc: array<f32, 16>;
  var i = 0u;
  while (i + step <= size) {
    for (var j = 0u; j < step; j += 1u) {
      for (var k = 0u; k < 4u; k += 1u) {
        acc[k] += x[((i + j) * 4u + k) * 3u + d];
      }
    }
    i += step;
    for (var level = 1u; level < 4u; level += 1u) {
      for (var k = 0u; k < 4u; k += 1u) {
        acc[level * 4u + k] += acc[(level - 1u) * 4u + k];
        acc[(level - 1u) * 4u + k] = 0.0;
      }
      if ((i & (mask << (level * levelPower))) != 0u) { break; }
    }
  }
  while (i < size) {
    for (var k = 0u; k < 4u; k += 1u) { acc[k] += x[(i * 4u + k) * 3u + d]; }
    i += 1u;
  }
  for (var level = 1u; level < 4u; level += 1u) {
    for (var k = 0u; k < 4u; k += 1u) { acc[k] += acc[level * 4u + k]; }
  }
  for (var j = size * 4u; j < L; j += 1u) { acc[0] += x[j * 3u + d]; }
  var sum = acc[0];
  for (var k = 1u; k < 4u; k += 1u) { sum += acc[k]; }
  y[d] = rfdiv(sum, f32(L));
}`;
