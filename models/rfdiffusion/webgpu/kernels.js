// Reusable Naga/Dawn-compatible tiled WGSL compute kernels for RFdiffusion GPUAlgebra.
// Strict WebGPU floor compliance: workgroup_size <= 256, workgroupStorageSize <= 16384 B,
// 2D-folded dispatch < 65535, uniform control flow across all barriers, zero dynamic const/value-array indexing.

import { numericWGSL } from "../shared/numeric.js";

export const MAX_GRID_X = 65535;

export function foldGrid2D(groups) {
  if (groups <= MAX_GRID_X) return [Math.max(1, groups), 1];
  return [MAX_GRID_X, Math.ceil(groups / MAX_GRID_X)];
}

function weightLoaderDecl(dtype, useF16) {
  const stageF16 = dtype === "F16" && useF16;
  const header = `${stageF16 ? "enable f16;\n" : ""}${numericWGSL}
struct P {
  m: u32, n: u32, k: u32, batches: u32,
  sA0: u32, sB0: u32, sC0: u32, scaleOff: u32,
  hasBias: u32, act: u32, alphaBits: u32, accMode: u32,
  pad0: u32, pad1: u32, pad2: u32, pad3: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<${dtype === "F32" ? "f32" : "u32"}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;
`;
  let readB;
  if (dtype === "F32") {
    readB = `fn loadB(baseB: u32, row: u32, col: u32) -> f32 {
  return b[baseB + row * p.k + col];
}`;
  } else if (dtype === "F16") {
    readB = `fn loadB(baseB: u32, row: u32, col: u32) -> ${stageF16 ? "f16" : "f32"} {
  let idx = baseB + row * p.k + col;
  let pair = unpack2x16float(b[idx >> 1u]);
  let val = select(pair.x, pair.y, (idx & 1u) != 0u);
  return ${stageF16 ? "f16(val)" : "val"};
}`;
  } else {
    readB = `fn loadB(baseB: u32, row: u32, col: u32) -> f32 {
  let idx = baseB + row * p.k + col;
  let shift = (idx & 3u) * 8u;
  let q = (i32(b[idx >> 2u] >> shift) << 24u) >> 24u;
  return f32(q) * bitcast<f32>(b[p.scaleOff + row]);
}`;
  }
  const epilogueFn = `fn writeOut(at: u32, val: f32) {
  let alpha = bitcast<f32>(p.alphaBits);
  var v = select(val, val * alpha, p.alphaBits != 0x3f800000u);
  if (p.act == 1u) { v = max(v, 0.0); }
  else if (p.act == 2u) { v = 1.0 / (1.0 + rfExp(-v)); }
  if (p.accMode == 1u) { v = v + c[at]; }
  c[at] = v;
}`;
  return { header, readB, epilogueFn, EB: stageF16 ? "f16" : "f32" };
}

export function gemm64x128Source({ dtype = "F32", useF16 = false } = {}) {
  const { header, readB, epilogueFn, EB } = weightLoaderDecl(dtype, useF16);
  const rows = [0, 1, 2, 3];
  const unrolled = Array.from(
    { length: 16 },
    (_, kk) => `{
        let av = ta[${kk * 17}u + tm4];
        let bv0 = vec4<f32>(tb[${kk * 33}u + tx]);
        let bv1 = vec4<f32>(tb[${kk * 33 + 16}u + tx]);
        acc_0_0 = fma(vec4<f32>(av.x), bv0, acc_0_0); acc_0_1 = fma(vec4<f32>(av.x), bv1, acc_0_1);
        acc_1_0 = fma(vec4<f32>(av.y), bv0, acc_1_0); acc_1_1 = fma(vec4<f32>(av.y), bv1, acc_1_1);
        acc_2_0 = fma(vec4<f32>(av.z), bv0, acc_2_0); acc_2_1 = fma(vec4<f32>(av.z), bv1, acc_2_1);
        acc_3_0 = fma(vec4<f32>(av.w), bv0, acc_3_0); acc_3_1 = fma(vec4<f32>(av.w), bv1, acc_3_1);
      }`,
  ).join("\n      ");

  return `${header}
${readB}
${epilogueFn}
var<workgroup> ta: array<vec4<f32>, 272>;
var<workgroup> tb: array<vec4<${EB}>, 528>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesN = (p.n + 127u) / 128u;
  let tilesM = (p.m + 63u) / 64u;
  let tilesPerBatch = tilesN * tilesM;
  let flat = wid.x + wid.y * 65535u;
  let batch = flat / tilesPerBatch;
  let inBatch = batch < p.batches;
  let bat = min(batch, p.batches - 1u);
  let tile = flat % tilesPerBatch;
  let m0 = (tile / tilesN) * 64u;
  let n0 = (tile % tilesN) * 128u;
  let baseA = bat * p.sA0;
  let baseB = bat * p.sB0;

  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * 4u;
  let tn0 = tx * 4u;
  let tn1 = tn0 + 64u;

  var init0 = vec4<f32>(0.0);
  var init1 = vec4<f32>(0.0);
  if (p.hasBias != 0u) {
    let c0 = n0 + tn0;
    let c1 = n0 + tn1;
    if (c0 < p.n) { init0.x = bias[c0]; }
    if (c0 + 1u < p.n) { init0.y = bias[c0 + 1u]; }
    if (c0 + 2u < p.n) { init0.z = bias[c0 + 2u]; }
    if (c0 + 3u < p.n) { init0.w = bias[c0 + 3u]; }
    if (c1 < p.n) { init1.x = bias[c1]; }
    if (c1 + 1u < p.n) { init1.y = bias[c1 + 1u]; }
    if (c1 + 2u < p.n) { init1.z = bias[c1 + 2u]; }
    if (c1 + 3u < p.n) { init1.w = bias[c1 + 3u]; }
  }
  var acc_0_0 = init0; var acc_0_1 = init1;
  var acc_1_0 = init0; var acc_1_1 = init1;
  var acc_2_0 = init0; var acc_2_1 = init1;
  var acc_3_0 = init0; var acc_3_1 = init1;

  let kkA = li % 16u;
  let r4A = li / 16u;
  let rowA0 = m0 + r4A * 4u;
  let fullA = rowA0 + 3u < p.m;
  let taWrite = kkA * 17u + r4A;

  for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {
    {
      var va = vec4<f32>(0.0);
      let k = k0 + kkA;
      if (inBatch && k < p.k) {
        let aOff = baseA + rowA0 * p.k + k;
        if (fullA) {
          va = vec4<f32>(a[aOff], a[aOff + p.k], a[aOff + 2u * p.k], a[aOff + 3u * p.k]);
        } else {
          if (rowA0 < p.m) { va.x = a[aOff]; }
          if (rowA0 + 1u < p.m) { va.y = a[aOff + p.k]; }
          if (rowA0 + 2u < p.m) { va.z = a[aOff + 2u * p.k]; }
          if (rowA0 + 3u < p.m) { va.w = a[aOff + 3u * p.k]; }
        }
      }
      ta[taWrite] = va;
    }
    for (var e = 0u; e < 2u; e = e + 1u) {
      let slot = li + e * 256u;
      let kk = slot % 16u;
      let r4 = slot / 16u;
      let k = k0 + kk;
      var vb = vec4<${EB}>(${EB}(0.0));
      if (inBatch && k < p.k) {
        let row0 = n0 + r4 * 4u;
        if (row0 + 3u < p.n) {
          vb = vec4<${EB}>(
            loadB(baseB, row0, k),
            loadB(baseB, row0 + 1u, k),
            loadB(baseB, row0 + 2u, k),
            loadB(baseB, row0 + 3u, k)
          );
        } else {
          if (row0 < p.n) { vb.x = loadB(baseB, row0, k); }
          if (row0 + 1u < p.n) { vb.y = loadB(baseB, row0 + 1u, k); }
          if (row0 + 2u < p.n) { vb.z = loadB(baseB, row0 + 2u, k); }
          if (row0 + 3u < p.n) { vb.w = loadB(baseB, row0 + 3u, k); }
        }
      }
      tb[kk * 33u + r4] = vb;
    }
    workgroupBarrier();
    if (k0 + 16u <= p.k) {
      ${unrolled}
    } else {
      let kRem = p.k - k0;
      for (var kk = 0u; kk < kRem; kk = kk + 1u) {
        let av = ta[kk * 17u + tm4];
        let bv0 = vec4<f32>(tb[kk * 33u + tx]);
        let bv1 = vec4<f32>(tb[kk * 33u + 16u + tx]);
        acc_0_0 = fma(vec4<f32>(av.x), bv0, acc_0_0); acc_0_1 = fma(vec4<f32>(av.x), bv1, acc_0_1);
        acc_1_0 = fma(vec4<f32>(av.y), bv0, acc_1_0); acc_1_1 = fma(vec4<f32>(av.y), bv1, acc_1_1);
        acc_2_0 = fma(vec4<f32>(av.z), bv0, acc_2_0); acc_2_1 = fma(vec4<f32>(av.z), bv1, acc_2_1);
        acc_3_0 = fma(vec4<f32>(av.w), bv0, acc_3_0); acc_3_1 = fma(vec4<f32>(av.w), bv1, acc_3_1);
      }
    }
    workgroupBarrier();
  }

  if (!inBatch) { return; }
  let baseC = bat * p.sC0;
  ${rows
    .map(
      (i) => `{
    let m = m0 + tm + ${i}u;
    if (m < p.m) {
      let rowC = baseC + m * p.n;
      let r0 = acc_${i}_0;
      let r1 = acc_${i}_1;
      ${[0, 1, 2, 3, 4, 5, 6, 7]
        .map((j) => {
          const nExpr = j < 4 ? `n0 + tn0 + ${j}u` : `n0 + tn1 + ${j - 4}u`;
          const valExpr = j < 4 ? `r0.${"xyzw"[j]}` : `r1.${"xyzw"[j - 4]}`;
          return `{ let n = ${nExpr}; if (n < p.n) { writeOut(rowC + n, ${valExpr}); } }`;
        })
        .join("\n      ")}
    }
  }`,
    )
    .join("\n  ")}
}
`;
}

export function gemm16x64Source({ dtype = "F32", useF16 = false } = {}) {
  const { header, readB, epilogueFn, EB } = weightLoaderDecl(dtype, useF16);
  const unrolled = Array.from(
    { length: 32 },
    (_, kk) => `{
        let av = ta[ty * 33u + ${kk}u];
        let bv = vec4<f32>(tb[${kk * 17}u + tx]);
        acc = fma(vec4<f32>(av), bv, acc);
      }`,
  ).join("\n      ");

  return `${header}
${readB}
${epilogueFn}
var<workgroup> ta: array<f32, 528>;
var<workgroup> tb: array<vec4<${EB}>, 544>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesN = (p.n + 63u) / 64u;
  let tilesM = (p.m + 15u) / 16u;
  let tilesPerBatch = tilesN * tilesM;
  let flat = wid.x + wid.y * 65535u;
  let batch = flat / tilesPerBatch;
  let inBatch = batch < p.batches;
  let bat = min(batch, p.batches - 1u);
  let tile = flat % tilesPerBatch;
  let m0 = (tile / tilesN) * 16u;
  let n0 = (tile % tilesN) * 64u;
  let baseA = bat * p.sA0;
  let baseB = bat * p.sB0;

  let ty = li / 16u;
  let tx = li % 16u;
  let col0 = n0 + tx * 4u;

  var acc = vec4<f32>(0.0);
  if (p.hasBias != 0u) {
    if (col0 < p.n) { acc.x = bias[col0]; }
    if (col0 + 1u < p.n) { acc.y = bias[col0 + 1u]; }
    if (col0 + 2u < p.n) { acc.z = bias[col0 + 2u]; }
    if (col0 + 3u < p.n) { acc.w = bias[col0 + 3u]; }
  }

  for (var k0 = 0u; k0 < p.k; k0 = k0 + 32u) {
    for (var e = 0u; e < 2u; e = e + 1u) {
      let slot = li + e * 256u;
      let r = slot / 32u;
      let kk = slot % 32u;
      var va = 0.0;
      if (inBatch && m0 + r < p.m && k0 + kk < p.k) {
        va = a[baseA + (m0 + r) * p.k + k0 + kk];
      }
      ta[r * 33u + kk] = va;
    }
    for (var e = 0u; e < 2u; e = e + 1u) {
      let slot = li + e * 256u;
      let kk = slot / 16u;
      let c4 = slot % 16u;
      let k = k0 + kk;
      var vb = vec4<${EB}>(${EB}(0.0));
      if (inBatch && k < p.k) {
        let rB = n0 + c4 * 4u;
        if (rB + 3u < p.n) {
          vb = vec4<${EB}>(
            loadB(baseB, rB, k),
            loadB(baseB, rB + 1u, k),
            loadB(baseB, rB + 2u, k),
            loadB(baseB, rB + 3u, k)
          );
        } else {
          if (rB < p.n) { vb.x = loadB(baseB, rB, k); }
          if (rB + 1u < p.n) { vb.y = loadB(baseB, rB + 1u, k); }
          if (rB + 2u < p.n) { vb.z = loadB(baseB, rB + 2u, k); }
          if (rB + 3u < p.n) { vb.w = loadB(baseB, rB + 3u, k); }
        }
      }
      tb[kk * 17u + c4] = vb;
    }
    workgroupBarrier();
    if (k0 + 32u <= p.k) {
      ${unrolled}
    } else {
      let kRem = p.k - k0;
      for (var kk = 0u; kk < kRem; kk = kk + 1u) {
        let av = ta[ty * 33u + kk];
        let bv = vec4<f32>(tb[kk * 17u + tx]);
        acc = fma(vec4<f32>(av), bv, acc);
      }
    }
    workgroupBarrier();
  }

  if (!inBatch) { return; }
  let m = m0 + ty;
  if (m < p.m) {
    let rowC = bat * p.sC0 + m * p.n;
    if (col0 < p.n) { writeOut(rowC + col0, acc.x); }
    if (col0 + 1u < p.n) { writeOut(rowC + col0 + 1u, acc.y); }
    if (col0 + 2u < p.n) { writeOut(rowC + col0 + 2u, acc.z); }
    if (col0 + 3u < p.n) { writeOut(rowC + col0 + 3u, acc.w); }
  }
}
`;
}

export function gemm64x16Source({ dtype = "F32", useF16 = false } = {}) {
  const { header, readB, epilogueFn, EB } = weightLoaderDecl(dtype, useF16);
  const unrolled = Array.from(
    { length: 32 },
    (_, kk) => `{
        let av = ta[${kk * 65}u + ty];
        let bv = vec4<f32>(tb[${kk * 5}u + tx]);
        acc = fma(vec4<f32>(av), bv, acc);
      }`,
  ).join("\n      ");

  return `${header}
${readB}
${epilogueFn}
var<workgroup> ta: array<f32, 2080>;
var<workgroup> tb: array<vec4<${EB}>, 160>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesN = (p.n + 15u) / 16u;
  let tilesM = (p.m + 63u) / 64u;
  let tilesPerBatch = tilesN * tilesM;
  let flat = wid.x + wid.y * 65535u;
  let batch = flat / tilesPerBatch;
  let inBatch = batch < p.batches;
  let bat = min(batch, p.batches - 1u);
  let tile = flat % tilesPerBatch;
  let m0 = (tile / tilesN) * 64u;
  let n0 = (tile % tilesN) * 16u;
  let baseA = bat * p.sA0;
  let baseB = bat * p.sB0;

  let ty = li / 4u;
  let tx = li % 4u;
  let col0 = n0 + tx * 4u;

  var acc = vec4<f32>(0.0);
  if (p.hasBias != 0u) {
    if (col0 < p.n) { acc.x = bias[col0]; }
    if (col0 + 1u < p.n) { acc.y = bias[col0 + 1u]; }
    if (col0 + 2u < p.n) { acc.z = bias[col0 + 2u]; }
    if (col0 + 3u < p.n) { acc.w = bias[col0 + 3u]; }
  }

  for (var k0 = 0u; k0 < p.k; k0 = k0 + 32u) {
    for (var e = 0u; e < 8u; e = e + 1u) {
      let slot = li + e * 256u;
      let r = slot / 32u;
      let kk = slot % 32u;
      var va = 0.0;
      if (inBatch && m0 + r < p.m && k0 + kk < p.k) {
        va = a[baseA + (m0 + r) * p.k + k0 + kk];
      }
      ta[kk * 65u + r] = va;
    }
    if (li < 128u) {
      let kk = li / 4u;
      let c4 = li % 4u;
      let k = k0 + kk;
      var vb = vec4<${EB}>(${EB}(0.0));
      if (inBatch && k < p.k) {
        let rB = n0 + c4 * 4u;
        if (rB + 3u < p.n) {
          vb = vec4<${EB}>(
            loadB(baseB, rB, k),
            loadB(baseB, rB + 1u, k),
            loadB(baseB, rB + 2u, k),
            loadB(baseB, rB + 3u, k)
          );
        } else {
          if (rB < p.n) { vb.x = loadB(baseB, rB, k); }
          if (rB + 1u < p.n) { vb.y = loadB(baseB, rB + 1u, k); }
          if (rB + 2u < p.n) { vb.z = loadB(baseB, rB + 2u, k); }
          if (rB + 3u < p.n) { vb.w = loadB(baseB, rB + 3u, k); }
        }
      }
      tb[kk * 5u + c4] = vb;
    }
    workgroupBarrier();
    if (k0 + 32u <= p.k) {
      ${unrolled}
    } else {
      let kRem = p.k - k0;
      for (var kk = 0u; kk < kRem; kk = kk + 1u) {
        let av = ta[kk * 65u + ty];
        let bv = vec4<f32>(tb[kk * 5u + tx]);
        acc = fma(vec4<f32>(av), bv, acc);
      }
    }
    workgroupBarrier();
  }

  if (!inBatch) { return; }
  let m = m0 + ty;
  if (m < p.m) {
    let rowC = bat * p.sC0 + m * p.n;
    if (col0 < p.n) { writeOut(rowC + col0, acc.x); }
    if (col0 + 1u < p.n) { writeOut(rowC + col0 + 1u, acc.y); }
    if (col0 + 2u < p.n) { writeOut(rowC + col0 + 2u, acc.z); }
    if (col0 + 3u < p.n) { writeOut(rowC + col0 + 3u, acc.w); }
  }
}
`;
}

export function gemm32x32Source() {
  const { header, readB, epilogueFn } = weightLoaderDecl("F32", false);
  const unrolled = Array.from(
    { length: 16 },
    (_, kk) => `{
        let av = ta[${kk * 17}u + ty];
        let bv = tb[${kk * 17}u + tx];
        acc0 = fma(vec2<f32>(av.x), bv, acc0);
        acc1 = fma(vec2<f32>(av.y), bv, acc1);
      }`,
  ).join("\n      ");

  return `${header}
${readB}
${epilogueFn}
var<workgroup> ta: array<vec2<f32>, 272>;
var<workgroup> tb: array<vec2<f32>, 272>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesN = (p.n + 31u) / 32u;
  let tilesM = (p.m + 31u) / 32u;
  let tilesPerBatch = tilesN * tilesM;
  let flat = wid.x + wid.y * 65535u;
  let batch = flat / tilesPerBatch;
  let inBatch = batch < p.batches;
  let bat = min(batch, p.batches - 1u);
  let tile = flat % tilesPerBatch;
  let m0 = (tile / tilesN) * 32u;
  let n0 = (tile % tilesN) * 32u;
  let baseA = bat * p.sA0;
  let baseB = bat * p.sB0;

  let ty = li / 16u;
  let tx = li % 16u;
  let col0 = n0 + tx * 2u;

  var init = vec2<f32>(0.0);
  if (p.hasBias != 0u) {
    if (col0 < p.n) { init.x = bias[col0]; }
    if (col0 + 1u < p.n) { init.y = bias[col0 + 1u]; }
  }
  var acc0 = init;
  var acc1 = init;

  let kkLoad = li % 16u;
  let r2Load = li / 16u;

  for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {
    let k = k0 + kkLoad;
    {
      var va = vec2<f32>(0.0);
      let rA = m0 + r2Load * 2u;
      if (inBatch && k < p.k) {
        if (rA < p.m) { va.x = a[baseA + rA * p.k + k]; }
        if (rA + 1u < p.m) { va.y = a[baseA + (rA + 1u) * p.k + k]; }
      }
      ta[kkLoad * 17u + r2Load] = va;
    }
    {
      var vb = vec2<f32>(0.0);
      let rB = n0 + r2Load * 2u;
      if (inBatch && k < p.k) {
        if (rB < p.n) { vb.x = loadB(baseB, rB, k); }
        if (rB + 1u < p.n) { vb.y = loadB(baseB, rB + 1u, k); }
      }
      tb[kkLoad * 17u + r2Load] = vb;
    }
    workgroupBarrier();
    if (k0 + 16u <= p.k) {
      ${unrolled}
    } else {
      let kRem = p.k - k0;
      for (var kk = 0u; kk < kRem; kk = kk + 1u) {
        let av = ta[kk * 17u + ty];
        let bv = tb[kk * 17u + tx];
        acc0 = fma(vec2<f32>(av.x), bv, acc0);
        acc1 = fma(vec2<f32>(av.y), bv, acc1);
      }
    }
    workgroupBarrier();
  }

  if (!inBatch) { return; }
  let baseC = bat * p.sC0;
  let r0 = m0 + ty * 2u;
  if (r0 < p.m) {
    if (col0 < p.n) { writeOut(baseC + r0 * p.n + col0, acc0.x); }
    if (col0 + 1u < p.n) { writeOut(baseC + r0 * p.n + col0 + 1u, acc0.y); }
  }
  if (r0 + 1u < p.m) {
    if (col0 < p.n) { writeOut(baseC + (r0 + 1u) * p.n + col0, acc1.x); }
    if (col0 + 1u < p.n) { writeOut(baseC + (r0 + 1u) * p.n + col0 + 1u, acc1.y); }
  }
}
`;
}

export const SE3_EOC_ECK_SOURCE = `
struct P { E: u32, O: u32, C: u32, K: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> shB: array<vec4<f32>, 128>;

@compute @workgroup_size(64, 4)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tilesO = (p.O + 63u) / 64u;
  let flat = wid.x + wid.y * 65535u;
  let edgeGroup = flat / tilesO;
  let oTile = flat % tilesO;
  let e = edgeGroup * 4u + lid.y;
  let oCol = oTile * 64u + lid.x;
  let validE = e < p.E;
  let validO = validE && (oCol < p.O);
  let aRowBase = (e * p.O + oCol) * p.C;
  let bEdgeBase = e * p.C * p.K;

  var acc = vec4<f32>(0.0);
  for (var c0 = 0u; c0 < p.C; c0 = c0 + 32u) {
    if (lid.x < 32u) {
      var v = vec4<f32>(0.0);
      let cIdx = c0 + lid.x;
      if (validE && cIdx < p.C) {
        let base = bEdgeBase + cIdx * p.K;
        v.x = b[base];
        if (p.K > 1u) { v.y = b[base + 1u]; }
        if (p.K > 2u) { v.z = b[base + 2u]; }
        if (p.K > 3u) { v.w = b[base + 3u]; }
      }
      shB[lid.y * 32u + lid.x] = v;
    }
    workgroupBarrier();
    let cRem = min(32u, p.C - c0);
    if (validO) {
      for (var cc = 0u; cc < cRem; cc = cc + 1u) {
        let av = a[aRowBase + c0 + cc];
        acc = fma(vec4<f32>(av), shB[lid.y * 32u + cc], acc);
      }
    }
    workgroupBarrier();
  }
  if (!validO) { return; }
  let outBase = (e * p.O + oCol) * p.K;
  out[outBase] = acc.x;
  if (p.K > 1u) { out[outBase + 1u] = acc.y; }
  if (p.K > 2u) { out[outBase + 2u] = acc.z; }
  if (p.K > 3u) { out[outBase + 3u] = acc.w; }
}
`;

const RF_ACCURATE_MATH = `
fn rfdiv(a:f32,b:f32)->f32 {
  let signBit=(bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u;
  let x=abs(a);let y=abs(b);var bits=bitcast<u32>(x/y);
  if(bits==0u || bits>=0x7f800000u){return a/b;}
  for(var k=0u;k<4u;k++){
    let q=bitcast<f32>(bits);let e=fma(-q,y,x);
    let up=e>0.0;let nextBits=select(bits-1u,bits+1u,up);
    let halfGap=abs(bitcast<f32>(nextBits)-q)*0.5;
    let boundary=halfGap*y;
    if(abs(e)>boundary || (abs(e)==boundary && (bits&1u)==1u)){bits=nextBits;}else{break;}
  }
  return bitcast<f32>(bits|signBit);
}
fn rfSqrt(x:f32)->f32 {let y=sqrt(x);if(x==0.0){return 0.0;}return y+fma(-y,y,x)/(2.0*y);}
`;

export const NORM_SOURCE = `${RF_ACCURATE_MATH}
struct P {
  rows: u32, width: u32, isGroup: u32, hasRelu: u32,
  eps: f32, epsLo: f32, pad1: u32, pad2: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> shX: array<f32, 512>;
var<workgroup> means: array<f32, 32>;
var<workgroup> vars: array<f32, 32>;
var<workgroup> counts: array<u32, 8>;
var<workgroup> rowMean: f32;
var<workgroup> rowInv: f32;

fn sealNorm(val: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(val) ^ p.pad1);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let xb = row * p.width;

  if (p.width <= 512u && (p.width & 3u) == 0u) {
    // Coalesced single-pass load into SRAM + parallel-chunk exact ARM64 Welford!
    for (var c = li; c < p.width; c = c + 64u) {
      shX[c] = x[xb + c];
    }
    workgroupBarrier();

    let chunks = (p.width + 63u) / 64u;
    let totalVecs = p.width / 4u;
    if (li < chunks * 4u) {
      let ch = li / 4u;
      let lane = li % 4u;
      let count = min(16u, totalVecs - ch * 16u);
      if (lane == 0u) { counts[ch] = count; }
      var m = 0.0;
      var v = 0.0;
      let chBase = ch * 64u + lane;
      for (var j = 0u; j < count; j = j + 1u) {
        let z = shX[chBase + j * 4u];
        let d = z - m;
        let rc = rfdiv(1.0, f32(j + 1u));
        m = fma(d, rc, m);
        v = fma(d, z - m, v);
      }
      means[li] = m;
      vars[li] = v;
    }
    workgroupBarrier();

    if (li == 0u) {
      for (var stride = 1u; stride < chunks; stride = stride * 2u) {
        for (var a = 0u; a + stride < chunks; a = a + stride * 2u) {
          let bIdx = a + stride;
          let n0 = counts[a];
          let n1 = counts[bIdx];
          let c = rfdiv(f32(n1), f32(n0 + n1));
          for (var lane = 0u; lane < 4u; lane = lane + 1u) {
            let pIdx = a * 4u + lane;
            let qIdx = bIdx * 4u + lane;
            let d = means[qIdx] - means[pIdx];
            means[pIdx] = fma(c, d, means[pIdx]);
            vars[pIdx] = vars[pIdx] + fma((d * d) * c, f32(n0), vars[qIdx]);
          }
          counts[a] = n0 + n1;
        }
      }
      var cnt = 0u;
      var m = 0.0;
      var v = 0.0;
      for (var lane = 0u; lane < 4u; lane = lane + 1u) {
        let n1 = counts[0];
        let c = rfdiv(f32(n1), f32(cnt + n1));
        let d = means[lane] - m;
        m = fma(c, d, m);
        v = v + fma((d * d) * c, f32(cnt), vars[lane]);
        cnt = cnt + n1;
      }
      let variance = sealNorm(v / f32(p.width));
      let xVar = sealNorm(variance + p.eps);
      let r0 = sealNorm(rfdiv(1.0, rfSqrt(xVar)));
      var inv = r0;
      if (p.isGroup != 0u) {
        let r2 = sealNorm(r0 * r0);
        let low = ((variance - xVar) + p.eps) + p.epsLo;
        let r2low = fma(r0, r0, -r2);
        let err = (fma(-r2, xVar, 1.0) - r2low * xVar) - r2 * low;
        inv = r0 + (0.5 * r0) * err;
      }
      rowMean = m;
      rowInv = inv;
    }
    workgroupBarrier();

    if (!live) { return; }
    let mean = rowMean;
    let inv = rowInv;
    if (p.isGroup != 0u) {
      for (var c = li; c < p.width; c = c + 64u) {
        let scale = inv * w[c];
        let bias = fma(-scale, mean, b[c]);
        var outV = fma(scale, shX[c], bias);
        if (p.hasRelu != 0u) { outV = max(outV, 0.0); }
        y[xb + c] = outV;
      }
    } else {
      for (var c = li; c < p.width; c = c + 64u) {
        let v = (shX[c] - mean) * inv;
        var outV = fma(v, w[c], b[c]);
        if (p.hasRelu != 0u) { outV = max(outV, 0.0); }
        y[xb + c] = outV;
      }
    }
    return;
  }

  // Fallback for width > 512
  var s = 0.0;
  for (var c = li; c < p.width; c = c + 64u) { s = s + x[xb + c]; }
  shX[li] = s;
  workgroupBarrier();
  if (li < 32u) { shX[li] = shX[li] + shX[li + 32u]; } workgroupBarrier();
  if (li < 16u) { shX[li] = shX[li] + shX[li + 16u]; } workgroupBarrier();
  if (li < 8u) { shX[li] = shX[li] + shX[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    rowMean = (((shX[0] + shX[4]) + (shX[2] + shX[6])) + ((shX[1] + shX[5]) + (shX[3] + shX[7]))) / f32(p.width);
  }
  workgroupBarrier();
  let mean = rowMean;
  var q = 0.0;
  for (var c = li; c < p.width; c = c + 64u) { let d = x[xb + c] - mean; q = fma(d, d, q); }
  shX[li] = q;
  workgroupBarrier();
  if (li < 32u) { shX[li] = shX[li] + shX[li + 32u]; } workgroupBarrier();
  if (li < 16u) { shX[li] = shX[li] + shX[li + 16u]; } workgroupBarrier();
  if (li < 8u) { shX[li] = shX[li] + shX[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    let variance = (((shX[0] + shX[4]) + (shX[2] + shX[6])) + ((shX[1] + shX[5]) + (shX[3] + shX[7]))) / f32(p.width);
    rowInv = rfdiv(1.0, rfSqrt(variance + p.eps));
  }
  workgroupBarrier();
  if (!live) { return; }
  let inv = rowInv;
  for (var c = li; c < p.width; c = c + 64u) {
    let v = (x[xb + c] - mean) * inv;
    var outV = fma(v, w[c], b[c]);
    if (p.hasRelu != 0u) { outV = max(outV, 0.0); }
    y[xb + c] = outV;
  }
}
`;

export const SOFTMAX_SOURCE = `${numericWGSL}${RF_ACCURATE_MATH}
struct P { slices: u32, n: u32, inner: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 64>;
var<workgroup> ew: array<f32, 512>;
var<workgroup> rowSum: f32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let sIdx = wid.x + wid.y * 65535u;
  let live = sIdx < p.slices;
  let sl = min(sIdx, p.slices - 1u);
  let outerIdx = sl / p.inner;
  let innerIdx = sl % p.inner;
  let base = outerIdx * (p.n * p.inner) + innerIdx;

  var mx = -3.402823e38;
  for (var j = li; j < p.n; j = j + 64u) {
    mx = max(mx, x[base + j * p.inner]);
  }
  red[li] = mx;
  workgroupBarrier();
  if (li < 32u) { red[li] = max(red[li], red[li + 32u]); } workgroupBarrier();
  if (li < 16u) { red[li] = max(red[li], red[li + 16u]); } workgroupBarrier();
  if (li < 8u) { red[li] = max(red[li], red[li + 8u]); } workgroupBarrier();
  if (li == 0u) {
    red[0] = max(max(max(red[0], red[4]), max(red[2], red[6])),
                 max(max(red[1], red[5]), max(red[3], red[7])));
  }
  workgroupBarrier();
  let top = red[0];
  workgroupBarrier();

  if (p.n <= 512u) {
    for (var j = li; j < p.n; j = j + 64u) {
      ew[j] = rfExp(x[base + j * p.inner] - top);
    }
    workgroupBarrier();
    if (li == 0u) {
      if (p.inner > 1u) {
        var s = 0.0;
        for (var j = 0u; j < p.n; j = j + 1u) { s = s + ew[j]; }
        rowSum = s;
      } else {
        var lanes: array<f32, 4>;
        for (var j = 0u; j < p.n; j = j + 1u) {
          let k = j % 4u;
          lanes[k] = lanes[k] + ew[j];
        }
        if (p.n < 4u) {
          rowSum = (lanes[0] + lanes[1]) + lanes[2];
        } else {
          rowSum = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
        }
      }
    }
    workgroupBarrier();
    if (!live) { return; }
    let total = rowSum;
    if (p.inner == 1u) {
      let inv = rfdiv(1.0, total);
      for (var j = li; j < p.n; j = j + 64u) {
        y[base + j] = ew[j] * inv;
      }
    } else {
      for (var j = li; j < p.n; j = j + 64u) {
        y[base + j * p.inner] = rfdiv(ew[j], total);
      }
    }
    return;
  }

  var s = 0.0;
  for (var j = li; j < p.n; j = j + 64u) {
    s = s + rfExp(x[base + j * p.inner] - top);
  }
  red[li] = s;
  workgroupBarrier();
  if (li < 32u) { red[li] = red[li] + red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] = red[li] + red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] = red[li] + red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    rowSum = ((red[0] + red[4]) + (red[2] + red[6]))
           + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  if (!live) { return; }
  let total = rowSum;
  if (p.inner == 1u) {
    let inv = rfdiv(1.0, total);
    for (var j = li; j < p.n; j = j + 64u) { y[base + j] = rfExp(x[base + j] - top) * inv; }
  } else {
    for (var j = li; j < p.n; j = j + 64u) { y[base + j * p.inner] = rfdiv(rfExp(x[base + j * p.inner] - top), total); }
  }
}
`;

export const EDGE_SOFTMAX_SOURCE = `${RF_ACCURATE_MATH}
struct P { n: u32, inner: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> tables: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 64>;
var<workgroup> ew: array<f32, 512>;
var<workgroup> colSum: f32;

fn sealEdge(val: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(val) ^ p.pad0);
}
struct Pair { hi: f32, lo: f32 };
fn pairMul(a: Pair, b: Pair) -> Pair {
  let hi = sealEdge(a.hi * b.hi);
  let err = fma(a.hi, b.hi, -hi);
  let lo = sealEdge(((err + a.hi * b.lo) + a.lo * b.hi) + a.lo * b.lo);
  return Pair(hi, lo);
}
fn pairAdd(a: Pair, b: Pair) -> Pair {
  let hi = sealEdge(a.hi + b.hi);
  let bv = sealEdge(hi - a.hi);
  let lo = sealEdge(((a.hi - (hi - bv)) + (b.hi - bv)) + (a.lo + b.lo));
  return Pair(hi, lo);
}
fn armScalarExp(rawVal: f32) -> f32 {
  let val = sealEdge(rawVal);
  if (val <= -104.0) { return 0.0; }
  let xc = sealEdge(max(val, -104.0));
  let invLn2 = Pair(46.16624069213867, 6.163081707200035e-7);
  let z = pairMul(Pair(xc, 0.0), invLn2);
  var q = floor(z.hi + 0.5);
  if (z.hi + 0.5 == q && z.lo < 0.0) { q = q - 1.0; }
  q = sealEdge(q);
  let r = Pair(sealEdge(z.hi - q), z.lo);
  let qi = i32(q);
  let j = u32(((qi % 32) + 32) % 32);
  let expo = i32(floor(f32(qi) / 32.0));
  let s = Pair(sealEdge(ldexp(tables[j], expo)), sealEdge(ldexp(tables[32u + j], expo)));
  let c0 = Pair(1.6938358839979628e-6, 4.109405754592656e-14);
  let c1 = Pair(0.00023459810472559184, -6.8305018759151626e-12);
  let c2 = Pair(0.021660849452018738, -5.5405603716085494e-11);
  let pPoly = pairAdd(pairMul(c0, r), c1);
  let r2 = pairMul(r, r);
  let lin = pairAdd(pairMul(c2, r), Pair(1.0, 0.0));
  let poly = pairAdd(pairMul(pPoly, r2), lin);
  let ans = pairMul(poly, s);
  return sealEdge(ans.hi + ans.lo);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let col = wid.x + wid.y * 65535u;
  let live = col < p.inner;
  let c = min(col, p.inner - 1u);

  var mx = -3.402823e38;
  for (var j = li; j < p.n; j = j + 64u) {
    mx = max(mx, x[j * p.inner + c]);
  }
  red[li] = mx;
  workgroupBarrier();
  if (li < 32u) { red[li] = max(red[li], red[li + 32u]); } workgroupBarrier();
  if (li < 16u) { red[li] = max(red[li], red[li + 16u]); } workgroupBarrier();
  if (li < 8u) { red[li] = max(red[li], red[li + 8u]); } workgroupBarrier();
  if (li == 0u) {
    red[0] = max(max(max(red[0], red[4]), max(red[2], red[6])),
                 max(max(red[1], red[5]), max(red[3], red[7])));
  }
  workgroupBarrier();
  let top = red[0];
  workgroupBarrier();

  if (p.n <= 512u) {
    for (var j = li; j < p.n; j = j + 64u) {
      ew[j] = armScalarExp(x[j * p.inner + c] - top);
    }
    workgroupBarrier();
    if (li == 0u) {
      var s = 0.0;
      for (var j = 0u; j < p.n; j = j + 1u) { s = s + ew[j]; }
      colSum = s;
    }
    workgroupBarrier();
    if (!live) { return; }
    let total = colSum;
    for (var j = li; j < p.n; j = j + 64u) {
      y[j * p.inner + c] = rfdiv(ew[j], total);
    }
    return;
  }

  var s = 0.0;
  for (var j = li; j < p.n; j = j + 64u) {
    let ev = armScalarExp(x[j * p.inner + c] - top);
    if (live) { y[j * p.inner + c] = ev; }
    s = s + ev;
  }
  red[li] = s;
  workgroupBarrier();
  if (li < 32u) { red[li] = red[li] + red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] = red[li] + red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] = red[li] + red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    colSum = ((red[0] + red[4]) + (red[2] + red[6]))
           + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let total = colSum;
  if (!live) { return; }
  for (var j = li; j < p.n; j = j + 64u) {
    y[j * p.inner + c] = rfdiv(y[j * p.inner + c], total);
  }
}
`;

export function permuteOpSource(dtype = "F32") {
  let loadExpr = "dst[i] = src[srcIdx];";
  if (dtype === "F16") {
    loadExpr = "{ let u = unpack2x16float(src[srcIdx >> 1u]); dst[i] = select(u.x, u.y, (srcIdx & 1u) != 0u); }";
  } else if (dtype === "I8") {
    loadExpr = "{ let q = (i32(src[srcIdx >> 2u] >> ((srcIdx & 3u) * 8u)) << 24u) >> 24u; dst[i] = f32(q) * bitcast<f32>(src[p.scaleOff + srcIdx / p.rowWidth]); }";
  }
  return `
struct P {
  count: u32, rank: u32, scaleOff: u32, rowWidth: u32,
  os0: vec4<u32>, os1: vec4<u32>,
  ot0: vec4<u32>, ot1: vec4<u32>,
  is0: vec4<u32>, is1: vec4<u32>,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<${dtype === "F32" ? "f32" : "u32"}>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.count) { return; }
  var srcIdx = 0u;
  if (p.rank >= 1u) { srcIdx += ((i / p.ot0.x) % p.os0.x) * p.is0.x; }
  if (p.rank >= 2u) { srcIdx += ((i / p.ot0.y) % p.os0.y) * p.is0.y; }
  if (p.rank >= 3u) { srcIdx += ((i / p.ot0.z) % p.os0.z) * p.is0.z; }
  if (p.rank >= 4u) { srcIdx += ((i / p.ot0.w) % p.os0.w) * p.is0.w; }
  if (p.rank >= 5u) { srcIdx += ((i / p.ot1.x) % p.os1.x) * p.is1.x; }
  if (p.rank >= 6u) { srcIdx += ((i / p.ot1.y) % p.os1.y) * p.is1.y; }
  ${loadExpr}
}
`;
}

export const BINARY_OP_SOURCE = `${numericWGSL}${RF_ACCURATE_MATH}
struct P {
  count: u32, op: u32, mode: u32, rank: u32,
  scalarBits: u32, pad0: u32, pad1: u32, pad2: u32,
  outShape: vec4<u32>,
  outStride: vec4<u32>,
  aStride: vec4<u32>,
  bStride: vec4<u32>,
  ext: vec4<u32>,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.count) { return; }
  var va = 0.0;
  var vb = 0.0;
  if (p.mode == 1u) {
    va = a[i];
    vb = bitcast<f32>(p.scalarBits);
  } else if (p.mode == 2u) {
    va = bitcast<f32>(p.scalarBits);
    vb = a[i];
  } else if (p.rank == 0u) {
    va = a[i];
    vb = b[i];
  } else {
    var ia = 0u;
    var ib = 0u;
    if (p.rank >= 1u) { let c0 = (i / p.outStride.x) % p.outShape.x; ia += c0 * p.aStride.x; ib += c0 * p.bStride.x; }
    if (p.rank >= 2u) { let c1 = (i / p.outStride.y) % p.outShape.y; ia += c1 * p.aStride.y; ib += c1 * p.bStride.y; }
    if (p.rank >= 3u) { let c2 = (i / p.outStride.z) % p.outShape.z; ia += c2 * p.aStride.z; ib += c2 * p.bStride.z; }
    if (p.rank >= 4u) { let c3 = (i / p.outStride.w) % p.outShape.w; ia += c3 * p.aStride.w; ib += c3 * p.bStride.w; }
    if (p.rank >= 5u) { let c4 = (i / p.ext.y) % p.ext.x; ia += c4 * p.ext.z; ib += c4 * p.ext.w; }
    va = a[ia];
    vb = b[ib];
  }
  var res = 0.0;
  if (p.op == 0u) { res = va + vb; }
  else if (p.op == 1u) { res = va - vb; }
  else if (p.op == 2u) { res = va * vb; }
  else if (p.op == 3u) { res = rfdiv(va, vb); }
  else { res = va * rfdiv(1.0, 1.0 + rfExp(-vb)); }
  out[i] = res;
}
`;

export const UNARY_OP_SOURCE = `${numericWGSL}${RF_ACCURATE_MATH}
struct P { count: u32, op: u32, scalarBits: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.count) { return; }
  let c = bitcast<f32>(p.scalarBits);
  if (p.op == 4u) { out[i] = c; return; }
  let x = a[i];
  var res = x;
  if (p.op == 0u) { res = max(x, 0.0); }
  else if (p.op == 1u) { res = rfdiv(1.0, 1.0 + rfExp(-x)); }
  else if (p.op == 2u) { res = rfExp(-x); }
  else if (p.op == 3u) { res = max(x, c); }
  out[i] = res;
}
`;

export const SLICE_OP_SOURCE = `
struct P { count: u32, axisIn: u32, axisOut: u32, inner: u32, start: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.count) { return; }
  let rem = i / p.inner;
  let inIdx = i % p.inner;
  let ax = rem % p.axisOut;
  let outer = rem / p.axisOut;
  dst[i] = src[(outer * p.axisIn + ax + p.start) * p.inner + inIdx];
}
`;

export function gatherOpSource(dtype = "F32") {
  let writeBody = "out[i] = tbl[srcIdx];";
  if (dtype === "F16") {
    writeBody = "let u = unpack2x16float(tbl[srcIdx >> 1u]); out[i] = select(u.x, u.y, (srcIdx & 1u) != 0u);";
  } else if (dtype === "I8") {
    writeBody = "let q = (i32(tbl[srcIdx >> 2u] >> ((srcIdx & 3u) * 8u)) << 24u) >> 24u; out[i] = f32(q) * bitcast<f32>(tbl[p.scaleOff + row]);";
  }
  return `
struct P { count: u32, width: u32, scaleOff: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tbl: array<${dtype === "F32" ? "f32" : "u32"}>;
@group(0) @binding(2) var<storage, read> idx: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.count) { return; }
  let row = u32(idx[i / p.width]);
  let col = i % p.width;
  let srcIdx = row * p.width + col;
  ${writeBody}
}
`;
}

export const SUM_AXIS0_SOURCE = `
struct P { A: u32, M: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.M) { return; }
  var v = 0.0;
  for (var a = 0u; a < p.A; a = a + 1u) {
    v = v + x[a * p.M + i];
  }
  out[i] = v;
}
`;

// === RFdiffusion Domain-Specific Fused WGSL Kernels ===

/**
 * Fused RBF encoding [N, 36]: replaces 4 [L, L, 36] kernels + 1 upload per block!
 */
export const RBF_FUSED_SOURCE = `${numericWGSL}${RF_ACCURATE_MATH}
struct P { count: u32, invSigma: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dist: array<f32>;
@group(0) @binding(2) var<storage, read> mu: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.count) { return; }
  let d = bitcast<f32>(bitcast<u32>(dist[i / 36u] - mu[i % 36u]) ^ p.pad0);
  let q = bitcast<f32>(bitcast<u32>(rfdiv(d, p.invSigma)) ^ p.pad0);
  let q2 = bitcast<f32>(bitcast<u32>(q * q) ^ p.pad0);
  out[i] = rfExp(-q2);
}
`;
