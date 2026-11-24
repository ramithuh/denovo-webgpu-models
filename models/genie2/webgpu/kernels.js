// Reusable Naga/Dawn-compatible tiled WGSL compute kernels for Genie 2 GPUAlgebra.
// Strict WebGPU floor compliance: workgroup_size <= 256, workgroupStorageSize <= 16384 B,
// 2D-folded dispatch < 65535, uniform control flow across all barriers, zero dynamic const/value-array indexing.

import { numericWGSL } from "../shared/numeric.js";
import { eigenBody } from "../shared/quaternion.js";

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

export const NORM_SOURCE = `
struct P {
  rows: u32, width: u32, isGroup: u32, hasRelu: u32,
  eps: f32, epsLo: f32, pad1: u32, pad2: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let xb = row * p.width;

  var s = 0.0;
  for (var c = li; c < p.width; c = c + 64u) {
    s = s + x[xb + c];
  }
  red[li] = s;
  workgroupBarrier();
  if (li < 32u) { red[li] = red[li] + red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] = red[li] + red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] = red[li] + red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
           + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let mean = red[0] / f32(p.width);
  workgroupBarrier();

  var q = 0.0;
  for (var c = li; c < p.width; c = c + 64u) {
    let d = x[xb + c] - mean;
    q = fma(d, d, q);
  }
  red[li] = q;
  workgroupBarrier();
  if (li < 32u) { red[li] = red[li] + red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] = red[li] + red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] = red[li] + red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
           + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let variance = red[0] / f32(p.width);
  var inv = inverseSqrt(variance + p.eps);
  {
    let xVar = variance + p.eps;
    let r0 = inv;
    let r2 = r0 * r0;
    let low = select(0.0, ((variance - xVar) + p.eps) + p.epsLo, p.isGroup != 0u);
    let r2low = fma(r0, r0, -r2);
    let err = (fma(-r2, xVar, 1.0) - r2low * xVar) - r2 * low;
    inv = r0 + (0.5 * r0) * err;
  }
  if (!live) { return; }
  if (p.isGroup != 0u) {
    for (var c = li; c < p.width; c = c + 64u) {
      let scale = inv * w[c];
      let bias = fma(-scale, mean, b[c]);
      var outV = fma(scale, x[xb + c], bias);
      if (p.hasRelu != 0u) { outV = max(outV, 0.0); }
      y[xb + c] = outV;
    }
  } else {
    for (var c = li; c < p.width; c = c + 64u) {
      let v = (x[xb + c] - mean) * inv;
      var outV = fma(v, w[c], b[c]);
      if (p.hasRelu != 0u) { outV = max(outV, 0.0); }
      y[xb + c] = outV;
    }
  }
}
`;

export const TRANSPOSE_NORM_SOURCE = `
struct P { rows: u32, width: u32, eps: f32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);

  var s = 0.0;
  for (var c = li; c < p.width; c = c + 64u) {
    s = s + x[c * p.rows + row];
  }
  red[li] = s;
  workgroupBarrier();
  if (li < 32u) { red[li] = red[li] + red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] = red[li] + red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] = red[li] + red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
           + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let mean = red[0] / f32(p.width);
  workgroupBarrier();

  var q = 0.0;
  for (var c = li; c < p.width; c = c + 64u) {
    let d = x[c * p.rows + row] - mean;
    q = fma(d, d, q);
  }
  red[li] = q;
  workgroupBarrier();
  if (li < 32u) { red[li] = red[li] + red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] = red[li] + red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] = red[li] + red[li + 8u]; } workgroupBarrier();
  if (li == 0u) {
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
           + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let variance = red[0] / f32(p.width);
  var inv = inverseSqrt(variance + p.eps);
  {
    let xVar = variance + p.eps;
    let r0 = inv;
    let r2 = r0 * r0;
    let r2low = fma(r0, r0, -r2);
    let err = fma(-r2, xVar, 1.0) - r2low * xVar;
    inv = r0 + (0.5 * r0) * err;
  }
  if (!live) { return; }
  let yb = row * p.width;
  for (var c = li; c < p.width; c = c + 64u) {
    let v = (x[c * p.rows + row] - mean) * inv;
    y[yb + c] = fma(v, w[c], b[c]);
  }
}
`;

export const SOFTMAX_SOURCE = `${numericWGSL}
struct P { slices: u32, n: u32, inner: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 64>;

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
    red[0] = ((red[0] + red[4]) + (red[2] + red[6]))
           + ((red[1] + red[5]) + (red[3] + red[7]));
  }
  workgroupBarrier();
  let total = red[0];
  let inv = 1.0 / total;
  if (!live) { return; }
  if (p.inner == 1u) {
    for (var j = li; j < p.n; j = j + 64u) {
      y[base + j] = rfExp(x[base + j] - top) * inv;
    }
  } else {
    for (var j = li; j < p.n; j = j + 64u) {
      y[base + j * p.inner] = rfExp(x[base + j * p.inner] - top) / total;
    }
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

export const BINARY_OP_SOURCE = `${numericWGSL}
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
  else if (p.op == 3u) { res = va / vb; }
  else { res = va * (1.0 / (1.0 + rfExp(-vb))); }
  out[i] = res;
}
`;

export const UNARY_OP_SOURCE = `${numericWGSL}
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
  else if (p.op == 1u) { res = 1.0 / (1.0 + rfExp(-x)); }
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

// === Genie 2 Domain-Specific Fused WGSL Kernels ===

/**
 * Fused PairFeatureNet._encode_orientations:
 * 1 thread per (i, j) pair (instead of 4x redundant eigensolver invocations + 4 dispatches!).
 */
const eigenCore = eigenBody
  .replace("for(var j=0u;j<16u;j++){a[j]=a0[base+j];v[j]=0.0;}", "for(var j=0u;j<16u;j++){v[j]=0.0;}")
  .replace("o[i]=v[component*4u+best]*signFlip;", `
let mVal = mask[rI] * mask[rJ];
let outOff = pairIdx * 4u;
out[outOff + 0u] = v[best] * signFlip * mVal;
out[outOff + 1u] = v[4u + best] * signFlip * mVal;
out[outOff + 2u] = v[8u + best] * signFlip * mVal;
out[outOff + 3u] = v[12u + best] * signFlip * mVal;
`)
  .replaceAll("sqrt(", "rfSqrt(");

export const ORIENTATIONS_FUSED_SOURCE = `
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

struct P { L: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> rots: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let pairIdx = (wid.x + wid.y * 65535u) * 64u + li;
  if (pairIdx >= p.L * p.L) { return; }
  let rI = pairIdx / p.L;
  let rJ = pairIdx % p.L;
  let bJ = rJ * 9u;
  let bI = rI * 9u;
  // Upstream R_j @ R_i (without transpose)
  let xx = fma(rots[bJ+2u], rots[bI+6u], fma(rots[bJ+1u], rots[bI+3u], fma(rots[bJ+0u], rots[bI+0u], 0.0)));
  let xy = fma(rots[bJ+2u], rots[bI+7u], fma(rots[bJ+1u], rots[bI+4u], fma(rots[bJ+0u], rots[bI+1u], 0.0)));
  let xz = fma(rots[bJ+2u], rots[bI+8u], fma(rots[bJ+1u], rots[bI+5u], fma(rots[bJ+0u], rots[bI+2u], 0.0)));
  let yx = fma(rots[bJ+5u], rots[bI+6u], fma(rots[bJ+4u], rots[bI+3u], fma(rots[bJ+3u], rots[bI+0u], 0.0)));
  let yy = fma(rots[bJ+5u], rots[bI+7u], fma(rots[bJ+4u], rots[bI+4u], fma(rots[bJ+3u], rots[bI+1u], 0.0)));
  let yz = fma(rots[bJ+5u], rots[bI+8u], fma(rots[bJ+4u], rots[bI+5u], fma(rots[bJ+3u], rots[bI+2u], 0.0)));
  let zx = fma(rots[bJ+8u], rots[bI+6u], fma(rots[bJ+7u], rots[bI+3u], fma(rots[bJ+6u], rots[bI+0u], 0.0)));
  let zy = fma(rots[bJ+8u], rots[bI+7u], fma(rots[bJ+7u], rots[bI+4u], fma(rots[bJ+6u], rots[bI+1u], 0.0)));
  let zz = fma(rots[bJ+8u], rots[bI+8u], fma(rots[bJ+7u], rots[bI+5u], fma(rots[bJ+6u], rots[bI+2u], 0.0)));

  var a: array<f32, 16>;
  var v: array<f32, 16>;
  var d: array<f32, 4>;
  var e: array<f32, 4>;
  var h: array<f32, 4>;
  var w: array<f32, 4>;
  var cs: array<f32, 4>;
  var sn: array<f32, 4>;
  let third = 0.3333333333333333;
  a[0] = ((xx + yy) + zz) * third;
  a[1] = (zy - yz) * third;
  a[2] = (xz - zx) * third;
  a[3] = (yx - xy) * third;
  a[4] = a[1];
  a[5] = ((xx - yy) - zz) * third;
  a[6] = (xy + yx) * third;
  a[7] = (xz + zx) * third;
  a[8] = a[2];
  a[9] = a[6];
  a[10] = ((yy - xx) - zz) * third;
  a[11] = (yz + zy) * third;
  a[12] = a[3];
  a[13] = a[7];
  a[14] = a[11];
  a[15] = ((zz - xx) - yy) * third;
  ${eigenCore
    .replace("var a:array<f32,16>;var v:array<f32,16>;var d:array<f32,4>;var e:array<f32,4>;", "")
    .replace("var h:array<f32,4>;var w:array<f32,4>;var cs:array<f32,4>;var sn:array<f32,4>;", "")}
}
`;

/**
 * Fused TriMul gate + pairMask + channel-major pack to [C, L, L]:
 * outgoing != 0: [i, k, c] -> [c, i, k]
 * outgoing == 0: [k, i, c] -> [c, i, k]
 */
export const TRI_GATE_PACK_SOURCE = `${numericWGSL}
struct P { L: u32, C: u32, outgoing: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ap: array<f32>;
@group(0) @binding(2) var<storage, read> ag: array<f32>;
@group(0) @binding(3) var<storage, read> bp: array<f32>;
@group(0) @binding(4) var<storage, read> bg: array<f32>;
@group(0) @binding(5) var<storage, read> pm: array<f32>;
@group(0) @binding(6) var<storage, read_write> aOut: array<f32>;
@group(0) @binding(7) var<storage, read_write> bOut: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let nn = p.L * p.L;
  if (idx >= nn * p.C) { return; }
  let c = idx % p.C;
  let pair = idx / p.C;
  let r0 = pair / p.L;
  let r1 = pair % p.L;
  let m = pm[pair];
  let aVal = (ap[idx] * (1.0 / (1.0 + rfExp(-ag[idx])))) * m;
  let bVal = (bp[idx] * (1.0 / (1.0 + rfExp(-bg[idx])))) * m;
  let dstPair = select(r1 * p.L + r0, pair, p.outgoing != 0u);
  let outIdx = c * nn + dstPair;
  aOut[outIdx] = aVal;
  bOut[outIdx] = bVal;
}
`;

/**
 * Fused TriMul / PairTransition residual update:
 * mode 0: out[i] = p[i] + x[i] * sigmoid(g[i])
 * mode 1: out[i] = (p[i] + x[i] * pm[i / C]) * pm[i / C]
 */
export const PAIR_RESIDUAL_SOURCE = `${numericWGSL}
struct P { count: u32, C: u32, mode: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pairIn: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> aux: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.count) { return; }
  if (p.mode == 0u) {
    out[i] = pairIn[i] + x[i] * (1.0 / (1.0 + rfExp(-aux[i])));
  } else {
    let m = aux[i / p.C];
    out[i] = (pairIn[i] + x[i] * m) * m;
  }
}
`;

/**
 * Fused Full IPA Attention Logits [H, L, L]:
 * Reads directly from q [L, H, K], kv [L, H, 2*K], qp [L, H, Q, 3], kvp [L, H, Q+V, 3], bias [L, L, H], headWeights [H], pm [L, L].
 */
export const IPA_LOGITS_FUSED_SOURCE = `
struct P {
  L: u32, H: u32, K: u32, Q: u32,
  QV: u32, pad0: u32, pad1: u32, pad2: u32,
  scalarScale: f32, pairScale: f32, ptScale: f32, infVal: f32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read> qp: array<f32>;
@group(0) @binding(4) var<storage, read> kvp: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read> hw: array<f32>;
@group(0) @binding(7) var<storage, read> pm: array<f32>;
@group(0) @binding(8) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let nn = p.L * p.L;
  if (idx >= p.H * nn) { return; }
  let pair = idx % nn;
  let h = idx / nn;
  let a = pair / p.L;
  let b = pair % p.L;

  let qBase = (a * p.H + h) * p.K;
  let kBase = (b * p.H + h) * (2u * p.K);
  var dotQK = 0.0;
  for (var d = 0u; d < p.K; d = d + 1u) {
    dotQK = fma(q[qBase + d], kv[kBase + d], dotQK);
  }

  let wHead = log(1.0 + exp(hw[h])) * p.ptScale;
  let qpBase = (a * p.H + h) * p.Q * 3u;
  let kpBase = (b * p.H + h) * p.QV * 3u;
  var ptSum = 0.0;
  for (var pt = 0u; pt < p.Q; pt = pt + 1u) {
    let dx = qp[qpBase + pt * 3u + 0u] - kvp[kpBase + pt * 3u + 0u];
    let dy = qp[qpBase + pt * 3u + 1u] - kvp[kpBase + pt * 3u + 1u];
    let dz = qp[qpBase + pt * 3u + 2u] - kvp[kpBase + pt * 3u + 2u];
    let dd = (dx * dx + dy * dy) + dz * dz;
    ptSum = ptSum + dd * wHead;
  }

  let v1 = dotQK * p.scalarScale + bias[pair * p.H + h] * p.pairScale;
  let v2 = v1 + (-0.5 * ptSum);
  logits[idx] = v2 + (pm[pair] - 1.0) * p.infVal;
}
`;

export const IPA_SCALAR_CONCAT_SOURCE = `
struct P { L: u32, H: u32, K: u32, concatDim: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> attn: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read_write> concat: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let hk = p.H * p.K;
  if (idx >= p.L * hk) { return; }
  let d = idx % p.K;
  let rem = idx / p.K;
  let h = rem % p.H;
  let i = rem / p.H;
  let attnBase = (h * p.L + i) * p.L;
  var acc = 0.0;
  for (var j = 0u; j < p.L; j = j + 1u) {
    let vIdx = (j * p.H + h) * (2u * p.K) + p.K + d;
    acc = fma(attn[attnBase + j], kv[vIdx], acc);
  }
  concat[i * p.concatDim + h * p.K + d] = acc;
}
`;

export const IPA_POINT_CONCAT_SOURCE = `
fn rfSqrt(x:f32)->f32 {let y=sqrt(x);if(x==0.0){return 0.0;}return y+fma(-y,y,x)/(2.0*y);}
struct P {
  L: u32, H: u32, Q: u32, V: u32,
  ptOff: u32, concatDim: u32, pad0: u32, pad1: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> attn: array<f32>;
@group(0) @binding(2) var<storage, read> kvp: array<f32>;
@group(0) @binding(3) var<storage, read> rots: array<f32>;
@group(0) @binding(4) var<storage, read> trans: array<f32>;
@group(0) @binding(5) var<storage, read_write> concat: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let hv = p.H * p.V;
  if (idx >= p.L * hv) { return; }
  let pt = idx % p.V;
  let rem = idx / p.V;
  let h = rem % p.H;
  let i = rem / p.H;

  let attnBase = (h * p.L + i) * p.L;
  let qv = p.Q + p.V;
  var gx = 0.0;
  var gy = 0.0;
  var gz = 0.0;
  for (var j = 0u; j < p.L; j = j + 1u) {
    let a = attn[attnBase + j];
    let vBase = ((j * p.H + h) * qv + p.Q + pt) * 3u;
    gx = fma(a, kvp[vBase + 0u], gx);
    gy = fma(a, kvp[vBase + 1u], gy);
    gz = fma(a, kvp[vBase + 2u], gz);
  }

  let t = i * 3u;
  let dx = gx - trans[t + 0u];
  let dy = gy - trans[t + 1u];
  let dz = gz - trans[t + 2u];
  let r = i * 9u;
  let lx = (rots[r + 0u] * dx + rots[r + 3u] * dy) + rots[r + 6u] * dz;
  let ly = (rots[r + 1u] * dx + rots[r + 4u] * dy) + rots[r + 7u] * dz;
  let lz = (rots[r + 2u] * dx + rots[r + 5u] * dy) + rots[r + 8u] * dz;
  let nrm = rfSqrt(((lx * lx + ly * ly) + lz * lz) + 1e-8);

  let hp = h * p.V + pt;
  let base = i * p.concatDim + p.ptOff;
  concat[base + hp] = lx;
  concat[base + hv + hp] = ly;
  concat[base + 2u * hv + hp] = lz;
  concat[base + 3u * hv + hp] = nrm;
}
`;

/**
 * Coalesced workgroup IPA pair output: 1 workgroup (128 threads) per (i, h) pair!
 * All 128 threads read adjacent channels z[(i*L + j)*P + c] in lockstep and write directly into concat.
 */
export const IPA_PAIR_CONCAT_SOURCE = `
struct P { L: u32, H: u32, cZ: u32, pairOff: u32, concatDim: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> attn: array<f32>;
@group(0) @binding(2) var<storage, read> z: array<f32>;
@group(0) @binding(3) var<storage, read_write> concat: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let ih = wid.x + wid.y * 65535u;
  if (ih >= p.L * p.H) { return; }
  let i = ih / p.H;
  let h = ih % p.H;
  let attnBase = (h * p.L + i) * p.L;
  let zRowBase = i * p.L * p.cZ;
  let outBase = i * p.concatDim + p.pairOff + h * p.cZ;

  for (var c = li; c < p.cZ; c = c + 128u) {
    var acc = 0.0;
    for (var j = 0u; j < p.L; j = j + 1u) {
      acc = fma(attn[attnBase + j], z[zRowBase + j * p.cZ + c], acc);
    }
    concat[outBase + c] = acc;
  }
}
`;

/**
 * Fused BackboneUpdate + SE(3) composition (1 thread per residue i).
 */
export const BACKBONE_UPDATE_FUSED_SOURCE = `
struct P { L: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read> rotsIn: array<f32>;
@group(0) @binding(3) var<storage, read> transIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> rotsOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> transOut: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 64u + li;
  if (i >= p.L) { return; }
  let pb = i * 6u;
  let b1 = params[pb + 0u];
  let c1 = params[pb + 1u];
  let d1 = params[pb + 2u];
  let tx = params[pb + 3u];
  let ty = params[pb + 4u];
  let tz = params[pb + 5u];

  let norm = sqrt(((b1 * b1 + c1 * c1) + d1 * d1) + 1.0);
  let a = 1.0 / norm;
  let b = b1 / norm;
  let c2 = c1 / norm;
  let d = d1 / norm;

  let d00 = ((a * a + b * b) - c2 * c2) - d * d;
  let d01 = -2.0 * a * d + 2.0 * b * c2;
  let d02 = 2.0 * a * c2 + 2.0 * b * d;
  let d10 = 2.0 * a * d + 2.0 * b * c2;
  let d11 = ((a * a - b * b) + c2 * c2) - d * d;
  let d12 = -2.0 * a * b + 2.0 * c2 * d;
  let d20 = -2.0 * a * c2 + 2.0 * b * d;
  let d21 = 2.0 * a * b + 2.0 * c2 * d;
  let d22 = ((a * a - b * b) - c2 * c2) + d * d;

  let r = i * 9u;
  let r00 = rotsIn[r + 0u]; let r01 = rotsIn[r + 1u]; let r02 = rotsIn[r + 2u];
  let r10 = rotsIn[r + 3u]; let r11 = rotsIn[r + 4u]; let r12 = rotsIn[r + 5u];
  let r20 = rotsIn[r + 6u]; let r21 = rotsIn[r + 7u]; let r22 = rotsIn[r + 8u];

  rotsOut[r + 0u] = (r00 * d00 + r01 * d10) + r02 * d20;
  rotsOut[r + 1u] = (r00 * d01 + r01 * d11) + r02 * d21;
  rotsOut[r + 2u] = (r00 * d02 + r01 * d12) + r02 * d22;
  rotsOut[r + 3u] = (r10 * d00 + r11 * d10) + r12 * d20;
  rotsOut[r + 4u] = (r10 * d01 + r11 * d11) + r12 * d21;
  rotsOut[r + 5u] = (r10 * d02 + r11 * d12) + r12 * d22;
  rotsOut[r + 6u] = (r20 * d00 + r21 * d10) + r22 * d20;
  rotsOut[r + 7u] = (r20 * d01 + r21 * d11) + r22 * d21;
  rotsOut[r + 8u] = (r20 * d02 + r21 * d12) + r22 * d22;

  let t = i * 3u;
  transOut[t + 0u] = ((r00 * tx + r01 * ty) + r02 * tz) + transIn[t + 0u];
  transOut[t + 1u] = ((r10 * tx + r11 * ty) + r12 * tz) + transIn[t + 1u];
  transOut[t + 2u] = ((r20 * tx + r21 * ty) + r22 * tz) + transIn[t + 2u];
}
`;

/**
 * Fused uniform-parameterized DDPM reverseStep (zero shader recompilations across 1000 steps!):
 */
export const REVERSE_STEP_FUSED_SOURCE = `
struct P {
  count: u32, hasNoise: u32, pad0: u32, pad1: u32,
  wz: f32, recip: f32, wNoise: f32, pad2: f32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> z: array<f32>;
@group(0) @binding(3) var<storage, read> noise: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.count) { return; }
  let m = mask[i / 3u];
  var mean = ((x[i] - z[i] * p.wz) * p.recip) * m;
  if (p.hasNoise != 0u) {
    mean = (mean + noise[i] * p.wNoise) * m;
  }
  out[i] = mean;
}
`;
