import {
  broadcast,
  contiguous,
  indexExpression,
  sizeOf,
  strides,
  tensor,
} from "../shared/tensor.js";

const MAX_GRID = 65535;
const fold2 = (groups) => {
  const g = Math.max(1, groups);
  return g <= MAX_GRID ? [g, 1, 1] : [MAX_GRID, Math.ceil(g / MAX_GRID), 1];
};

const fl = (x) => {
  if (x === Infinity) return "3.4028234663852886e38";
  if (x === -Infinity) return "-3.4028234663852886e38";
  if (!Number.isFinite(x)) throw Error(`Invalid shader literal ${x}`);
  const s = String(x);
  return /[.eE]/.test(s) ? s : s + ".0";
};
const ui = (x) => `${x}u`;

const UNIFORM_STRUCT = `struct U {
  iOff: vec4<u32>,
  dims: vec4<u32>,
  b0: vec4<u32>,
  b1: vec4<u32>,
  f0: vec4<f32>,
  f1: vec4<f32>,
};`;

const UNPACK_I8_FN = `fn unpack4i8(word: u32) -> vec4<f32> {
  let s = i32(word);
  return vec4<f32>(
    f32((s << 24u) >> 24u),
    f32((s << 16u) >> 24u),
    f32((s << 8u) >> 24u),
    f32(s >> 24u)
  );
}`;

function buildLinearGemm({ a, b, c, out, M, N, K, sAm, alpha, beta, hasF16 }) {
  const bEnc = b.storage.encoding ?? "F32";
  const hasBias = Boolean(c);
  const useF16Panel = Boolean(bEnc === "F16" && hasF16);
  const EB = useF16Panel ? "f16" : "f32";
  const large = Math.ceil(M / 64) * Math.ceil(N / 128) >= 20;
  const BM = large ? 64 : 32;
  const BN = large ? 64 * 2 : 64;
  const tilesM = Math.ceil(M / BM);
  const tilesN = Math.ceil(N / BN);
  const dispatch = fold2(tilesM * tilesN);

  const bindings = [
    { storage: a.storage, name: "a", write: false, binding: 0 },
    { storage: b.storage, name: "b", write: false, binding: 1 },
  ];
  let nextBind = 2;
  let scaleName = null;
  if (bEnc === "I8") {
    scaleName = "bScale";
    bindings.push({ storage: b.storage.scale, name: scaleName, write: false, binding: nextBind++ });
  }
  if (hasBias) {
    bindings.push({ storage: c.storage, name: "bias", write: false, binding: nextBind++ });
  }
  bindings.push({ storage: out.storage, name: "out", write: true, binding: nextBind++ });
  const uniformBinding = nextBind;

  const u = new Uint32Array(24);
  const f = new Float32Array(u.buffer);
  u[0] = a.offset;
  u[1] = b.offset;
  u[2] = c ? c.offset : 0;
  u[3] = out.offset;
  u[4] = M;
  u[5] = N;
  u[6] = K;
  u[7] = sAm;
  u[8] = tilesN;
  u[9] = tilesM * tilesN;
  f[16] = alpha;
  f[17] = beta;

  const header = [
    useF16Panel ? "enable f16;" : "",
    bEnc === "I8" ? UNPACK_I8_FN : "",
    UNIFORM_STRUCT,
    ...bindings.map(
      (x) =>
        `@group(0) @binding(${x.binding}) var<storage,${x.write ? "read_write" : "read"}> ${x.name}:array<${x.storage.encoding ? "u32" : "f32"}>;`,
    ),
    `@group(0) @binding(${uniformBinding}) var<uniform> u: U;`,
  ]
    .filter(Boolean)
    .join("\n");

  const readBias = (nExpr) => {
    if (!hasBias) return "0.0";
    if (!c.storage.encoding) return `bias[u.iOff.z + ${nExpr}] * u.f0.y`;
    if (c.storage.encoding === "F16")
      return `unpack2x16float(bias[(u.iOff.z + ${nExpr}) / 2u])[(u.iOff.z + ${nExpr}) % 2u] * u.f0.y`;
    return `0.0`;
  };

  if (large) {
    // 64x128 register-tiled GEMM (256 threads, 4x8 per thread, BK=16, <=12800B shared)
    const loadBSetup =
      bEnc === "F16"
        ? `let kkB = (li % 8u) * 2u;
  let r4B = li / 8u;
  let rowB0 = n0 + r4B * 4u;
  let sBnHalf = K / 2u;
  let bBase0 = (rowB0 * K + kkB) / 2u;
  let fullB = rowB0 + 3u < N;
  let tbWrite0 = kkB * 33u + r4B;
  let tbWrite1 = (kkB + 1u) * 33u + r4B;`
        : bEnc === "I8"
          ? `let kkB = (li % 4u) * 4u;
  let r4B = li / 4u;
  let rowB0 = n0 + r4B * 4u;
  let sBnQuad = K / 4u;
  let bBase0 = (rowB0 * K + kkB) / 4u;
  let fullB = rowB0 + 3u < N;
  var scB = vec4<f32>(0.0);
  if (li < 128u) {
    if (rowB0 < N) { scB.x = bScale[rowB0]; }
    if (rowB0 + 1u < N) { scB.y = bScale[rowB0 + 1u]; }
    if (rowB0 + 2u < N) { scB.z = bScale[rowB0 + 2u]; }
    if (rowB0 + 3u < N) { scB.w = bScale[rowB0 + 3u]; }
  }`
          : "";

    const loadBLoop =
      bEnc === "F16"
        ? `{
      var lo = vec4<${EB}>(0.0);
      var hi = vec4<${EB}>(0.0);
      if (inTile && k0 + kkB < K) {
        let bOff = bBase0 + (k0 / 2u);
        if (fullB) {
          let p0 = unpack2x16float(b[bOff]);
          let p1 = unpack2x16float(b[bOff + sBnHalf]);
          let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]);
          let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]);
          lo = vec4<${EB}>(${EB}(p0.x), ${EB}(p1.x), ${EB}(p2.x), ${EB}(p3.x));
          hi = vec4<${EB}>(${EB}(p0.y), ${EB}(p1.y), ${EB}(p2.y), ${EB}(p3.y));
        } else {
          if (rowB0 < N) { let p0 = unpack2x16float(b[bOff]); lo.x = ${EB}(p0.x); hi.x = ${EB}(p0.y); }
          if (rowB0 + 1u < N) { let p1 = unpack2x16float(b[bOff + sBnHalf]); lo.y = ${EB}(p1.x); hi.y = ${EB}(p1.y); }
          if (rowB0 + 2u < N) { let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]); lo.z = ${EB}(p2.x); hi.z = ${EB}(p2.y); }
          if (rowB0 + 3u < N) { let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]); lo.w = ${EB}(p3.x); hi.w = ${EB}(p3.y); }
        }
      }
      tb[tbWrite0] = lo;
      tb[tbWrite1] = hi;
    }`
        : bEnc === "I8"
          ? `if (li < 128u) {
      var q0 = vec4<f32>(0.0); var q1 = vec4<f32>(0.0);
      var q2 = vec4<f32>(0.0); var q3 = vec4<f32>(0.0);
      if (inTile && k0 + kkB < K) {
        let bOff = bBase0 + (k0 / 4u);
        if (fullB) {
          q0 = unpack4i8(b[bOff]) * scB.x;
          q1 = unpack4i8(b[bOff + sBnQuad]) * scB.y;
          q2 = unpack4i8(b[bOff + 2u * sBnQuad]) * scB.z;
          q3 = unpack4i8(b[bOff + 3u * sBnQuad]) * scB.w;
        } else {
          if (rowB0 < N) { q0 = unpack4i8(b[bOff]) * scB.x; }
          if (rowB0 + 1u < N) { q1 = unpack4i8(b[bOff + sBnQuad]) * scB.y; }
          if (rowB0 + 2u < N) { q2 = unpack4i8(b[bOff + 2u * sBnQuad]) * scB.z; }
          if (rowB0 + 3u < N) { q3 = unpack4i8(b[bOff + 3u * sBnQuad]) * scB.w; }
        }
      }
      tb[(kkB + 0u) * 33u + r4B] = vec4<f32>(q0.x, q1.x, q2.x, q3.x);
      tb[(kkB + 1u) * 33u + r4B] = vec4<f32>(q0.y, q1.y, q2.y, q3.y);
      tb[(kkB + 2u) * 33u + r4B] = vec4<f32>(q0.z, q1.z, q2.z, q3.z);
      tb[(kkB + 3u) * 33u + r4B] = vec4<f32>(q0.w, q1.w, q2.w, q3.w);
    }`
          : `for (var e = 0u; e < 2u; e++) {
      let slot = li + e * 256u;
      let kk = slot % 16u;
      let r4 = slot / 16u;
      let k = k0 + kk;
      var v = vec4<f32>(0.0);
      if (inTile && k < K) {
        let row0 = n0 + r4 * 4u;
        let bAddr0 = u.iOff.y + row0 * K + k;
        if (row0 + 3u < N) {
          v = vec4<f32>(b[bAddr0], b[bAddr0 + K], b[bAddr0 + 2u * K], b[bAddr0 + 3u * K]);
        } else {
          if (row0 < N) { v.x = b[bAddr0]; }
          if (row0 + 1u < N) { v.y = b[bAddr0 + K]; }
          if (row0 + 2u < N) { v.z = b[bAddr0 + 2u * K]; }
          if (row0 + 3u < N) { v.w = b[bAddr0 + 3u * K]; }
        }
      }
      tb[kk * 33u + r4] = v;
    }`;

    const fmaBody = Array.from(
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
    ).join("\n    ");

    const storeBody = [0, 1, 2, 3]
      .map(
        (i) => `{
    let m = m0 + tm + ${i}u;
    if (m < M) {
      let r0 = acc_${i}_0 * u.f0.x;
      let r1 = acc_${i}_1 * u.f0.x;
      let rowOut = u.iOff.w + m * N;
      ${[0, 1, 2, 3, 4, 5, 6, 7]
        .map((j) => {
          const nExpr = j < 4 ? `n0 + tn0 + ${j}u` : `n0 + tn1 + ${j - 4}u`;
          const valExpr = j < 4 ? `r0.${"xyzw"[j]}` : `r1.${"xyzw"[j - 4]}`;
          return `{ let n = ${nExpr}; if (n < N) { out[rowOut + n] = ${valExpr}${hasBias ? ` + ${readBias("n")}` : ""}; } }`;
        })
        .join("\n      ")}
    }
  }`,
      )
      .join("\n  ");

    const code = `${header}
var<workgroup> ta: array<vec4<f32>, 272>;
var<workgroup> tb: array<vec4<${EB}>, 528>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let M = u.dims.x; let N = u.dims.y; let K = u.dims.z; let sAm = u.dims.w;
  let tilesN = u.b0.x; let totalTiles = u.b0.y;
  let flat = wid.x + wid.y * 65535u;
  let inTile = flat < totalTiles;
  let tile = min(flat, totalTiles - 1u);
  let m0 = (tile / tilesN) * 64u;
  let n0 = (tile % tilesN) * 128u;
  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * 4u;
  let tn0 = tx * 4u;
  let tn1 = tn0 + 64u;
  var acc_0_0 = vec4<f32>(0.0); var acc_0_1 = vec4<f32>(0.0);
  var acc_1_0 = vec4<f32>(0.0); var acc_1_1 = vec4<f32>(0.0);
  var acc_2_0 = vec4<f32>(0.0); var acc_2_1 = vec4<f32>(0.0);
  var acc_3_0 = vec4<f32>(0.0); var acc_3_1 = vec4<f32>(0.0);
  let kkA = li % 16u;
  let r4A = li / 16u;
  let rowA0 = m0 + r4A * 4u;
  let aBase0 = u.iOff.x + rowA0 * sAm + kkA;
  let fullA = rowA0 + 3u < M;
  let taWrite = kkA * 17u + r4A;
  ${loadBSetup}
  for (var k0 = 0u; k0 < K; k0 += 16u) {
    {
      var v = vec4<f32>(0.0);
      if (inTile && k0 + kkA < K) {
        let aOff = aBase0 + k0;
        if (fullA) {
          v = vec4<f32>(a[aOff], a[aOff + sAm], a[aOff + 2u * sAm], a[aOff + 3u * sAm]);
        } else {
          if (rowA0 < M) { v.x = a[aOff]; }
          if (rowA0 + 1u < M) { v.y = a[aOff + sAm]; }
          if (rowA0 + 2u < M) { v.z = a[aOff + 2u * sAm]; }
          if (rowA0 + 3u < M) { v.w = a[aOff + 3u * sAm]; }
        }
      }
      ta[taWrite] = v;
    }
    ${loadBLoop}
    workgroupBarrier();
    ${fmaBody}
    workgroupBarrier();
  }
  if (!inTile) { return; }
  ${storeBody}
}`;
    return { code, bindings, dispatch, uniforms: u, uniformBinding };
  }

  // 32x64 register-tiled GEMM for small-M high SM occupancy (256 threads, 2x4 per thread, BK=16, <=6656B shared)
  const loadBSetupSmall =
    bEnc === "F16"
      ? `let kkB = (li % 8u) * 2u;
  let r4B = li / 8u;
  let rowB0 = n0 + r4B * 4u;
  let sBnHalf = K / 2u;
  let bBase0 = (rowB0 * K + kkB) / 2u;
  let fullB = rowB0 + 3u < N;
  let tbWrite0 = kkB * 17u + r4B;
  let tbWrite1 = (kkB + 1u) * 17u + r4B;`
      : bEnc === "I8"
        ? `let kkB = (li % 4u) * 4u;
  let r4B = li / 4u;
  let rowB0 = n0 + r4B * 4u;
  let sBnQuad = K / 4u;
  let bBase0 = (rowB0 * K + kkB) / 4u;
  let fullB = rowB0 + 3u < N;
  var scB = vec4<f32>(0.0);
  if (li < 64u) {
    if (rowB0 < N) { scB.x = bScale[rowB0]; }
    if (rowB0 + 1u < N) { scB.y = bScale[rowB0 + 1u]; }
    if (rowB0 + 2u < N) { scB.z = bScale[rowB0 + 2u]; }
    if (rowB0 + 3u < N) { scB.w = bScale[rowB0 + 3u]; }
  }`
        : `let kkB = li % 16u;
  let r4B = li / 16u;
  let rowB0 = n0 + r4B * 4u;
  let bBase0 = u.iOff.y + rowB0 * K + kkB;
  let fullB = rowB0 + 3u < N;
  let tbWrite = kkB * 17u + r4B;`;

  const loadBLoopSmall =
    bEnc === "F16"
      ? `if (li < 128u) {
      var lo = vec4<${EB}>(0.0);
      var hi = vec4<${EB}>(0.0);
      if (inTile && k0 + kkB < K) {
        let bOff = bBase0 + (k0 / 2u);
        if (fullB) {
          let p0 = unpack2x16float(b[bOff]);
          let p1 = unpack2x16float(b[bOff + sBnHalf]);
          let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]);
          let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]);
          lo = vec4<${EB}>(${EB}(p0.x), ${EB}(p1.x), ${EB}(p2.x), ${EB}(p3.x));
          hi = vec4<${EB}>(${EB}(p0.y), ${EB}(p1.y), ${EB}(p2.y), ${EB}(p3.y));
        } else {
          if (rowB0 < N) { let p0 = unpack2x16float(b[bOff]); lo.x = ${EB}(p0.x); hi.x = ${EB}(p0.y); }
          if (rowB0 + 1u < N) { let p1 = unpack2x16float(b[bOff + sBnHalf]); lo.y = ${EB}(p1.x); hi.y = ${EB}(p1.y); }
          if (rowB0 + 2u < N) { let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]); lo.z = ${EB}(p2.x); hi.z = ${EB}(p2.y); }
          if (rowB0 + 3u < N) { let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]); lo.w = ${EB}(p3.x); hi.w = ${EB}(p3.y); }
        }
      }
      tb[tbWrite0] = lo;
      tb[tbWrite1] = hi;
    }`
      : bEnc === "I8"
        ? `if (li < 64u) {
      var q0 = vec4<f32>(0.0); var q1 = vec4<f32>(0.0);
      var q2 = vec4<f32>(0.0); var q3 = vec4<f32>(0.0);
      if (inTile && k0 + kkB < K) {
        let bOff = bBase0 + (k0 / 4u);
        if (fullB) {
          q0 = unpack4i8(b[bOff]) * scB.x;
          q1 = unpack4i8(b[bOff + sBnQuad]) * scB.y;
          q2 = unpack4i8(b[bOff + 2u * sBnQuad]) * scB.z;
          q3 = unpack4i8(b[bOff + 3u * sBnQuad]) * scB.w;
        } else {
          if (rowB0 < N) { q0 = unpack4i8(b[bOff]) * scB.x; }
          if (rowB0 + 1u < N) { q1 = unpack4i8(b[bOff + sBnQuad]) * scB.y; }
          if (rowB0 + 2u < N) { q2 = unpack4i8(b[bOff + 2u * sBnQuad]) * scB.z; }
          if (rowB0 + 3u < N) { q3 = unpack4i8(b[bOff + 3u * sBnQuad]) * scB.w; }
        }
      }
      tb[(kkB + 0u) * 17u + r4B] = vec4<f32>(q0.x, q1.x, q2.x, q3.x);
      tb[(kkB + 1u) * 17u + r4B] = vec4<f32>(q0.y, q1.y, q2.y, q3.y);
      tb[(kkB + 2u) * 17u + r4B] = vec4<f32>(q0.z, q1.z, q2.z, q3.z);
      tb[(kkB + 3u) * 17u + r4B] = vec4<f32>(q0.w, q1.w, q2.w, q3.w);
    }`
        : `{
      var v = vec4<f32>(0.0);
      if (inTile && k0 + kkB < K) {
        let bOff = bBase0 + k0;
        if (fullB) {
          v = vec4<f32>(b[bOff], b[bOff + K], b[bOff + 2u * K], b[bOff + 3u * K]);
        } else {
          if (rowB0 < N) { v.x = b[bOff]; }
          if (rowB0 + 1u < N) { v.y = b[bOff + K]; }
          if (rowB0 + 2u < N) { v.z = b[bOff + 2u * K]; }
          if (rowB0 + 3u < N) { v.w = b[bOff + 3u * K]; }
        }
      }
      tb[tbWrite] = v;
    }`;

  const fmaSmall = Array.from(
    { length: 16 },
    (_, kk) => `{
      let av4 = ta[${kk * 9}u + r4Read];
      let av = select(av4.xy, av4.zw, hiPair);
      let bv = vec4<f32>(tb[${kk * 17}u + tx]);
      acc0 = fma(vec4<f32>(av.x), bv, acc0);
      acc1 = fma(vec4<f32>(av.y), bv, acc1);
    }`,
  ).join("\n    ");

  const storeSmall = [0, 1]
    .map(
      (i) => `{
    let m = m0 + tm + ${i}u;
    if (m < M) {
      let r = acc${i} * u.f0.x;
      let rowOut = u.iOff.w + m * N;
      ${[0, 1, 2, 3]
        .map(
          (j) =>
            `{ let n = n0 + tn0 + ${j}u; if (n < N) { out[rowOut + n] = r.${"xyzw"[j]}${hasBias ? ` + ${readBias("n")}` : ""}; } }`,
        )
        .join("\n      ")}
    }
  }`,
    )
    .join("\n  ");

  const code = `${header}
var<workgroup> ta: array<vec4<f32>, 144>;
var<workgroup> tb: array<vec4<${EB}>, 272>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let M = u.dims.x; let N = u.dims.y; let K = u.dims.z; let sAm = u.dims.w;
  let tilesN = u.b0.x; let totalTiles = u.b0.y;
  let flat = wid.x + wid.y * 65535u;
  let inTile = flat < totalTiles;
  let tile = min(flat, totalTiles - 1u);
  let m0 = (tile / tilesN) * 32u;
  let n0 = (tile % tilesN) * 64u;
  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * 2u;
  let tn0 = tx * 4u;
  let r4Read = tm4 / 2u;
  let hiPair = (tm4 & 1u) != 0u;
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  let kkA = li % 16u;
  let r4A = li / 16u;
  let rowA0 = m0 + r4A * 4u;
  let aBase0 = u.iOff.x + rowA0 * sAm + kkA;
  let fullA = rowA0 + 3u < M;
  let taWrite = kkA * 9u + r4A;
  ${loadBSetupSmall}
  for (var k0 = 0u; k0 < K; k0 += 16u) {
    if (li < 128u) {
      var v = vec4<f32>(0.0);
      if (inTile && k0 + kkA < K) {
        let aOff = aBase0 + k0;
        if (fullA) {
          v = vec4<f32>(a[aOff], a[aOff + sAm], a[aOff + 2u * sAm], a[aOff + 3u * sAm]);
        } else {
          if (rowA0 < M) { v.x = a[aOff]; }
          if (rowA0 + 1u < M) { v.y = a[aOff + sAm]; }
          if (rowA0 + 2u < M) { v.z = a[aOff + 2u * sAm]; }
          if (rowA0 + 3u < M) { v.w = a[aOff + 3u * sAm]; }
        }
      }
      ta[taWrite] = v;
    }
    ${loadBLoopSmall}
    workgroupBarrier();
    ${fmaSmall}
    workgroupBarrier();
  }
  if (!inTile) { return; }
  ${storeSmall}
}`;
  return { code, bindings, dispatch, uniforms: u, uniformBinding };
}

export function shader(kind, inputs, outputs, attrs = {}) {
  const [a, b, c] = inputs,
    out = outputs[0],
    shape = out.shape,
    count = sizeOf(shape);

  if (kind === "matmul") {
    const M = a.shape.at(-2),
      K = a.shape.at(-1),
      N = b.shape.at(-1),
      batch = count / (M * N),
      alpha = attrs.alpha ?? 1,
      beta = attrs.beta ?? 1;
    const aUnitK = K === 1 || a.stride.at(-1) === 1;
    const bTransposedRowMajor =
      (K === 1 || b.stride.at(-2) === 1) && (N === 1 || b.stride.at(-1) === K);
    const bEnc = b.storage.encoding ?? "F32";
    const bAlignOk =
      bEnc === "F32" ||
      (bEnc === "F16" && K % 2 === 0 && b.offset === 0) ||
      (bEnc === "I8" && K % 4 === 0 && b.offset === 0 && b.storage.columns === K);
    const bias1D =
      !c ||
      (c.shape.length === 1 && c.shape[0] === N && c.stride[0] === 1) ||
      (c.stride.at(-1) === 1 && c.stride.slice(0, -1).every((s) => s === 0));
    if (
      batch === 1 &&
      !a.storage.encoding &&
      aUnitK &&
      bTransposedRowMajor &&
      bAlignOk &&
      bias1D &&
      a.storage !== b.storage
    ) {
      const sAm = M > 1 ? a.stride.at(-2) : K;
      return buildLinearGemm({
        a,
        b,
        c,
        out,
        M,
        N,
        K,
        sAm,
        alpha,
        beta,
        hasF16: attrs.hasF16,
      });
    }
  }

  const bindings = [],
    map = new Map();
  function bind(t, write = false) {
    if (!t || typeof t === "number") return;
    let bnd = map.get(t.storage);
    if (!bnd) {
      bnd = { storage: t.storage, name: "b" + bindings.length, write };
      bindings.push(bnd);
      map.set(t.storage, bnd);
    }
    bnd.write ||= write;
    return bnd.name;
  }
  inputs.forEach((t) => bind(t));
  outputs.forEach((t) => bind(t, true));

  const u = new Uint32Array(24);
  const f = new Float32Array(u.buffer);

  const rawElem = (t, index) => {
    const name = bind(t);
    if (t.storage.encoding === "F16")
      return `unpack2x16float(${name}[(${index})/2u])[(${index})%2u]`;
    if (t.storage.encoding === "I8") {
      const scale = bind({ storage: t.storage.scale });
      return `(f32(i32((${name}[(${index})/4u]>>(((${index})%4u)*8u))<<24u)>>24u)*${scale}[(${index})/${ui(t.storage.columns)}])`;
    }
    return `${name}[${index}]`;
  };

  const exprWithOff = (t, i, offExpr) => {
    if (contiguous(t)) {
      return offExpr ? `(${i}+${offExpr})` : `(${i})`;
    }
    const zeroOff = tensor(t.storage, t.shape, t.stride, 0);
    const base = indexExpression(zeroOff, i, true);
    return offExpr ? `(${base}+${offExpr})` : base;
  };

  const getSlot = (t, slot, i = "i") => {
    if (typeof t === "number") {
      if (Number.isFinite(t) && slot < 3) {
        f[16 + slot] = t;
        return `u.f0.${"xyz"[slot]}`;
      }
      return fl(t);
    }
    if (slot >= 0 && slot < 3) {
      u[slot] = t.offset;
      return rawElem(t, exprWithOff(t, i, `u.iOff.${"xyz"[slot]}`));
    }
    return rawElem(t, indexExpression(t, i, true));
  };

  const get = (t, i = "i") => {
    if (typeof t === "number") return fl(t);
    return rawElem(t, indexExpression(t, i, true));
  };

  const putOut0 = (i, v) => {
    u[3] = out.offset;
    const idx = exprWithOff(out, i, "u.iOff.w");
    return `${bind(out, true)}[${idx}]=${v};`;
  };
  const put = (t, i, v) =>
    t === out ? putOut0(i, v) : `${bind(t, true)}[${indexExpression(t, i, true)}]=${v};`;

  let body = "",
    extra = "",
    workgroup = "64",
    dispatch = [1, 1, 1],
    invocations = count,
    customMain = false;
  const loop = (name, n, code) =>
    `for(var ${name}=0u;${name}<${ui(n)};${name}++){${code}}`;

  switch (kind) {
    case "onehot":
      body = putOut0(
        "i",
        `select(0.0,1.0,u32(${getSlot(a, 0, `(i/${ui(attrs.width)})`)})==i%${ui(attrs.width)})`,
      );
      break;
    case "window": {
      const d = shape[2],
        n = a.shape[1];
      body =
        `let win=i/${ui(128 * d)};let key=(i/${ui(d)})%128u;let j=i32(win*32u+key)-48;var value=0.0;if(j>=0&&j<${n}){value=${getSlot(a, 0, `(u32(j)*${ui(d)}+i%${ui(d)})`)};}` +
        putOut0("i", "value");
      break;
    }
    case "fill":
      if (Number.isFinite(attrs.value)) {
        f[20] = attrs.value;
        body = putOut0("i", "u.f1.x");
      } else {
        body = putOut0("i", fl(attrs.value));
      }
      break;
    case "range":
      body = putOut0("i", `${fl(attrs.start)}+f32(i)*${fl(attrs.step)}`);
      break;
    case "eye":
      body = putOut0(
        "i",
        `select(0.0,1.0,i/${ui(attrs.width)}==i%${ui(attrs.width)})`,
      );
      break;
    case "copy":
    case "cast": {
      let value = getSlot(broadcast(a, shape), 0);
      if (kind === "cast") {
        if (attrs.dtype === "torch.bool")
          value = `select(0.0,1.0,${value}!=0.0)`;
        else if (/torch.int/.test(attrs.dtype)) value = `trunc(${value})`;
      }
      body = putOut0("i", value);
      break;
    }
    case "pointwise": {
      const vals = inputs.map((t, idx) =>
          t === null
            ? null
            : getSlot(typeof t === "number" ? t : broadcast(t, shape), idx),
        ),
        [x, y, z] = vals;
      f[19] = attrs.alpha ?? 1;
      let v;
      switch (attrs.kind) {
        case "add":
          v = `${x}+${y}*u.f0.w`;
          break;
        case "sub":
          v = `${x}-${y}*u.f0.w`;
          break;
        case "rsub":
          v = `${y}-${x}*u.f0.w`;
          break;
        case "mul":
          v = `${x}*${y}`;
          break;
        case "div":
          v = `${x}/${y}`;
          break;
        case "pow":
          v = inputs[1] === 2 ? `${x}*${x}` : `pow(${x},${y})`;
          break;
        case "sigmoid":
          v = `1.0/(1.0+exp(-${x}))`;
          break;
        case "silu":
          v = `${x}/(1.0+exp(-${x}))`;
          break;
        case "relu":
          v = `max(0.0,${x})`;
          break;
        case "sqrt":
        case "exp":
        case "log":
        case "cos":
        case "abs":
          v = `${attrs.kind}(${x})`;
          break;
        case "reciprocal":
          v = `1.0/${x}`;
          break;
        case "neg":
          v = `-${x}`;
          break;
        case "eq":
        case "ne":
        case "lt":
        case "gt":
        case "le":
          v = `select(0.0,1.0,${x}${{ eq: "==", ne: "!=", lt: "<", gt: ">", le: "<=" }[attrs.kind]}${y})`;
          break;
        case "bitwise_and":
          v = `f32(i32(${x})&i32(${y}))`;
          break;
        case "bitwise_not":
          v = `select(0.0,1.0,${x}==0.0)`;
          break;
        case "where":
          v = `select(${z},${y},${x}!=0.0)`;
          break;
        case "clamp":
          v = `min(${z ?? fl(Infinity)},max(${y ?? fl(-Infinity)},${x}))`;
          break;
        default:
          throw Error(`Unsupported pointwise ${attrs.kind}`);
      }
      body = putOut0("i", v);
      break;
    }
    case "matmul": {
      // Register-tiled 32x32 batched/general-stride GEMM (2x2 per thread, 2D-folded < 65535)
      const M = a.shape.at(-2),
        K = a.shape.at(-1),
        N = b.shape.at(-1),
        batch = count / (M * N),
        bias = c ? broadcast(c, shape) : null,
        tilesM = Math.ceil(M / 32),
        tilesN = Math.ceil(N / 32),
        totalWG = batch * tilesM * tilesN;
      customMain = true;
      workgroup = "16,16";
      dispatch = fold2(totalWG);
      u[4] = M;
      u[5] = N;
      u[6] = K;
      u[7] = batch;
      u[8] = tilesN;
      u[9] = tilesM * tilesN;
      f[16] = attrs.alpha ?? 1;
      f[17] = attrs.beta ?? 1;

      const bDims = a.shape.slice(0, -2);
      const fastStrides =
        bDims.length <= 3 && !a.storage.encoding && !b.storage.encoding;
      if (fastStrides) {
        const pad3 = (arr, fill) => {
          const r = [...arr];
          while (r.length < 3) r.unshift(fill);
          return r;
        };
        const [B0, B1, B2] = pad3(bDims, 1);
        const [sA0, sA1, sA2] = pad3(a.stride.slice(0, -2), 0);
        const [sB0, sB1, sB2] = pad3(b.stride.slice(0, -2), 0);
        const sAm = M > 1 ? a.stride.at(-2) : 0;
        const sAk = K > 1 ? a.stride.at(-1) : 0;
        const sBk = K > 1 ? b.stride.at(-2) : 0;
        const sBn = N > 1 ? b.stride.at(-1) : 0;
        const transposeLoadB = sBk === 1 && sBn !== 1;
        u[0] = a.offset;
        u[1] = b.offset;
        u[3] = out.offset;
        u[10] = B1;
        u[11] = B2;
        u[12] = sA0;
        u[13] = sA1;
        u[14] = sA2;
        u[15] = sAm;
        f[20] = 0; // unused
        // Store sAk, sB0, sB1, sB2, sBk, sBn in remaining uniform slots or constants?
        // Wait: b0/b1 have 8 u32s (u[8..15]); let's inline sAk/sBk===1 specialization & pass strides cleanly:
        const aName = bind(a),
          bName = bind(b);
        extra = "var<workgroup> tileA:array<f32,544>;var<workgroup> tileB:array<f32,528>;";
        body = `let M=u.dims.x;let N=u.dims.y;let K=u.dims.z;let batch=u.dims.w;
          let tilesN=u.b0.x;let perBatch=u.b0.y;
          let flat=wg.x+wg.y*65535u;let h=flat/perBatch;let inBatch=h<batch;let hc=min(h,batch-1u);
          let rem=flat%perBatch;let m0=(rem/tilesN)*32u;let n0=(rem%tilesN)*32u;
          let bi2=hc%${ui(B2)};let ht=hc/${ui(B2)};let bi1=ht%${ui(B1)};let bi0=ht/${ui(B1)};
          let baseA=u.iOff.x+bi0*${ui(sA0)}+bi1*${ui(sA1)}+bi2*${ui(sA2)};
          let baseB=u.iOff.y+bi0*${ui(sB0)}+bi1*${ui(sB1)}+bi2*${ui(sB2)};
          let lx=lid.x;let ly=lid.y;
          var acc00=0.0;var acc01=0.0;var acc10=0.0;var acc11=0.0;
          for(var k0=0u;k0<K;k0+=16u){
            let r0=m0+ly;let r1=r0+16u;let kA=k0+lx;
            tileA[ly*17u+lx]=select(0.0,${aName}[baseA+min(r0,M-1u)*${ui(sAm)}+min(kA,K-1u)*${ui(sAk)}],inBatch&&r0<M&&kA<K);
            tileA[(ly+16u)*17u+lx]=select(0.0,${aName}[baseA+min(r1,M-1u)*${ui(sAm)}+min(kA,K-1u)*${ui(sAk)}],inBatch&&r1<M&&kA<K);
            ${
              transposeLoadB
                ? `let c0=n0+ly;let c1=c0+16u;let kB=k0+lx;
            tileB[lx*33u+ly]=select(0.0,${bName}[baseB+min(kB,K-1u)+min(c0,N-1u)*${ui(sBn)}],inBatch&&c0<N&&kB<K);
            tileB[lx*33u+ly+16u]=select(0.0,${bName}[baseB+min(kB,K-1u)+min(c1,N-1u)*${ui(sBn)}],inBatch&&c1<N&&kB<K);`
                : `let c0=n0+lx;let c1=c0+16u;let kB=k0+ly;
            tileB[ly*33u+lx]=select(0.0,${bName}[baseB+min(kB,K-1u)*${ui(sBk)}+min(c0,N-1u)*${ui(sBn)}],inBatch&&c0<N&&kB<K);
            tileB[ly*33u+lx+16u]=select(0.0,${bName}[baseB+min(kB,K-1u)*${ui(sBk)}+min(c1,N-1u)*${ui(sBn)}],inBatch&&c1<N&&kB<K);`
            }
            workgroupBarrier();
            for(var kk=0u;kk<16u;kk++){
              let a0=tileA[(ly*2u)*17u+kk];let a1=tileA[(ly*2u+1u)*17u+kk];
              let bv0=tileB[kk*33u+(lx*2u)];let bv1=tileB[kk*33u+(lx*2u+1u)];
              acc00=fma(a0,bv0,acc00);acc01=fma(a0,bv1,acc01);
              acc10=fma(a1,bv0,acc10);acc11=fma(a1,bv1,acc11);
            }
            workgroupBarrier();
          }
          if(!inBatch){return;}
          ${[
            [0, 0, "acc00"],
            [0, 1, "acc01"],
            [1, 0, "acc10"],
            [1, 1, "acc11"],
          ]
            .map(
              ([ri, ci, acc]) =>
                `{let m=m0+ly*2u+${ri}u;let n=n0+lx*2u+${ci}u;if(m<M&&n<N){let i=(hc*M+m)*N+n;${putOut0("i", `${acc}*u.f0.x${bias ? "+" + getSlot(bias, 2, "i") + "*u.f0.y" : ""}`)}}}`,
            )
            .join("")}`;
      } else {
        extra = "var<workgroup> tileA:array<f32,544>;var<workgroup> tileB:array<f32,528>;";
        body = `let M=u.dims.x;let N=u.dims.y;let K=u.dims.z;let batch=u.dims.w;
          let tilesN=u.b0.x;let perBatch=u.b0.y;
          let flat=wg.x+wg.y*65535u;let h=flat/perBatch;let inBatch=h<batch;let hc=min(h,batch-1u);
          let rem=flat%perBatch;let m0=(rem/tilesN)*32u;let n0=(rem%tilesN)*32u;
          let lx=lid.x;let ly=lid.y;
          var acc00=0.0;var acc01=0.0;var acc10=0.0;var acc11=0.0;
          for(var k0=0u;k0<K;k0+=16u){
            let r0=m0+ly;let r1=r0+16u;let kA=k0+lx;
            var av0=0.0;var av1=0.0;
            if(inBatch&&r0<M&&kA<K){av0=${getSlot(a, 0, `((hc*M+r0)*K+kA)`)};}
            if(inBatch&&r1<M&&kA<K){av1=${getSlot(a, 0, `((hc*M+r1)*K+kA)`)};}
            tileA[ly*17u+lx]=av0;tileA[(ly+16u)*17u+lx]=av1;
            let c0=n0+lx;let c1=c0+16u;let kB=k0+ly;
            var bv0=0.0;var bv1=0.0;
            if(inBatch&&c0<N&&kB<K){bv0=${getSlot(b, 1, `((hc*K+kB)*N+c0)`)};}
            if(inBatch&&c1<N&&kB<K){bv1=${getSlot(b, 1, `((hc*K+kB)*N+c1)`)};}
            tileB[ly*33u+lx]=bv0;tileB[ly*33u+lx+16u]=bv1;
            workgroupBarrier();
            for(var kk=0u;kk<16u;kk++){
              let a0=tileA[(ly*2u)*17u+kk];let a1=tileA[(ly*2u+1u)*17u+kk];
              let tB0=tileB[kk*33u+(lx*2u)];let tB1=tileB[kk*33u+(lx*2u+1u)];
              acc00=fma(a0,tB0,acc00);acc01=fma(a0,tB1,acc01);
              acc10=fma(a1,tB0,acc10);acc11=fma(a1,tB1,acc11);
            }
            workgroupBarrier();
          }
          if(!inBatch){return;}
          ${[
            [0, 0, "acc00"],
            [0, 1, "acc01"],
            [1, 0, "acc10"],
            [1, 1, "acc11"],
          ]
            .map(
              ([ri, ci, acc]) =>
                `{let m=m0+ly*2u+${ri}u;let n=n0+lx*2u+${ci}u;if(m<M&&n<N){let i=(hc*M+m)*N+n;${putOut0("i", `${acc}*u.f0.x${bias ? "+" + getSlot(bias, 2, "i") + "*u.f0.y" : ""}`)}}}`,
            )
            .join("")}`;
      }
      break;
    }
    case "norm": {
      const w = attrs.width,
        rows = count / w;
      customMain = true;
      workgroup = "64";
      dispatch = fold2(rows);
      u[4] = rows;
      u[5] = w;
      f[21] = attrs.epsilon;
      extra = "var<workgroup> red:array<f32,64>;";
      body = `let rows=u.dims.x;let w=u.dims.y;let r=wg.x+wg.y*65535u;let live=r<rows;let row=min(r,rows-1u);let li=lid.x;
        var s=0.0;for(var j=li;j<w;j+=64u){s+=${getSlot(a, 0, "(row*w+j)")};}
        red[li]=s;workgroupBarrier();
        if(li<32u){red[li]+=red[li+32u];}workgroupBarrier();
        if(li<16u){red[li]+=red[li+16u];}workgroupBarrier();
        if(li<8u){red[li]+=red[li+8u];}workgroupBarrier();
        if(li==0u){red[0]=((red[0]+red[4])+(red[2]+red[6]))+((red[1]+red[5])+(red[3]+red[7]));}
        workgroupBarrier();let mean=red[0]/f32(w);workgroupBarrier();
        var q=0.0;for(var j=li;j<w;j+=64u){let d=${getSlot(a, 0, "(row*w+j)")}-mean;q+=d*d;}
        red[li]=q;workgroupBarrier();
        if(li<32u){red[li]+=red[li+32u];}workgroupBarrier();
        if(li<16u){red[li]+=red[li+16u];}workgroupBarrier();
        if(li<8u){red[li]+=red[li+8u];}workgroupBarrier();
        if(li==0u){red[0]=((red[0]+red[4])+(red[2]+red[6]))+((red[1]+red[5])+(red[3]+red[7]));}
        workgroupBarrier();let inv=inverseSqrt(red[0]/f32(w)+u.f1.y);
        if(!live){return;}
        ${outputs.length > 1 ? `if(li==0u){${put(outputs[1], "row", "mean")}${put(outputs[2], "row", "inv")}}` : ""}
        for(var j=li;j<w;j+=64u){let idx=row*w+j;${putOut0("idx", `(${getSlot(a, 0, "idx")}-mean)*inv*${b ? getSlot(b, 1, "j") : "1.0"}+${c ? getSlot(c, 2, "j") : "0.0"}`)}}`;
      break;
    }
    case "softmax": {
      const d = attrs.dim,
        w = shape[d],
        inner = sizeOf(shape.slice(d + 1));
      if (inner === 1) {
        const rows = count / w;
        customMain = true;
        workgroup = "64";
        dispatch = fold2(rows);
        u[4] = rows;
        u[5] = w;
        extra = "var<workgroup> red:array<f32,64>;";
        body = `let rows=u.dims.x;let w=u.dims.y;let r=wg.x+wg.y*65535u;let live=r<rows;let row=min(r,rows-1u);let li=lid.x;
          var mx=${fl(-Infinity)};
          for(var j=li;j<w;j+=64u){mx=max(mx,${getSlot(a, 0, "(row*w+j)")});}
          red[li]=mx;workgroupBarrier();
          if(li<32u){red[li]=max(red[li],red[li+32u]);}workgroupBarrier();
          if(li<16u){red[li]=max(red[li],red[li+16u]);}workgroupBarrier();
          if(li<8u){red[li]=max(red[li],red[li+8u]);}workgroupBarrier();
          if(li==0u){red[0]=max(max(max(red[0],red[4]),max(red[2],red[6])),max(max(red[1],red[5]),max(red[3],red[7])));}
          workgroupBarrier();let top=red[0];workgroupBarrier();
          var s=0.0;
          if(top>${fl(-Infinity)}){for(var j=li;j<w;j+=64u){s+=exp(${getSlot(a, 0, "(row*w+j)")}-top);}}
          red[li]=s;workgroupBarrier();
          if(li<32u){red[li]+=red[li+32u];}workgroupBarrier();
          if(li<16u){red[li]+=red[li+16u];}workgroupBarrier();
          if(li<8u){red[li]+=red[li+8u];}workgroupBarrier();
          if(li==0u){red[0]=((red[0]+red[4])+(red[2]+red[6]))+((red[1]+red[5])+(red[3]+red[7]));}
          workgroupBarrier();let sum=red[0];
          if(!live){return;}
          for(var j=li;j<w;j+=64u){let idx=row*w+j;${putOut0("idx", `select(exp(${getSlot(a, 0, "idx")}-top)/sum,0.0,top==${fl(-Infinity)})`)}}`;
      } else {
        invocations = count / w;
        body = `let base=(i/${ui(inner)})*${ui(w * inner)}+i%${ui(inner)};var mx=${fl(-Infinity)};
          ${loop("j", w, `mx=max(mx,${getSlot(a, 0, `(base+j*${ui(inner)})`)});`)}
          var sum=0.0;${loop("j", w, `sum+=exp(${getSlot(a, 0, `(base+j*${ui(inner)})`)}-mx);`)}
          ${loop("j", w, putOut0(`(base+j*${ui(inner)})`, `select(exp(${getSlot(a, 0, `(base+j*${ui(inner)})`)}-mx)/sum,0.0,mx==${fl(-Infinity)})`))}`;
      }
      break;
    }
    case "reduce": {
      const dims = new Set(attrs.dims),
        rv = tensor(
          a.storage,
          a.shape.filter((_, i) => dims.has(i)),
          a.stride.filter((_, i) => dims.has(i)),
          0,
        ),
        nv = tensor(
          a.storage,
          a.shape.filter((_, i) => !dims.has(i)),
          a.stride.filter((_, i) => !dims.has(i)),
          0,
        ),
        rn = sizeOf(rv.shape);
      u[0] = a.offset;
      if (
        (attrs.kind === "sum" || attrs.kind === "linalg_vector_norm") &&
        count <= 512 &&
        rn >= 32
      ) {
        customMain = true;
        workgroup = "64";
        dispatch = fold2(count);
        u[4] = count;
        u[5] = rn;
        extra = "var<workgroup> red:array<f32,64>;";
        const val = `${bind(a)}[u.iOff.x+${indexExpression(nv, "row", true)}+${indexExpression(rv, "j", true)}]`;
        const term = attrs.kind === "linalg_vector_norm" ? `(${val})*(${val})` : val;
        body = `let rows=u.dims.x;let rn=u.dims.y;let r=wg.x+wg.y*65535u;let live=r<rows;let row=min(r,rows-1u);let li=lid.x;
          var s=0.0;for(var j=li;j<rn;j+=64u){s+=${term};}
          red[li]=s;workgroupBarrier();
          if(li<32u){red[li]+=red[li+32u];}workgroupBarrier();
          if(li<16u){red[li]+=red[li+16u];}workgroupBarrier();
          if(li<8u){red[li]+=red[li+8u];}workgroupBarrier();
          if(li==0u&&live){let total=((red[0]+red[4])+(red[2]+red[6]))+((red[1]+red[5])+(red[3]+red[7]));${putOut0("row", attrs.kind === "linalg_vector_norm" ? "sqrt(total)" : "total")}}`;
      } else {
        const value = `${bind(a)}[u.iOff.x+${indexExpression(nv, "i", true)}+${indexExpression(rv, "j", true)}]`;
        const init =
          attrs.kind === "min"
            ? fl(Infinity)
            : attrs.kind === "max"
              ? fl(-Infinity)
              : "0.0";
        const update =
          attrs.kind === "min"
            ? `acc=min(acc,${value});`
            : attrs.kind === "max"
              ? `acc=max(acc,${value});`
              : attrs.kind === "any"
                ? `acc=select(acc,1.0,${value}!=0.0);`
                : `acc+=${attrs.kind === "linalg_vector_norm" ? `${value}*${value}` : value};`;
        body = `var acc=${init};${loop("j", rn, update)}${putOut0("i", attrs.kind === "linalg_vector_norm" ? "sqrt(acc)" : "acc")}`;
      }
      break;
    }
    case "cat":
    case "stack": {
      const d = attrs.dim,
        inner = sizeOf(shape.slice(d + 1)),
        w = shape[d];
      let offset = 0;
      body = `let col=(i/${ui(inner)})%${ui(w)};`;
      for (const t of inputs) {
        const tw = kind === "stack" ? 1 : t.shape[d];
        body += `if(col>=${ui(offset)}&&col<${ui(offset + tw)}){${putOut0("i", get(t, `((i/${ui(w * inner)})*${ui(tw * inner)}+(col-${ui(offset)})*${ui(inner)}+i%${ui(inner)})`))}}`;
        offset += tw;
      }
      break;
    }
    case "embedding": {
      const w = a.shape[1];
      body = putOut0(
        "i",
        getSlot(a, 0, `(u32(${getSlot(b, 1, `(i/${ui(w)})`)})*${ui(w)}+i%${ui(w)})`),
      );
      break;
    }
    case "scatter": {
      invocations = sizeOf(b.shape);
      const bs = strides(b.shape),
        os = strides(shape);
      const idx = shape
        .map((n, d) =>
          d === attrs.dim
            ? `u32(${getSlot(b, 1)})*${ui(os[d])}`
            : `((i/${ui(bs[d])})%${ui(b.shape[d])})*${ui(os[d])}`,
        )
        .join("+");
      body = putOut0(`(${idx})`, fl(attrs.value));
      break;
    }
    case "repeat": {
      const os = strides(shape),
        is = strides(a.shape),
        shift = shape.length - a.shape.length;
      const idx =
        a.shape
          .map((n, d) => `((i/${ui(os[d + shift])})%${ui(n)})*${ui(is[d])}`)
          .join("+") || "0u";
      body = putOut0("i", getSlot(a, 0, `(${idx})`));
      break;
    }
    case "pad": {
      const os = strides(shape),
        is = strides(a.shape),
        p = attrs.pad;
      let valid = [],
        idx = [];
      for (let d = 0; d < shape.length; d++) {
        const before = p[2 * (shape.length - 1 - d)] ?? 0;
        body += `let q${d}=i32((i/${ui(os[d])})%${ui(shape[d])})-${before};`;
        valid.push(`q${d}>=0&&q${d}<${a.shape[d]}`);
        idx.push(`u32(q${d})*${ui(is[d])}`);
      }
      body += `var v=${fl(attrs.value)};if(${valid.join("&&")}){v=${getSlot(a, 0, `(${idx.join("+")})`)};}${putOut0("i", "v")}`;
      break;
    }
    case "cdist": {
      const m = a.shape.at(-2),
        n = b.shape.at(-2),
        k = a.shape.at(-1);
      body = `let h=i/${ui(m * n)};let r=(i/${ui(n)})%${ui(m)};let col=i%${ui(n)};var sum=0.0;
        ${loop("j", k, `let delta=${getSlot(a, 0, `((h*${ui(m)}+r)*${ui(k)}+j)`)}-${getSlot(b, 1, `((h*${ui(n)}+col)*${ui(k)}+j)`)};sum+=delta*delta;`)}${putOut0("i", "sqrt(sum)")}`;
      break;
    }
    case "svd": {
      invocations = sizeOf(a.shape) / 9;
      body = `var mat:array<f32,9>;var v:array<f32,9>;
        ${loop("j", 9, `mat[j]=${getSlot(a, 0, "(i*9u+j)")};v[j]=select(0.0,1.0,j/3u==j%3u);`)}
        for(var sweep=0u;sweep<16u;sweep++){for(var p=0u;p<2u;p++){for(var q=p+1u;q<3u;q++){
          var aa=0.0;var bb=0.0;var ab=0.0;
          for(var j=0u;j<3u;j++){aa+=mat[j*3u+p]*mat[j*3u+p];bb+=mat[j*3u+q]*mat[j*3u+q];ab+=mat[j*3u+p]*mat[j*3u+q];}
          if(abs(ab)<1e-7*sqrt(aa*bb)){continue;}
          let tau=(bb-aa)/(2.0*ab);let t=select(-1.0,1.0,tau>=0.0)/(abs(tau)+sqrt(1.0+tau*tau));let co=inverseSqrt(1.0+t*t);let sn=t*co;
          for(var j=0u;j<3u;j++){let x=mat[j*3u+p];let y=mat[j*3u+q];mat[j*3u+p]=co*x-sn*y;mat[j*3u+q]=sn*x+co*y;
            let vx=v[j*3u+p];let vy=v[j*3u+q];v[j*3u+p]=co*vx-sn*vy;v[j*3u+q]=sn*vx+co*vy;}
        }}}
        var norms:array<f32,3>;var order:array<u32,3>;
        for(var j=0u;j<3u;j++){norms[j]=sqrt(mat[j]*mat[j]+mat[3u+j]*mat[3u+j]+mat[6u+j]*mat[6u+j]);order[j]=j;}
        for(var p=0u;p<2u;p++){for(var q=p+1u;q<3u;q++){if(norms[order[q]]>norms[order[p]]){let tmp=order[p];order[p]=order[q];order[q]=tmp;}}}
        for(var j=0u;j<3u;j++){${put(outputs[1], "(i*3u+j)", "norms[order[j]]")}
          for(var k=0u;k<3u;k++){${putOut0("(i*9u+k*3u+j)", "mat[k*3u+order[j]]/max(norms[order[j]],1e-30)")}${put(outputs[2], "(i*9u+j*3u+k)", "v[k*3u+order[j]]")}}}`;
      break;
    }
    case "det": {
      body = `var mat:array<f32,9>;${loop("j", 9, `mat[j]=${getSlot(a, 0, "(i*9u+j)")};`)}${putOut0("i", "mat[0]*(mat[4]*mat[8]-mat[5]*mat[7])-mat[1]*(mat[3]*mat[8]-mat[5]*mat[6])+mat[2]*(mat[3]*mat[7]-mat[4]*mat[6])")}`;
      break;
    }
    default:
      throw Error(`Unsupported shader ${kind}`);
  }

  if (!customMain) {
    u[4] = invocations;
    dispatch = [
      Math.min(MAX_GRID, Math.ceil(invocations / 64)),
      Math.ceil(invocations / (64 * MAX_GRID)),
      1,
    ];
    body = `let i=gid.x+gid.y*4194240u;if(i>=u.dims.x){return;}` + body;
  }

  const used = bindings.filter((bnd) =>
    new RegExp("\\b" + bnd.name + "\\[").test(body),
  );
  const renumbered = used.map((bnd, binding) => ({ ...bnd, binding }));
  // Rename b_old -> b_old remains unique variable name; binding index is compact 0..used.length-1
  const uniformBinding = renumbered.length;
  const header = [
    UNIFORM_STRUCT,
    ...renumbered.map(
      (bnd) =>
        `@group(0) @binding(${bnd.binding}) var<storage,${bnd.write ? "read_write" : "read"}> ${bnd.name}:array<${bnd.storage.encoding ? "u32" : "f32"}>;`,
    ),
    `@group(0) @binding(${uniformBinding}) var<uniform> u: U;`,
  ].join("\n");

  return {
    code: `${header}\n${extra}\n@compute @workgroup_size(${workgroup}) fn main(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wg:vec3<u32>){${body}}`,
    bindings: renumbered,
    dispatch,
    uniforms: u,
    uniformBinding,
  };
}
