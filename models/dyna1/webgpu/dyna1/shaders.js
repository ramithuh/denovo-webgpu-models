/** WGSL generators: FP32 accumulation for every storage precision. */
const decl = (i, name, type = 'f32', rw = false) =>
  `@group(0) @binding(${i}) var<storage,${rw ? 'read_write' : 'read'}> ${name}: array<${type}>;\n`;

function weightHelper(dtype, native) {
  const get =
    dtype === 'F32'
      ? 'return w[i];'
      : dtype === 'F16'
        ? native
          ? 'return f32(w[i]);'
          : 'let pair=unpack2x16float(w[i/2u]);return select(pair.x,pair.y,(i&1u)==1u);'
        : 'let word=w[i/4u];let sh=(i&3u)*8u;let sb=bitcast<i32>(((word>>sh)&255u)<<24u)>>24u;return f32(sb)*scale[row];';
  return `fn weight(i:u32,row:u32)->f32 {${get}}`;
}

export function matmulGrid(M, N) {
  if (N === 1) return M <= 65535 ? [Math.max(1, M)] : [65535, Math.ceil(M / 65535)];
  const groups =
    M <= 160
      ? Math.ceil(M / 32) * Math.ceil(N / 64)
      : Math.ceil(M / 64) * Math.ceil(N / 128);
  return groups <= 65535 ? [Math.max(1, groups)] : [65535, Math.ceil(groups / 65535)];
}

export function matmulSource(M, N, K, dtype = 'F32', hasBias = false, useF16 = false) {
  const native = dtype === 'F16' && useF16;
  const wType = dtype === 'F32' ? 'f32' : native ? 'f16' : 'u32';
  const EB = native ? 'f16' : 'f32';

  // Fast 1-output-column GEMV (e.g. head.output_layer D -> 1)
  if (N === 1) {
    return `${native ? 'enable f16;\n' : ''}${decl(0, 'a')}${decl(1, 'w', wType)}${decl(2, 'bias')}${decl(3, 'scale')}${decl(4, 'y', 'f32', true)}
${weightHelper(dtype, native)}
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let rows = arrayLength(&a) / ${K}u;
  let r = wid.x + wid.y * 65535u;
  let live = r < rows;
  let row = min(r, rows - 1u);
  let base = row * ${K}u;
  var s = 0.0;
  for (var k = li; k < ${K}u; k += 64u) {
    s = fma(a[base + k], weight(k, 0u), s);
  }
  red[li] = s; workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (live && li == 0u) {
    let tot = ((red[0] + red[4]) + (red[2] + red[6])) + ((red[1] + red[5]) + (red[3] + red[7]));
    y[row] = tot${hasBias ? ' + bias[0]' : ''};
  }
}`;
  }

  // 32x64 register-tiled GEMM (2x4 per thread, 256 threads) for M <= 160
  if (M <= 160) {
    const unroll = Array.from(
      { length: 16 },
      (_, kk) => `{
      let av = ta[${kk * 17}u + tm2];
      let bv = vec4<f32>(tb[${kk * 17}u + tx]);
      acc0 = fma(vec4<f32>(av.x), bv, acc0);
      acc1 = fma(vec4<f32>(av.y), bv, acc1);
    }`,
    ).join('\n    ');
    const loadB =
      dtype === 'F32' || native
        ? `{
      let c4 = li / 16u; let kk = li % 16u; let k = k0 + kk; let col0 = n0 + c4 * 4u;
      var bv = vec4<${EB}>(0.0);
      if (inTile && k < ${K}u) {
        let b0 = col0 * ${K}u + k;
        if (col0 + 3u < ${N}u) {
          bv = vec4<${EB}>(w[b0], w[b0 + ${K}u], w[b0 + ${2 * K}u], w[b0 + ${3 * K}u]);
        } else {
          if (col0 < ${N}u) { bv.x = w[b0]; }
          if (col0 + 1u < ${N}u) { bv.y = w[b0 + ${K}u]; }
          if (col0 + 2u < ${N}u) { bv.z = w[b0 + ${2 * K}u]; }
          if (col0 + 3u < ${N}u) { bv.w = w[b0 + ${3 * K}u]; }
        }
      }
      tb[kk * 17u + c4] = bv;
    }`
        : dtype === 'F16'
          ? `if (${K % 2 === 0}) {
      if (li < 128u) {
        let c4 = li / 8u; let kk = (li % 8u) * 2u; let col0 = n0 + c4 * 4u;
        var lo = vec4<f32>(0.0); var hi = vec4<f32>(0.0);
        if (inTile && k0 + kk < ${K}u) {
          let hasHi = k0 + kk + 1u < ${K}u; let kh = ${K / 2}u; let bOff = (col0 * ${K}u + k0 + kk) / 2u;
          if (col0 + 3u < ${N}u) {
            let p0 = unpack2x16float(w[bOff]); let p1 = unpack2x16float(w[bOff + kh]); let p2 = unpack2x16float(w[bOff + 2u * kh]); let p3 = unpack2x16float(w[bOff + 3u * kh]);
            lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x); if (hasHi) { hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y); }
          } else {
            if (col0 < ${N}u) { let p0 = unpack2x16float(w[bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
            if (col0 + 1u < ${N}u) { let p1 = unpack2x16float(w[bOff + kh]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
            if (col0 + 2u < ${N}u) { let p2 = unpack2x16float(w[bOff + 2u * kh]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
            if (col0 + 3u < ${N}u) { let p3 = unpack2x16float(w[bOff + 3u * kh]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }
          }
        }
        tb[kk * 17u + c4] = lo; tb[(kk + 1u) * 17u + c4] = hi;
      }
    }`
          : `if (${K % 4 === 0}) {
      if (li < 64u) {
        let c4 = li / 4u; let kk = (li % 4u) * 4u; let col0 = n0 + c4 * 4u;
        var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
        if (inTile && k0 + kk < ${K}u) {
          let kq = ${K / 4}u; let bOff = (col0 * ${K}u + k0 + kk) / 4u;
          if (col0 < ${N}u) { let u = w[bOff]; let sc = scale[col0]; v0.x = f32(bitcast<i32>(u << 24u) >> 24u) * sc; v1.x = f32(bitcast<i32>(u << 16u) >> 24u) * sc; v2.x = f32(bitcast<i32>(u << 8u) >> 24u) * sc; v3.x = f32(bitcast<i32>(u) >> 24u) * sc; }
          if (col0 + 1u < ${N}u) { let u = w[bOff + kq]; let sc = scale[col0 + 1u]; v0.y = f32(bitcast<i32>(u << 24u) >> 24u) * sc; v1.y = f32(bitcast<i32>(u << 16u) >> 24u) * sc; v2.y = f32(bitcast<i32>(u << 8u) >> 24u) * sc; v3.y = f32(bitcast<i32>(u) >> 24u) * sc; }
          if (col0 + 2u < ${N}u) { let u = w[bOff + 2u * kq]; let sc = scale[col0 + 2u]; v0.z = f32(bitcast<i32>(u << 24u) >> 24u) * sc; v1.z = f32(bitcast<i32>(u << 16u) >> 24u) * sc; v2.z = f32(bitcast<i32>(u << 8u) >> 24u) * sc; v3.z = f32(bitcast<i32>(u) >> 24u) * sc; }
          if (col0 + 3u < ${N}u) { let u = w[bOff + 3u * kq]; let sc = scale[col0 + 3u]; v0.w = f32(bitcast<i32>(u << 24u) >> 24u) * sc; v1.w = f32(bitcast<i32>(u << 16u) >> 24u) * sc; v2.w = f32(bitcast<i32>(u << 8u) >> 24u) * sc; v3.w = f32(bitcast<i32>(u) >> 24u) * sc; }
        }
        tb[kk * 17u + c4] = v0; tb[(kk + 1u) * 17u + c4] = v1; tb[(kk + 2u) * 17u + c4] = v2; tb[(kk + 3u) * 17u + c4] = v3;
      }
    }`;
    return `${native ? 'enable f16;\n' : ''}${decl(0, 'a')}${decl(1, 'w', wType)}${decl(2, 'bias')}${decl(3, 'scale')}${decl(4, 'y', 'f32', true)}
var<workgroup> ta: array<vec2<f32>, 272>;
var<workgroup> tb: array<vec4<${EB}>, 272>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let M_dyn = arrayLength(&a) / ${K}u;
  let tilesN = ${(N + 63) >> 6}u;
  let tilesM = (M_dyn + 31u) / 32u;
  let flat = wid.x + wid.y * 65535u;
  let inTile = flat < tilesM * tilesN;
  let tile = min(flat, tilesM * tilesN - 1u);
  let m0 = (tile / tilesN) * 32u;
  let n0 = (tile % tilesN) * 64u;
  let tm2 = li / 16u; let tx = li % 16u;
  var acc0 = vec4<f32>(0.0); var acc1 = vec4<f32>(0.0);
  let kkA = li % 16u; let r2A = li / 16u; let rowA0 = m0 + r2A * 2u; let fullA = rowA0 + 1u < M_dyn;
  _ = bias[0]; _ = scale[0];
  for (var k0 = 0u; k0 < ${K}u; k0 += 16u) {
    {
      var av = vec2<f32>(0.0);
      if (inTile && k0 + kkA < ${K}u) {
        let aOff = rowA0 * ${K}u + k0 + kkA;
        if (fullA) { av = vec2<f32>(a[aOff], a[aOff + ${K}u]); }
        else if (rowA0 < M_dyn) { av.x = a[aOff]; }
      }
      ta[kkA * 17u + r2A] = av;
    }
    ${loadB}
    workgroupBarrier();
    ${unroll}
    workgroupBarrier();
  }
  if (!inTile) { return; }
  let colBase = n0 + tx * 4u;
  var bVec = vec4<f32>(0.0);
  ${hasBias ? `if (colBase < ${N}u) { bVec.x = bias[colBase]; }
  if (colBase + 1u < ${N}u) { bVec.y = bias[colBase + 1u]; }
  if (colBase + 2u < ${N}u) { bVec.z = bias[colBase + 2u]; }
  if (colBase + 3u < ${N}u) { bVec.w = bias[colBase + 3u]; }` : ''}
  let r0 = m0 + tm2 * 2u;
  if (r0 < M_dyn) {
    let res = acc0 + bVec;
    if (colBase < ${N}u) { y[r0 * ${N}u + colBase] = res.x; }
    if (colBase + 1u < ${N}u) { y[r0 * ${N}u + colBase + 1u] = res.y; }
    if (colBase + 2u < ${N}u) { y[r0 * ${N}u + colBase + 2u] = res.z; }
    if (colBase + 3u < ${N}u) { y[r0 * ${N}u + colBase + 3u] = res.w; }
  }
  let r1 = r0 + 1u;
  if (r1 < M_dyn) {
    let res = acc1 + bVec;
    if (colBase < ${N}u) { y[r1 * ${N}u + colBase] = res.x; }
    if (colBase + 1u < ${N}u) { y[r1 * ${N}u + colBase + 1u] = res.y; }
    if (colBase + 2u < ${N}u) { y[r1 * ${N}u + colBase + 2u] = res.z; }
    if (colBase + 3u < ${N}u) { y[r1 * ${N}u + colBase + 3u] = res.w; }
  }
}`;
  }

  // 64x128 register-tiled GEMM (4x8 per thread, 256 threads) for M > 160
  const unroll = Array.from(
    { length: 16 },
    (_, kk) => `{
      let av = ta[${kk * 17}u + tm4];
      let bv0 = vec4<f32>(tb[${kk * 33}u + tx]);
      let bv1 = vec4<f32>(tb[${kk * 33 + 16}u + tx]);
      acc00 = fma(vec4<f32>(av.x), bv0, acc00); acc01 = fma(vec4<f32>(av.x), bv1, acc01);
      acc10 = fma(vec4<f32>(av.y), bv0, acc10); acc11 = fma(vec4<f32>(av.y), bv1, acc11);
      acc20 = fma(vec4<f32>(av.z), bv0, acc20); acc21 = fma(vec4<f32>(av.z), bv1, acc21);
      acc30 = fma(vec4<f32>(av.w), bv0, acc30); acc31 = fma(vec4<f32>(av.w), bv1, acc31);
    }`,
  ).join('\n    ');
  const loadB =
    dtype === 'F16' && !native
      ? `{
      let kk = (li % 8u) * 2u; let c4 = li / 8u; let col0 = n0 + c4 * 4u;
      var lo = vec4<f32>(0.0); var hi = vec4<f32>(0.0);
      if (inTile && k0 + kk < ${K}u) {
        let hasHi = k0 + kk + 1u < ${K}u; let kh = ${K / 2}u; let bOff = (col0 * ${K}u + k0 + kk) / 2u;
        if (col0 + 3u < ${N}u) {
          let p0 = unpack2x16float(w[bOff]); let p1 = unpack2x16float(w[bOff + kh]); let p2 = unpack2x16float(w[bOff + 2u * kh]); let p3 = unpack2x16float(w[bOff + 3u * kh]);
          lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x); if (hasHi) { hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y); }
        } else {
          if (col0 < ${N}u) { let p0 = unpack2x16float(w[bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
          if (col0 + 1u < ${N}u) { let p1 = unpack2x16float(w[bOff + kh]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
          if (col0 + 2u < ${N}u) { let p2 = unpack2x16float(w[bOff + 2u * kh]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
          if (col0 + 3u < ${N}u) { let p3 = unpack2x16float(w[bOff + 3u * kh]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }
        }
      }
      tb[kk * 33u + c4] = lo; tb[(kk + 1u) * 33u + c4] = hi;
    }`
      : dtype === 'I8'
        ? `if (li < 128u) {
      let c4 = li / 4u; let kk = (li % 4u) * 4u; let col0 = n0 + c4 * 4u;
      var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
      if (inTile && k0 + kk < ${K}u) {
        let kq = ${K / 4}u; let bOff = (col0 * ${K}u + k0 + kk) / 4u;
        if (col0 < ${N}u) { let u = w[bOff]; let sc = scale[col0]; v0.x = f32(bitcast<i32>(u << 24u) >> 24u) * sc; v1.x = f32(bitcast<i32>(u << 16u) >> 24u) * sc; v2.x = f32(bitcast<i32>(u << 8u) >> 24u) * sc; v3.x = f32(bitcast<i32>(u) >> 24u) * sc; }
        if (col0 + 1u < ${N}u) { let u = w[bOff + kq]; let sc = scale[col0 + 1u]; v0.y = f32(bitcast<i32>(u << 24u) >> 24u) * sc; v1.y = f32(bitcast<i32>(u << 16u) >> 24u) * sc; v2.y = f32(bitcast<i32>(u << 8u) >> 24u) * sc; v3.y = f32(bitcast<i32>(u) >> 24u) * sc; }
        if (col0 + 2u < ${N}u) { let u = w[bOff + 2u * kq]; let sc = scale[col0 + 2u]; v0.z = f32(bitcast<i32>(u << 24u) >> 24u) * sc; v1.z = f32(bitcast<i32>(u << 16u) >> 24u) * sc; v2.z = f32(bitcast<i32>(u << 8u) >> 24u) * sc; v3.z = f32(bitcast<i32>(u) >> 24u) * sc; }
        if (col0 + 3u < ${N}u) { let u = w[bOff + 3u * kq]; let sc = scale[col0 + 3u]; v0.w = f32(bitcast<i32>(u << 24u) >> 24u) * sc; v1.w = f32(bitcast<i32>(u << 16u) >> 24u) * sc; v2.w = f32(bitcast<i32>(u << 8u) >> 24u) * sc; v3.w = f32(bitcast<i32>(u) >> 24u) * sc; }
      }
      tb[kk * 33u + c4] = v0; tb[(kk + 1u) * 33u + c4] = v1; tb[(kk + 2u) * 33u + c4] = v2; tb[(kk + 3u) * 33u + c4] = v3;
    }`
        : `for (var e = 0u; e < 2u; e++) {
      let slot = li + e * 256u; let kk = slot % 16u; let c4 = slot / 16u; let k = k0 + kk; let col0 = n0 + c4 * 4u;
      var bv = vec4<${EB}>(0.0);
      if (inTile && k < ${K}u) {
        let b0 = col0 * ${K}u + k;
        if (col0 + 3u < ${N}u) {
          bv = vec4<${EB}>(w[b0], w[b0 + ${K}u], w[b0 + ${2 * K}u], w[b0 + ${3 * K}u]);
        } else {
          if (col0 < ${N}u) { bv.x = w[b0]; }
          if (col0 + 1u < ${N}u) { bv.y = w[b0 + ${K}u]; }
          if (col0 + 2u < ${N}u) { bv.z = w[b0 + ${2 * K}u]; }
          if (col0 + 3u < ${N}u) { bv.w = w[b0 + ${3 * K}u]; }
        }
      }
      tb[kk * 33u + c4] = bv;
    }`;
  const storeRows = [0, 1, 2, 3]
    .map(
      (i) => `{
      let m = m0 + tm4 * 4u + ${i}u;
      if (m < M_dyn) {
        let r0 = acc${i}0 + b0; let r1 = acc${i}1 + b1; let rowOff = m * ${N}u;
        if (c0 < ${N}u) { y[rowOff + c0] = r0.x; } if (c0 + 1u < ${N}u) { y[rowOff + c0 + 1u] = r0.y; } if (c0 + 2u < ${N}u) { y[rowOff + c0 + 2u] = r0.z; } if (c0 + 3u < ${N}u) { y[rowOff + c0 + 3u] = r0.w; }
        if (c1 < ${N}u) { y[rowOff + c1] = r1.x; } if (c1 + 1u < ${N}u) { y[rowOff + c1 + 1u] = r1.y; } if (c1 + 2u < ${N}u) { y[rowOff + c1 + 2u] = r1.z; } if (c1 + 3u < ${N}u) { y[rowOff + c1 + 3u] = r1.w; }
      }
    }`,
    )
    .join('\n    ');
  return `${native ? 'enable f16;\n' : ''}${decl(0, 'a')}${decl(1, 'w', wType)}${decl(2, 'bias')}${decl(3, 'scale')}${decl(4, 'y', 'f32', true)}
var<workgroup> ta: array<vec4<f32>, 272>;
var<workgroup> tb: array<vec4<${EB}>, 528>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let M_dyn = arrayLength(&a) / ${K}u;
  let tilesN = ${(N + 127) >> 7}u;
  let tilesM = (M_dyn + 63u) / 64u;
  let flat = wid.x + wid.y * 65535u;
  let inTile = flat < tilesM * tilesN;
  let tile = min(flat, tilesM * tilesN - 1u);
  let m0 = (tile / tilesN) * 64u;
  let n0 = (tile % tilesN) * 128u;
  let tm4 = li / 16u; let tx = li % 16u;
  var acc00 = vec4<f32>(0.0); var acc01 = vec4<f32>(0.0);
  var acc10 = vec4<f32>(0.0); var acc11 = vec4<f32>(0.0);
  var acc20 = vec4<f32>(0.0); var acc21 = vec4<f32>(0.0);
  var acc30 = vec4<f32>(0.0); var acc31 = vec4<f32>(0.0);
  let kkA = li % 16u; let r4A = li / 16u; let rowA0 = m0 + r4A * 4u; let fullA = rowA0 + 3u < M_dyn;
  _ = bias[0]; _ = scale[0];
  for (var k0 = 0u; k0 < ${K}u; k0 += 16u) {
    {
      var av = vec4<f32>(0.0);
      if (inTile && k0 + kkA < ${K}u) {
        let aOff = rowA0 * ${K}u + k0 + kkA;
        if (fullA) { av = vec4<f32>(a[aOff], a[aOff + ${K}u], a[aOff + ${2 * K}u], a[aOff + ${3 * K}u]); }
        else {
          if (rowA0 < M_dyn) { av.x = a[aOff]; }
          if (rowA0 + 1u < M_dyn) { av.y = a[aOff + ${K}u]; }
          if (rowA0 + 2u < M_dyn) { av.z = a[aOff + ${2 * K}u]; }
          if (rowA0 + 3u < M_dyn) { av.w = a[aOff + ${3 * K}u]; }
        }
      }
      ta[kkA * 17u + r4A] = av;
    }
    ${loadB}
    workgroupBarrier();
    ${unroll}
    workgroupBarrier();
  }
  if (!inTile) { return; }
  let c0 = n0 + tx * 4u; let c1 = c0 + 64u;
  var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0);
  ${hasBias ? `if (c0 < ${N}u) { b0.x = bias[c0]; } if (c0 + 1u < ${N}u) { b0.y = bias[c0 + 1u]; } if (c0 + 2u < ${N}u) { b0.z = bias[c0 + 2u]; } if (c0 + 3u < ${N}u) { b0.w = bias[c0 + 3u]; }
  if (c1 < ${N}u) { b1.x = bias[c1]; } if (c1 + 1u < ${N}u) { b1.y = bias[c1 + 1u]; } if (c1 + 2u < ${N}u) { b1.z = bias[c1 + 2u]; } if (c1 + 3u < ${N}u) { b1.w = bias[c1 + 3u]; }` : ''}
  ${storeRows}
}`;
}

export function normSource(L, D, { stride = D, offset = 0, eps = 1e-5, bias = true } = {}) {
  return `${decl(0, 'x')}${decl(1, 'w')}${decl(2, 'b')}${decl(3, 'y', 'f32', true)}
var<workgroup> temp: array<f32, 64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let rows = arrayLength(&y) / ${D}u;
  let r = g.x + g.y * 65535u;
  let live = r < rows;
  let row = min(r, rows - 1u);
  let base = row * ${stride}u + ${offset}u;
  _ = b[0];
  var s = 0.0;
  for (var j = t; j < ${D}u; j += 64u) { s += x[base + j]; }
  temp[t] = s; workgroupBarrier();
  if (t < 32u) { temp[t] += temp[t + 32u]; } workgroupBarrier();
  if (t < 16u) { temp[t] += temp[t + 16u]; } workgroupBarrier();
  if (t < 8u) { temp[t] += temp[t + 8u]; } workgroupBarrier();
  if (t == 0u) { temp[0] = ((temp[0] + temp[4]) + (temp[2] + temp[6])) + ((temp[1] + temp[5]) + (temp[3] + temp[7])); }
  workgroupBarrier();
  let mean = temp[0] / ${D}.0;
  workgroupBarrier();
  var v = 0.0;
  for (var j = t; j < ${D}u; j += 64u) { let z = x[base + j] - mean; v += z * z; }
  temp[t] = v; workgroupBarrier();
  if (t < 32u) { temp[t] += temp[t + 32u]; } workgroupBarrier();
  if (t < 16u) { temp[t] += temp[t + 16u]; } workgroupBarrier();
  if (t < 8u) { temp[t] += temp[t + 8u]; } workgroupBarrier();
  if (t == 0u) { temp[0] = ((temp[0] + temp[4]) + (temp[2] + temp[6])) + ((temp[1] + temp[5]) + (temp[3] + temp[7])); }
  workgroupBarrier();
  let inv = inverseSqrt(temp[0] / ${D}.0 + ${eps});
  if (!live) { return; }
  for (var j = t; j < ${D}u; j += 64u) { y[row * ${D}u + j] = (x[base + j] - mean) * inv * w[j]${bias ? ' + b[j]' : ''}; }
}`;
}

export function elementSource(n, expression, inputs = 1) {
  return (
    Array.from({ length: inputs }, (_, i) => decl(i, ['x', 'b', 'c'][i])).join('') +
    decl(inputs, 'y', 'f32', true) +
    `\n@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>) {let i=g.x+g.y*16776960u;if(i<arrayLength(&y)){${expression}}}`
  );
}

export function gatherSource(L, D) {
  return (
    decl(0, 'tokens', 'u32') +
    decl(1, 'w') +
    decl(2, 'y', 'f32', true) +
    `\n@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>) {let i=g.x+g.y*16776960u;if(i<arrayLength(&y)){y[i]=w[tokens[i/${D}u]*${D}u+i%${D}u];}}`
  );
}

export function gatherAddSource(L, D) {
  return (
    decl(0, 'seqTok', 'u32') +
    decl(1, 'structTok', 'u32') +
    decl(2, 'emb') +
    decl(3, 'delta') +
    decl(4, 'y', 'f32', true) +
    `\n@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>) {let i=g.x+g.y*16776960u;if(i<arrayLength(&y)){let r=i/${D}u;let d=i%${D}u;y[i]=emb[seqTok[r]*${D}u+d]+delta[structTok[r]*${D}u+d];}}`
  );
}

export function ropeSource(L, D, H) {
  const dh = D / H,
    half = dh / 2;
  return elementSource(
    L * D,
    `let p=i/${D}u;let d=i%${dh}u;let j=d%${half}u;let angle=f32(p)*pow(10000.0,-f32(j)*${2 / dh});let other=select(i+${half}u,i-${half}u,d>=${half}u);let sign=select(-1.0,1.0,d>=${half}u);y[i]=x[i]*cos(angle)+sign*x[other]*sin(angle);`,
  );
}

// Streaming register FlashAttention for 64-channel heads (ESM-3 24x64 backbone).
export function streamingAttn64Source(D, H) {
  const vectors = 16;
  const rowVecs = H * vectors;
  const perVec = (fn) => Array.from({ length: vectors }, (_, i) => fn(i)).join('\n  ');
  return `${decl(0, 'query', 'vec4<f32>')}${decl(1, 'key', 'vec4<f32>')}${decl(2, 'qkv', 'vec4<f32>')}${decl(3, 'out', 'vec4<f32>', true)}
var<workgroup> keys: array<vec4<f32>, 128>;
var<workgroup> vals: array<vec4<f32>, 128>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let L_dyn = arrayLength(&query) / ${rowVecs}u;
  let row = group.x * 64u + lane;
  let headBase = group.y * 16u;
  let safeRow = min(row, L_dyn - 1u);
  let base = safeRow * ${rowVecs}u + headBase;
  ${perVec((i) => `let q${i} = query[base + ${i}u]; var acc${i} = vec4<f32>(0.0);`)}
  var maximum = -3.402823e38;
  var denominator = 0.0;
  for (var begin = 0u; begin < L_dyn; begin += 8u) {
    for (var t = lane; t < 128u; t += 64u) {
      let col = begin + t / 16u;
      if (col < L_dyn) {
        let d = headBase + t % 16u;
        keys[t] = key[col * ${rowVecs}u + d];
        vals[t] = qkv[(3u * col + 2u) * ${rowVecs}u + d];
      }
    }
    workgroupBarrier();
    let endCol = min(begin + 8u, L_dyn);
    for (var col = begin; col < endCol; col++) {
      let offset = (col - begin) * 16u;
      var product = vec4<f32>(0.0);
      ${perVec((i) => `product = fma(q${i}, keys[offset + ${i}u], product);`)}
      let score = dot(product, vec4<f32>(0.125));
      let nextMax = max(maximum, score);
      let previous = exp(maximum - nextMax);
      let weight = exp(score - nextMax);
      denominator = denominator * previous + weight;
      ${perVec((i) => `acc${i} = acc${i} * previous + weight * vals[offset + ${i}u];`)}
      maximum = nextMax;
    }
    workgroupBarrier();
  }
  if (row < L_dyn) {
    let invDen = 1.0 / denominator;
    ${perVec((i) => `out[base + ${i}u] = acc${i} * invDen;`)}
  }
}`;
}

export function scoresSource(L, D, H, { packedQkv = false } = {}) {
  const dh = D / H;
  const rowStride = packedQkv ? 3 * D : D;
  const keyOffset = packedQkv ? D : 0;
  return (
    decl(0, 'q') +
    decl(1, 'k') +
    decl(2, 'y', 'f32', true) +
    `\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x+g.y*4194240u;if(i>=${H * L * L}u){return;}let h=i/${L * L}u;let r=(i/${L}u)%${L}u;let col=i%${L}u;var s=0.0;for(var j=0u;j<${dh}u;j++){s+=q[r*${rowStride}u+h*${dh}u+j]*k[col*${rowStride}u+${keyOffset}u+h*${dh}u+j];}y[i]=s*${1 / Math.sqrt(dh)};}`
  );
}

export function softmaxSource(L, H) {
  return (
    decl(0, 'x') +
    decl(1, 'y', 'f32', true) +
    `\nvar<workgroup> t:array<f32,64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) j:u32,@builtin(workgroup_id) g:vec3<u32>) {
 let rows=arrayLength(&y)/${L}u;
 let r=g.x+g.y*65535u;let live=r<rows;let row=min(r,rows-1u);let off=row*${L}u;var m=-3.4028234663852886e38;
 for(var k=j;k<${L}u;k+=64u){m=max(m,x[off+k]);}t[j]=m;workgroupBarrier();
 if(j<32u){t[j]=max(t[j],t[j+32u]);}workgroupBarrier();
 if(j<16u){t[j]=max(t[j],t[j+16u]);}workgroupBarrier();
 if(j<8u){t[j]=max(t[j],t[j+8u]);}workgroupBarrier();
 if(j==0u){t[0]=max(max(max(t[0],t[4]),max(t[2],t[6])),max(max(t[1],t[5]),max(t[3],t[7])));}workgroupBarrier();
 m=t[0];workgroupBarrier();
 var s=0.0;for(var k=j;k<${L}u;k+=64u){s+=exp(x[off+k]-m);}t[j]=s;workgroupBarrier();
 if(j<32u){t[j]+=t[j+32u];}workgroupBarrier();
 if(j<16u){t[j]+=t[j+16u];}workgroupBarrier();
 if(j<8u){t[j]+=t[j+8u];}workgroupBarrier();
 if(j==0u){t[0]=((t[0]+t[4])+(t[2]+t[6]))+((t[1]+t[5])+(t[3]+t[7]));}workgroupBarrier();
 let den=t[0];if(!live){return;}
 for(var k=j;k<${L}u;k+=64u){y[off+k]=exp(x[off+k]-m)/den;}
}`
  );
}

export function contextSource(L, D, H, { stride = D, offset = 0 } = {}) {
  const dh = D / H;
  return (
    decl(0, 'p') +
    decl(1, 'v') +
    decl(2, 'y', 'f32', true) +
    `\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x+g.y*4194240u;if(i>=${L * D}u){return;}let r=i/${D}u;let d=i%${D}u;let h=d/${dh}u;var s=0.0;
 for(var j=0u;j<${L}u;j++){s+=p[(h*${L}u+r)*${L}u+j]*v[j*${stride}u+${offset}u+d];}y[i]=s;
}`
  );
}

/** ESM-3 StructureTokenEncoder geometric attention and VQ codebook shaders. */
export function geomRotateSource(R, H) {
  return (
    decl(0, 'z') +
    decl(1, 'frames') +
    decl(2, 'y', 'f32', true) +
    `\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x+g.y*4194240u;if(i>=arrayLength(&y)){return;}
 let r=i/${H * 15}u;let v=(i%${H * 15}u)/3u;let d=i%3u;let off=r*${H * 15}u+v*3u;let f=r*13u+d*3u;
 var s=z[off]*frames[f]+z[off+1u]*frames[f+1u]+z[off+2u]*frames[f+2u];
 if(v>=${3 * H}u){s+=frames[r*13u+9u+d];}
 y[i]=s;
}`
  );
}

export function geomScoresSource(K, H, B) {
  return (
    decl(0, 'vec') +
    decl(1, 'frames') +
    decl(2, 'dScale') +
    decl(3, 'rScale') +
    decl(4, 'y', 'f32', true) +
    `\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x+g.y*4194240u;if(i>=arrayLength(&y)){return;}
 let col=i%${K}u;let row=(i/${K}u)%${K}u;let h=(i/${K * K}u)%${H}u;let batch=i/${H * K * K}u;
 let r=batch*${K}u+row;let c=batch*${K}u+col;
 let qi=r*${H * 15}u+h*3u;let ki=c*${H * 15}u+${H * 3}u+h*3u;
 var rot=0.0;var dis=0.0;
 for(var d=0u;d<3u;d++){rot+=vec[qi+d]*vec[ki+d];let e=vec[qi+${H * 9}u+d]-vec[ki+${H * 9}u+d];dis+=e*e;}
 let bias=select(-3.4028234663852886e38,1.0,frames[c*13u+12u]>0.0);
 let rs=rScale[h];let ds=dScale[h];
 let rw=select(log(1.0+exp(rs)),rs,rs>20.0);
 let dw=select(log(1.0+exp(ds)),ds,ds>20.0);
 y[i]=(rot*0.5773502691896258)*rw-(sqrt(dis)*0.5773502691896258)*dw+bias;
}`
  );
}

export function geomContextSource(K, H, B) {
  return (
    decl(0, 'prob') +
    decl(1, 'vec') +
    decl(2, 'frames') +
    decl(3, 'y', 'f32', true) +
    `\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x+g.y*4194240u;if(i>=arrayLength(&y)){return;}
 let d=i%3u;let h=(i/3u)%${H}u;let row=(i/${H * 3}u)%${K}u;let batch=i/${K * H * 3}u;let r=batch*${K}u+row;
 var v=vec3<f32>(0.0);
 for(var col=0u;col<${K}u;col++){
  let p=prob[((batch*${H}u+h)*${K}u+row)*${K}u+col];
  let k=(batch*${K}u+col)*${H * 15}u+${H * 6}u+h*3u;
  v+=p*vec3<f32>(vec[k],vec[k+1u],vec[k+2u]);
 }
 y[i]=v.x*frames[r*13u+d]+v.y*frames[r*13u+3u+d]+v.z*frames[r*13u+6u+d];
}`
  );
}

export function selectLocalSource(B, K, D) {
  return (
    decl(0, 'x') +
    decl(1, 'frames') +
    decl(2, 'y', 'f32', true) +
    `\n@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x+g.y*16776960u;if(i>=arrayLength(&y)){return;}
 let r=i/${D}u;
 y[i]=select(0.0,x[r*${K * D}u+i%${D}u],frames[r*13u+12u]>0.0);
}`
  );
}

export function codebookArgminSource(L, codes = 4096, dim = 128) {
  return (
    decl(0, 'z') +
    decl(1, 'dots') +
    decl(2, 'norms') +
    decl(3, 'y', 'u32', true) +
    `\nvar<workgroup> bestD:array<f32,256>;
var<workgroup> bestI:array<u32,256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) g:vec3<u32>) {
 let rows=arrayLength(&y);
 let r=g.x+g.y*65535u;if(r>=rows){return;}
 var zn=0.0;for(var d=0u;d<${dim}u;d++){let v=z[r*${dim}u+d];zn+=v*v;}
 var bd=3.4028234663852886e38;var bi=0u;
 for(var k=t;k<${codes}u;k+=256u){
  let dist=zn-2.0*dots[r*${codes}u+k]+norms[k];
  if(dist<bd||(dist==bd&&k<bi)){bd=dist;bi=k;}
 }
 bestD[t]=bd;bestI[t]=bi;workgroupBarrier();
 for(var n=128u;n>0u;n/=2u){
  if(t<n){
   let od=bestD[t+n];let oi=bestI[t+n];
   if(od<bestD[t]||(od==bestD[t]&&oi<bestI[t])){bestD[t]=od;bestI[t]=oi;}
  }
  workgroupBarrier();
 }
 if(t==0u){y[r]=bestI[0];}
}`
  );
}
