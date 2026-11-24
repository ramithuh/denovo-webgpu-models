// All dimensions arrive through uniforms. No shape/step-specialized shader source.
export const params = `struct P { n:u32,c:u32,k:u32,m:u32,a:u32,b:u32,d:u32,e:u32,t:f32,s:f32,z:f32,w:f32 }; @group(0) @binding(BIND) var<uniform> p:P;`;

export function shader(buffers, body) {
  return (
    buffers
      .map(
        (v, i) =>
          `@group(0) @binding(${i}) var<storage,${v[1] ?? "read"}> ${v[0]}:array<${v[2] ?? "f32"}>;`,
      )
      .join("\n") +
    "\n" +
    params.replace("BIND", buffers.length) +
    "\n" +
    body
  );
}

function weightScalarFn(dtype, half) {
  const fetch =
    dtype === "F32"
      ? "return bitcast<f32>(w[i]);"
      : dtype === "F16"
        ? half
          ? "return f32(w[i]);"
          : "let v=unpack2x16float(w[i/2u]);return select(v.x,v.y,(i&1u)==1u);"
        : "let word=w[i/4u];let sh=(i&3u)*8u;let sb=bitcast<i32>(((word>>sh)&255u)<<24u)>>24u;return f32(sb)*scale[o];";
  return `fn weight(i:u32,o:u32)->f32{${fetch}}`;
}

// Fast K-parallel GEMV/small-M kernel for M <= 4 and K % 4 == 0 (conditioning projections).
export function linearSmallM(dtype, useF16) {
  const half = dtype === "F16" && useF16;
  const loadVec4 =
    dtype === "F32"
      ? "let wi=sCol*p.k+kk;let wv=vec4<f32>(bitcast<f32>(w[wi]),bitcast<f32>(w[wi+1u]),bitcast<f32>(w[wi+2u]),bitcast<f32>(w[wi+3u]));"
      : dtype === "F16"
        ? half
          ? "let wi=sCol*p.k+kk;let wv=vec4<f32>(f32(w[wi]),f32(w[wi+1u]),f32(w[wi+2u]),f32(w[wi+3u]));"
          : "let wu=(sCol*p.k+kk)/2u;let wv=vec4<f32>(unpack2x16float(w[wu]),unpack2x16float(w[wu+1u]));"
        : "let packed=w[(sCol*p.k+kk)/4u];let sc=scale[sCol];let wv=vec4<f32>(f32(bitcast<i32>(packed<<24u)>>24u)*sc,f32(bitcast<i32>(packed<<16u)>>24u)*sc,f32(bitcast<i32>(packed<<8u)>>24u)*sc,f32(bitcast<i32>(packed)>>24u)*sc);";
  return (
    (half ? "enable f16;\n" : "") +
    shader(
      [
        ["x"],
        ["w", "read", half ? "f16" : "u32"],
        ["bias"],
        ["scale"],
        ["y", "read_write"],
      ],
      `
 var<workgroup> sx:array<vec4<f32>,64>;
 var<workgroup> red:array<vec4<f32>,128>;
 @compute @workgroup_size(128)
 fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let wg=wid.x+wid.y*65535u;
  let cSub=li/16u;
  let kLane=li%16u;
  let outCol=wg*8u+cSub;
  let colValid=outCol<p.c;
  let sCol=min(outCol,p.c-1u);
  var acc=vec4<f32>(0.0);
  for(var k0=0u;k0<p.k;k0+=64u){
   if(li<64u){
    let r=li/16u;let kl=li%16u;let kk=k0+kl*4u;
    var xv=vec4<f32>(0.0);
    if(r<p.n && kk<p.k){let xb=r*p.k+kk;xv=vec4<f32>(x[xb],x[xb+1u],x[xb+2u],x[xb+3u]);}
    sx[li]=xv;
   }
   workgroupBarrier();
   let kk=k0+kLane*4u;
   if(colValid && kk<p.k){
    ${loadVec4}
    acc+=vec4<f32>(dot(sx[kLane],wv),dot(sx[16u+kLane],wv),dot(sx[32u+kLane],wv),dot(sx[48u+kLane],wv));
   }
   workgroupBarrier();
  }
  red[li]=acc;
  workgroupBarrier();
  if(kLane<8u){red[li]+=red[li+8u];} workgroupBarrier();
  if(kLane<4u){red[li]+=red[li+4u];} workgroupBarrier();
  if(kLane<2u){red[li]+=red[li+2u];} workgroupBarrier();
  if(colValid && kLane==0u){
   let tot=red[li]+red[li+1u];
   var b=0.0;if(p.a!=0u){b=bias[outCol];}
   if(0u<p.n){y[outCol]=tot.x+b;}
   if(1u<p.n){y[p.c+outCol]=tot.y+b;}
   if(2u<p.n){y[2u*p.c+outCol]=tot.z+b;}
   if(3u<p.n){y[3u*p.c+outCol]=tot.w+b;}
  }
 }`,
    )
  );
}

// Register-tiled 32x64 GEMM (2x4 per thread, 256 threads, BK=16, padded vec2/vec4 shared panels).
export function linear32x64(dtype, useF16) {
  const half = dtype === "F16" && useF16;
  const EB = half ? "f16" : "f32";
  const unroll = Array.from(
    { length: 16 },
    (_, kk) => `{
    let av=ta[${kk * 17}u+tm2];
    let bv=vec4<f32>(tb[${kk * 17}u+tx]);
    acc0+=vec4<f32>(av.x)*bv;
    acc1+=vec4<f32>(av.y)*bv;
   }`,
  ).join("\n   ");
  const loadBBody =
    dtype === "F32" || half
      ? `{
   let c4=li/16u;let kk=li%16u;let k=k0+kk;let col0=n0+c4*4u;
   var bv=vec4<${EB}>(0.0);
   if(inTile && k<p.k){
    let b0=col0*p.k+k;
    if(col0+3u<p.c){
     bv=vec4<${EB}>(${half ? "w[b0],w[b0+p.k],w[b0+2u*p.k],w[b0+3u*p.k]" : "bitcast<f32>(w[b0]),bitcast<f32>(w[b0+p.k]),bitcast<f32>(w[b0+2u*p.k]),bitcast<f32>(w[b0+3u*p.k])"});
    } else {
     if(col0<p.c){bv.x=${half ? "w[b0]" : "bitcast<f32>(w[b0])"};}
     if(col0+1u<p.c){bv.y=${half ? "w[b0+p.k]" : "bitcast<f32>(w[b0+p.k])"};}
     if(col0+2u<p.c){bv.z=${half ? "w[b0+2u*p.k]" : "bitcast<f32>(w[b0+2u*p.k])"};}
     if(col0+3u<p.c){bv.w=${half ? "w[b0+3u*p.k]" : "bitcast<f32>(w[b0+3u*p.k])"};}
    }
   }
   tb[kk*17u+c4]=bv;
  }`
      : dtype === "F16"
        ? `if((p.k&1u)==0u){
   if(li<128u){
    let c4=li/8u;let kk=(li%8u)*2u;let col0=n0+c4*4u;
    var lo=vec4<f32>(0.0);var hi=vec4<f32>(0.0);
    if(inTile && k0+kk<p.k){
     let hasHi=k0+kk+1u<p.k;let kh=p.k/2u;let bOff=(col0*p.k+k0+kk)/2u;
     if(col0+3u<p.c){
      let p0=unpack2x16float(w[bOff]);let p1=unpack2x16float(w[bOff+kh]);let p2=unpack2x16float(w[bOff+2u*kh]);let p3=unpack2x16float(w[bOff+3u*kh]);
      lo=vec4<f32>(p0.x,p1.x,p2.x,p3.x);if(hasHi){hi=vec4<f32>(p0.y,p1.y,p2.y,p3.y);}
     } else {
      if(col0<p.c){let p0=unpack2x16float(w[bOff]);lo.x=p0.x;if(hasHi){hi.x=p0.y;}}
      if(col0+1u<p.c){let p1=unpack2x16float(w[bOff+kh]);lo.y=p1.x;if(hasHi){hi.y=p1.y;}}
      if(col0+2u<p.c){let p2=unpack2x16float(w[bOff+2u*kh]);lo.z=p2.x;if(hasHi){hi.z=p2.y;}}
      if(col0+3u<p.c){let p3=unpack2x16float(w[bOff+3u*kh]);lo.w=p3.x;if(hasHi){hi.w=p3.y;}}
     }
    }
    tb[kk*17u+c4]=lo;tb[(kk+1u)*17u+c4]=hi;
   }
  } else {
   let c4=li/16u;let kk=li%16u;let k=k0+kk;let col0=n0+c4*4u;var bv=vec4<f32>(0.0);
   if(inTile && k<p.k){
    if(col0<p.c){bv.x=weight(col0*p.k+k,col0);}
    if(col0+1u<p.c){bv.y=weight((col0+1u)*p.k+k,col0+1u);}
    if(col0+2u<p.c){bv.z=weight((col0+2u)*p.k+k,col0+2u);}
    if(col0+3u<p.c){bv.w=weight((col0+3u)*p.k+k,col0+3u);}
   }
   tb[kk*17u+c4]=bv;
  }`
        : `if((p.k&3u)==0u){
   if(li<64u){
    let c4=li/4u;let kk=(li%4u)*4u;let col0=n0+c4*4u;
    var v0=vec4<f32>(0.0);var v1=vec4<f32>(0.0);var v2=vec4<f32>(0.0);var v3=vec4<f32>(0.0);
    if(inTile && k0+kk<p.k){
     let kq=p.k/4u;let bOff=(col0*p.k+k0+kk)/4u;
     if(col0<p.c){let u=w[bOff];let sc=scale[col0];v0.x=f32(bitcast<i32>(u<<24u)>>24u)*sc;if(k0+kk+1u<p.k){v1.x=f32(bitcast<i32>(u<<16u)>>24u)*sc;}if(k0+kk+2u<p.k){v2.x=f32(bitcast<i32>(u<<8u)>>24u)*sc;}if(k0+kk+3u<p.k){v3.x=f32(bitcast<i32>(u)>>24u)*sc;}}
     if(col0+1u<p.c){let u=w[bOff+kq];let sc=scale[col0+1u];v0.y=f32(bitcast<i32>(u<<24u)>>24u)*sc;if(k0+kk+1u<p.k){v1.y=f32(bitcast<i32>(u<<16u)>>24u)*sc;}if(k0+kk+2u<p.k){v2.y=f32(bitcast<i32>(u<<8u)>>24u)*sc;}if(k0+kk+3u<p.k){v3.y=f32(bitcast<i32>(u)>>24u)*sc;}}
     if(col0+2u<p.c){let u=w[bOff+2u*kq];let sc=scale[col0+2u];v0.z=f32(bitcast<i32>(u<<24u)>>24u)*sc;if(k0+kk+1u<p.k){v1.z=f32(bitcast<i32>(u<<16u)>>24u)*sc;}if(k0+kk+2u<p.k){v2.z=f32(bitcast<i32>(u<<8u)>>24u)*sc;}if(k0+kk+3u<p.k){v3.z=f32(bitcast<i32>(u)>>24u)*sc;}}
     if(col0+3u<p.c){let u=w[bOff+3u*kq];let sc=scale[col0+3u];v0.w=f32(bitcast<i32>(u<<24u)>>24u)*sc;if(k0+kk+1u<p.k){v1.w=f32(bitcast<i32>(u<<16u)>>24u)*sc;}if(k0+kk+2u<p.k){v2.w=f32(bitcast<i32>(u<<8u)>>24u)*sc;}if(k0+kk+3u<p.k){v3.w=f32(bitcast<i32>(u)>>24u)*sc;}}
    }
    tb[kk*17u+c4]=v0;tb[(kk+1u)*17u+c4]=v1;tb[(kk+2u)*17u+c4]=v2;tb[(kk+3u)*17u+c4]=v3;
   }
  } else {
   let c4=li/16u;let kk=li%16u;let k=k0+kk;let col0=n0+c4*4u;var bv=vec4<f32>(0.0);
   if(inTile && k<p.k){
    if(col0<p.c){bv.x=weight(col0*p.k+k,col0);}
    if(col0+1u<p.c){bv.y=weight((col0+1u)*p.k+k,col0+1u);}
    if(col0+2u<p.c){bv.z=weight((col0+2u)*p.k+k,col0+2u);}
    if(col0+3u<p.c){bv.w=weight((col0+3u)*p.k+k,col0+3u);}
   }
   tb[kk*17u+c4]=bv;
  }`;
  return (
    (half ? "enable f16;\n" : "") +
    shader(
      [
        ["x"],
        ["w", "read", half ? "f16" : "u32"],
        ["bias"],
        ["scale"],
        ["y", "read_write"],
      ],
      `
 ${weightScalarFn(dtype, half)}
 var<workgroup> ta:array<vec2<f32>,272>;
 var<workgroup> tb:array<vec4<${EB}>,272>;
 @compute @workgroup_size(256)
 fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let tilesN=(p.c+63u)/64u;
  let tilesM=(p.n+31u)/32u;
  let flat=wid.x+wid.y*65535u;
  let inTile=flat<tilesM*tilesN;
  let tile=min(flat,tilesM*tilesN-1u);
  let m0=(tile/tilesN)*32u;
  let n0=(tile%tilesN)*64u;
  let tm2=li/16u;
  let tx=li%16u;
  var acc0=vec4<f32>(0.0);
  var acc1=vec4<f32>(0.0);
  let kkA=li%16u;
  let r2A=li/16u;
  let rowA0=m0+r2A*2u;
  let fullA=rowA0+1u<p.n;
  for(var k0=0u;k0<p.k;k0+=16u){
   {
    var av=vec2<f32>(0.0);
    if(inTile && k0+kkA<p.k){
     let aOff=rowA0*p.k+k0+kkA;
     if(fullA){av=vec2<f32>(x[aOff],x[aOff+p.k]);}
     else if(rowA0<p.n){av.x=x[aOff];}
    }
    ta[kkA*17u+r2A]=av;
   }
   ${loadBBody}
   workgroupBarrier();
   ${unroll}
   workgroupBarrier();
  }
  if(!inTile){return;}
  let colBase=n0+tx*4u;
  var bVec=vec4<f32>(0.0);
  if(p.a!=0u){
   if(colBase<p.c){bVec.x=bias[colBase];}
   if(colBase+1u<p.c){bVec.y=bias[colBase+1u];}
   if(colBase+2u<p.c){bVec.z=bias[colBase+2u];}
   if(colBase+3u<p.c){bVec.w=bias[colBase+3u];}
  }
  let r0=m0+tm2*2u;
  if(r0<p.n){
   let res=acc0+bVec;
   if(colBase<p.c){y[r0*p.c+colBase]=res.x;}
   if(colBase+1u<p.c){y[r0*p.c+colBase+1u]=res.y;}
   if(colBase+2u<p.c){y[r0*p.c+colBase+2u]=res.z;}
   if(colBase+3u<p.c){y[r0*p.c+colBase+3u]=res.w;}
  }
  let r1=r0+1u;
  if(r1<p.n){
   let res=acc1+bVec;
   if(colBase<p.c){y[r1*p.c+colBase]=res.x;}
   if(colBase+1u<p.c){y[r1*p.c+colBase+1u]=res.y;}
   if(colBase+2u<p.c){y[r1*p.c+colBase+2u]=res.z;}
   if(colBase+3u<p.c){y[r1*p.c+colBase+3u]=res.w;}
  }
 }`,
    )
  );
}

// Register-tiled 64x128 GEMM (4x8 per thread, 256 threads, BK=16, 12.5 KiB shared memory).
export function linear64x128(dtype, useF16) {
  const half = dtype === "F16" && useF16;
  const EB = half ? "f16" : "f32";
  const unroll = Array.from(
    { length: 16 },
    (_, kk) => `{
    let av=ta[${kk * 17}u+tm4];
    let bv0=vec4<f32>(tb[${kk * 33}u+tx]);
    let bv1=vec4<f32>(tb[${kk * 33 + 16}u+tx]);
    acc00+=vec4<f32>(av.x)*bv0;acc01+=vec4<f32>(av.x)*bv1;
    acc10+=vec4<f32>(av.y)*bv0;acc11+=vec4<f32>(av.y)*bv1;
    acc20+=vec4<f32>(av.z)*bv0;acc21+=vec4<f32>(av.z)*bv1;
    acc30+=vec4<f32>(av.w)*bv0;acc31+=vec4<f32>(av.w)*bv1;
   }`,
  ).join("\n   ");
  const loadBBody =
    dtype === "F16" && !half
      ? `if((p.k&1u)==0u){
   let kk=(li%8u)*2u;let c4=li/8u;let col0=n0+c4*4u;
   var lo=vec4<f32>(0.0);var hi=vec4<f32>(0.0);
   if(inTile && k0+kk<p.k){
    let hasHi=k0+kk+1u<p.k;let kh=p.k/2u;let bOff=(col0*p.k+k0+kk)/2u;
    if(col0+3u<p.c){
     let p0=unpack2x16float(w[bOff]);let p1=unpack2x16float(w[bOff+kh]);let p2=unpack2x16float(w[bOff+2u*kh]);let p3=unpack2x16float(w[bOff+3u*kh]);
     lo=vec4<f32>(p0.x,p1.x,p2.x,p3.x);if(hasHi){hi=vec4<f32>(p0.y,p1.y,p2.y,p3.y);}
    } else {
     if(col0<p.c){let p0=unpack2x16float(w[bOff]);lo.x=p0.x;if(hasHi){hi.x=p0.y;}}
     if(col0+1u<p.c){let p1=unpack2x16float(w[bOff+kh]);lo.y=p1.x;if(hasHi){hi.y=p1.y;}}
     if(col0+2u<p.c){let p2=unpack2x16float(w[bOff+2u*kh]);lo.z=p2.x;if(hasHi){hi.z=p2.y;}}
     if(col0+3u<p.c){let p3=unpack2x16float(w[bOff+3u*kh]);lo.w=p3.x;if(hasHi){hi.w=p3.y;}}
    }
   }
   tb[kk*33u+c4]=lo;tb[(kk+1u)*33u+c4]=hi;
  } else {
   for(var e=0u;e<2u;e++){
    let slot=li+e*256u;let kk=slot%16u;let c4=slot/16u;let k=k0+kk;let col0=n0+c4*4u;var bv=vec4<f32>(0.0);
    if(inTile && k<p.k){
     if(col0<p.c){bv.x=weight(col0*p.k+k,col0);}
     if(col0+1u<p.c){bv.y=weight((col0+1u)*p.k+k,col0+1u);}
     if(col0+2u<p.c){bv.z=weight((col0+2u)*p.k+k,col0+2u);}
     if(col0+3u<p.c){bv.w=weight((col0+3u)*p.k+k,col0+3u);}
    }
    tb[kk*33u+c4]=bv;
   }
  }`
      : dtype === "I8"
        ? `if((p.k&3u)==0u){
   if(li<128u){
    let c4=li/4u;let kk=(li%4u)*4u;let col0=n0+c4*4u;
    var v0=vec4<f32>(0.0);var v1=vec4<f32>(0.0);var v2=vec4<f32>(0.0);var v3=vec4<f32>(0.0);
    if(inTile && k0+kk<p.k){
     let kq=p.k/4u;let bOff=(col0*p.k+k0+kk)/4u;
     if(col0<p.c){let u=w[bOff];let sc=scale[col0];v0.x=f32(bitcast<i32>(u<<24u)>>24u)*sc;if(k0+kk+1u<p.k){v1.x=f32(bitcast<i32>(u<<16u)>>24u)*sc;}if(k0+kk+2u<p.k){v2.x=f32(bitcast<i32>(u<<8u)>>24u)*sc;}if(k0+kk+3u<p.k){v3.x=f32(bitcast<i32>(u)>>24u)*sc;}}
     if(col0+1u<p.c){let u=w[bOff+kq];let sc=scale[col0+1u];v0.y=f32(bitcast<i32>(u<<24u)>>24u)*sc;if(k0+kk+1u<p.k){v1.y=f32(bitcast<i32>(u<<16u)>>24u)*sc;}if(k0+kk+2u<p.k){v2.y=f32(bitcast<i32>(u<<8u)>>24u)*sc;}if(k0+kk+3u<p.k){v3.y=f32(bitcast<i32>(u)>>24u)*sc;}}
     if(col0+2u<p.c){let u=w[bOff+2u*kq];let sc=scale[col0+2u];v0.z=f32(bitcast<i32>(u<<24u)>>24u)*sc;if(k0+kk+1u<p.k){v1.z=f32(bitcast<i32>(u<<16u)>>24u)*sc;}if(k0+kk+2u<p.k){v2.z=f32(bitcast<i32>(u<<8u)>>24u)*sc;}if(k0+kk+3u<p.k){v3.z=f32(bitcast<i32>(u)>>24u)*sc;}}
     if(col0+3u<p.c){let u=w[bOff+3u*kq];let sc=scale[col0+3u];v0.w=f32(bitcast<i32>(u<<24u)>>24u)*sc;if(k0+kk+1u<p.k){v1.w=f32(bitcast<i32>(u<<16u)>>24u)*sc;}if(k0+kk+2u<p.k){v2.w=f32(bitcast<i32>(u<<8u)>>24u)*sc;}if(k0+kk+3u<p.k){v3.w=f32(bitcast<i32>(u)>>24u)*sc;}}
    }
    tb[kk*33u+c4]=v0;tb[(kk+1u)*33u+c4]=v1;tb[(kk+2u)*33u+c4]=v2;tb[(kk+3u)*33u+c4]=v3;
   }
  } else {
   for(var e=0u;e<2u;e++){
    let slot=li+e*256u;let kk=slot%16u;let c4=slot/16u;let k=k0+kk;let col0=n0+c4*4u;var bv=vec4<f32>(0.0);
    if(inTile && k<p.k){
     if(col0<p.c){bv.x=weight(col0*p.k+k,col0);}
     if(col0+1u<p.c){bv.y=weight((col0+1u)*p.k+k,col0+1u);}
     if(col0+2u<p.c){bv.z=weight((col0+2u)*p.k+k,col0+2u);}
     if(col0+3u<p.c){bv.w=weight((col0+3u)*p.k+k,col0+3u);}
    }
    tb[kk*33u+c4]=bv;
   }
  }`
        : `for(var e=0u;e<2u;e++){
   let slot=li+e*256u;let kk=slot%16u;let c4=slot/16u;let k=k0+kk;let col0=n0+c4*4u;
   var bv=vec4<${EB}>(0.0);
   if(inTile && k<p.k){
    let b0=col0*p.k+k;
    if(col0+3u<p.c){
     bv=vec4<${EB}>(${half ? "w[b0],w[b0+p.k],w[b0+2u*p.k],w[b0+3u*p.k]" : "bitcast<f32>(w[b0]),bitcast<f32>(w[b0+p.k]),bitcast<f32>(w[b0+2u*p.k]),bitcast<f32>(w[b0+3u*p.k])"});
    } else {
     if(col0<p.c){bv.x=${half ? "w[b0]" : "bitcast<f32>(w[b0])"};}
     if(col0+1u<p.c){bv.y=${half ? "w[b0+p.k]" : "bitcast<f32>(w[b0+p.k])"};}
     if(col0+2u<p.c){bv.z=${half ? "w[b0+2u*p.k]" : "bitcast<f32>(w[b0+2u*p.k])"};}
     if(col0+3u<p.c){bv.w=${half ? "w[b0+3u*p.k]" : "bitcast<f32>(w[b0+3u*p.k])"};}
    }
   }
   tb[kk*33u+c4]=bv;
  }`;
  const storeRows = [0, 1, 2, 3]
    .map(
      (i) => `{
   let m=m0+tm4*4u+${i}u;
   if(m<p.n){
    let r0=acc${i}0+b0;let r1=acc${i}1+b1;let rowOff=m*p.c;
    if(c0<p.c){y[rowOff+c0]=r0.x;} if(c0+1u<p.c){y[rowOff+c0+1u]=r0.y;} if(c0+2u<p.c){y[rowOff+c0+2u]=r0.z;} if(c0+3u<p.c){y[rowOff+c0+3u]=r0.w;}
    if(c1<p.c){y[rowOff+c1]=r1.x;} if(c1+1u<p.c){y[rowOff+c1+1u]=r1.y;} if(c1+2u<p.c){y[rowOff+c1+2u]=r1.z;} if(c1+3u<p.c){y[rowOff+c1+3u]=r1.w;}
   }
  }`,
    )
    .join("\n  ");
  return (
    (half ? "enable f16;\n" : "") +
    shader(
      [
        ["x"],
        ["w", "read", half ? "f16" : "u32"],
        ["bias"],
        ["scale"],
        ["y", "read_write"],
      ],
      `
 ${weightScalarFn(dtype, half)}
 var<workgroup> ta:array<vec4<f32>,272>;
 var<workgroup> tb:array<vec4<${EB}>,528>;
 @compute @workgroup_size(256)
 fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let tilesN=(p.c+127u)/128u;
  let tilesM=(p.n+63u)/64u;
  let flat=wid.x+wid.y*65535u;
  let inTile=flat<tilesM*tilesN;
  let tile=min(flat,tilesM*tilesN-1u);
  let m0=(tile/tilesN)*64u;
  let n0=(tile%tilesN)*128u;
  let tm4=li/16u;
  let tx=li%16u;
  var acc00=vec4<f32>(0.0);var acc01=vec4<f32>(0.0);
  var acc10=vec4<f32>(0.0);var acc11=vec4<f32>(0.0);
  var acc20=vec4<f32>(0.0);var acc21=vec4<f32>(0.0);
  var acc30=vec4<f32>(0.0);var acc31=vec4<f32>(0.0);
  let kkA=li%16u;
  let r4A=li/16u;
  let rowA0=m0+r4A*4u;
  let fullA=rowA0+3u<p.n;
  for(var k0=0u;k0<p.k;k0+=16u){
   {
    var av=vec4<f32>(0.0);
    if(inTile && k0+kkA<p.k){
     let aOff=rowA0*p.k+k0+kkA;
     if(fullA){av=vec4<f32>(x[aOff],x[aOff+p.k],x[aOff+2u*p.k],x[aOff+3u*p.k]);}
     else {
      if(rowA0<p.n){av.x=x[aOff];}
      if(rowA0+1u<p.n){av.y=x[aOff+p.k];}
      if(rowA0+2u<p.n){av.z=x[aOff+2u*p.k];}
      if(rowA0+3u<p.n){av.w=x[aOff+3u*p.k];}
     }
    }
    ta[kkA*17u+r4A]=av;
   }
   ${loadBBody}
   workgroupBarrier();
   ${unroll}
   workgroupBarrier();
  }
  if(!inTile){return;}
  let c0=n0+tx*4u;let c1=c0+64u;
  var b0=vec4<f32>(0.0);var b1=vec4<f32>(0.0);
  if(p.a!=0u){
   if(c0<p.c){b0.x=bias[c0];} if(c0+1u<p.c){b0.y=bias[c0+1u];} if(c0+2u<p.c){b0.z=bias[c0+2u];} if(c0+3u<p.c){b0.w=bias[c0+3u];}
   if(c1<p.c){b1.x=bias[c1];} if(c1+1u<p.c){b1.y=bias[c1+1u];} if(c1+2u<p.c){b1.z=bias[c1+2u];} if(c1+3u<p.c){b1.w=bias[c1+3u];}
  }
  ${storeRows}
 }`,
    )
  );
}

// Fused pair_norm affine + to_bias (512 -> 12) over pre-normalized unitPair [N^2, 512] with coalesced loads & sequential K accumulation.
export function pairBias(dtype, useF16) {
  const half = dtype === "F16" && useF16;
  return (
    (half ? "enable f16;\n" : "") +
    shader(
      [
        ["unitPair", "read", "vec4<f32>"],
        ["normW", "read", "vec4<f32>"],
        ["normB", "read", "vec4<f32>"],
        ["w", "read", half ? "f16" : "u32"],
        ["bias"],
        ["scale"],
        ["y", "read_write", "vec4<f32>"],
      ],
      `
 ${weightScalarFn(dtype, half)}
 var<workgroup> bw:array<vec4<f32>,96>;
 var<workgroup> nw:array<vec4<f32>,8>;
 var<workgroup> nb:array<vec4<f32>,8>;
 var<workgroup> sx:array<vec4<f32>,256>;
 @compute @workgroup_size(256)
 fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let wg=wid.x+wid.y*65535u;
  let rSub=li/8u;
  let kSub=li%8u;
  let row=wg*32u+rSub;
  let live=row<p.n;
  let sRow=min(row,p.n-1u);
  var acc=vec4<f32>(0.0);
  for(var k0=0u;k0<512u;k0+=32u){
   if(li<8u){
    let vIdx=(k0/4u)+li;
    nw[li]=normW[vIdx];
    nb[li]=normB[vIdx];
   }
   if(li<96u){
    let hVec=li/32u;let kk=li%32u;let h0=hVec*4u;let kIdx=k0+kk;
    bw[li]=vec4<f32>(weight(h0*512u+kIdx,h0),weight((h0+1u)*512u+kIdx,h0+1u),weight((h0+2u)*512u+kIdx,h0+2u),weight((h0+3u)*512u+kIdx,h0+3u));
   }
   workgroupBarrier();
   var x4=vec4<f32>(0.0);
   if(live){
    x4=unitPair[sRow*128u+(k0/4u)+kSub]*nw[kSub]+nb[kSub];
   }
   sx[li]=x4;
   workgroupBarrier();
   if(kSub<3u){
    let wOff=kSub*32u;
    let xOff=rSub*8u;
    for(var ks=0u;ks<8u;ks++){
     let xv=sx[xOff+ks];
     let kb=wOff+ks*4u;
     acc+=vec4<f32>(xv.x)*bw[kb];
     acc+=vec4<f32>(xv.y)*bw[kb+1u];
     acc+=vec4<f32>(xv.z)*bw[kb+2u];
     acc+=vec4<f32>(xv.w)*bw[kb+3u];
    }
   }
   workgroupBarrier();
  }
  if(live && kSub<3u){
   var bVec=vec4<f32>(0.0);
   if(p.a!=0u){
    let bo=kSub*4u;
    bVec=vec4<f32>(bias[bo],bias[bo+1u],bias[bo+2u],bias[bo+3u]);
   }
   y[row*3u+kSub]=acc+bVec;
  }
 }`,
    )
  );
}

// Fast 256-lane LayerNorm supporting optional strided input slice (p.b=stride, p.d=offset).
export const norm = shader(
  [["x"], ["w"], ["b"], ["y", "read_write"]],
  `
 var<workgroup> red:array<f32,256>;
 @compute @workgroup_size(256)
 fn main(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) g:vec3<u32>){
  let r=g.x+g.y*65535u;
  let live=r<p.n;
  let row=min(r,p.n-1u);
  let stride=select(p.c,p.b,p.b!=0u);
  let base=row*stride+p.d;
  var sum=0.0;
  for(var c=t;c<p.c;c+=256u){sum+=x[base+c];}
  red[t]=sum;workgroupBarrier();
  for(var s=128u;s>0u;s/=2u){if(t<s){red[t]+=red[t+s];}workgroupBarrier();}
  let mean=red[0]/f32(p.c);
  workgroupBarrier();
  var sq=0.0;
  for(var c=t;c<p.c;c+=256u){let d=x[base+c]-mean;sq+=d*d;}
  red[t]=sq;workgroupBarrier();
  for(var s=128u;s>0u;s/=2u){if(t<s){red[t]+=red[t+s];}workgroupBarrier();}
  let inv=inverseSqrt(red[0]/f32(p.c)+0.00001);
  if(!live){return;}
  let outBase=row*p.c;
  for(var c=t;c<p.c;c+=256u){
   var v=(x[base+c]-mean)*inv;
   if(p.a!=0u){v=v*w[c]+b[c];}
   y[outBase+c]=v;
  }
 }`,
);

// Fused parameter-free LayerNorm + AdaLN adapt.
export const normAdapt = shader(
  [["x"], ["g"], ["b"], ["y", "read_write"]],
  `
 fn sigmoid(v:f32)->f32{return 1.0/(1.0+exp(-v));}
 var<workgroup> red:array<f32,256>;
 @compute @workgroup_size(256)
 fn main(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let r=wid.x+wid.y*65535u;
  let live=r<p.n;
  let row=min(r,p.n-1u);
  let base=row*p.c;
  var sum=0.0;
  for(var c=t;c<p.c;c+=256u){sum+=x[base+c];}
  red[t]=sum;workgroupBarrier();
  for(var s=128u;s>0u;s/=2u){if(t<s){red[t]+=red[t+s];}workgroupBarrier();}
  let mean=red[0]/f32(p.c);
  workgroupBarrier();
  var sq=0.0;
  for(var c=t;c<p.c;c+=256u){let d=x[base+c]-mean;sq+=d*d;}
  red[t]=sq;workgroupBarrier();
  for(var s=128u;s>0u;s/=2u){if(t<s){red[t]+=red[t+s];}workgroupBarrier();}
  let inv=inverseSqrt(red[0]/f32(p.c)+0.00001);
  if(!live){return;}
  let gbOff=select(0u,select(0u,p.c,row>=10u),p.a==1u);
  for(var c=t;c<p.c;c+=256u){
   let idx=select((base+c)%p.m,gbOff+c,p.a==1u);
   y[base+c]=(x[base+c]-mean)*inv*sigmoid(g[idx])+b[idx];
  }
 }`,
);

export const element = shader(
  [["x"], ["a"], ["b"], ["y", "read_write"]],
  `
 fn sigmoid(v:f32)->f32{return 1.0/(1.0+exp(-v));}
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let i=(wid.x+wid.y*65535u)*256u+li;if(i>=p.n*p.c){return;}
  if(p.a==0u){y[i]=x[i]+a[i];}
  else if(p.a==1u){y[i]=x[i]*sigmoid(a[i%p.m])+b[i%p.m];}
  else if(p.a==2u){y[i]=x[i]+a[i]*sigmoid(b[i%p.m]);}
  else if(p.a==3u){let r=i/p.c;let c=i%p.c;let gate=x[r*p.c*2u+p.c+c];y[i]=x[r*p.c*2u+c]*(gate*sigmoid(gate));}
  else if(p.a==4u){let r=i/p.c;let c=i%p.c;let idx=select(0u,p.c,r>=10u)+c;y[i]=x[i]*sigmoid(a[idx])+b[idx];}
  else if(p.a==5u){let r=i/p.c;let c=i%p.c;let idx=select(0u,p.c,r>=10u)+c;y[i]=x[i]+a[i]*sigmoid(b[idx]);}
 }`,
);

export const rearrange = shader(
  [["x"], ["extra"], ["y", "read_write"]],
  `
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let i=(wid.x+wid.y*65535u)*256u+li;if(i>=p.n*p.c){return;}let r=i/p.c;let c=i%p.c;
  if(p.a==0u){y[i]=x[r*p.k+p.b+c];}
  else if(p.a==1u){if(i<p.k){y[i]=x[i];}else{y[i]=extra[i-p.k];}}
  else if(p.a==2u){if(r<10u){y[i]=0.0;}else{y[i]=x[c];}}
  else if(p.a==3u){if(r<10u){y[i]=extra[i];}else{y[i]=x[(r-10u)*p.c+c];}}
  else if(p.a==4u){let ii=r/p.k;let jj=r%p.k;if(ii<10u||jj<10u){y[i]=0.0;}else{y[i]=x[((ii-10u)*(p.k-10u)+jj-10u)*p.c+c];}}
  else if(p.a==5u){y[i]=x[(r+10u)*p.c+c];}
  else if(p.a==6u){if(r==0u){y[i]=0.0;}else{y[i]=x[c];}}
 }`,
);

export const time = shader(
  [["freq"], ["y", "read_write"]],
  `
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let i=(wid.x+wid.y*65535u)*256u+li;if(i<128u){let a=(p.t*2000.0)*freq[i];y[i]=sin(a);y[i+128u]=cos(a);}}`,
);

export const fold = shader(
  [["c"], ["a"], ["t"], ["y", "read_write"]],
  `
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let i=(wid.x+wid.y*65535u)*256u+li;if(i<256u){y[i]=c[p.a*256u+i];y[i+256u]=a[p.b*256u+i];y[i+512u]=t[p.d*256u+i];}}`,
);

export const features = shader(
  [["sc"], ["indices"], ["breaks"], ["divisor"], ["y", "read_write"]],
  `
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let i=(wid.x+wid.y*65535u)*256u+li;if(i>=p.n*132u){return;}let r=i/132u;let c=i%132u;
  if(c<128u){let index=indices[r]-indices[0]+1.0;let a=(index*3.141592653589793)/divisor[c%64u];if(c<64u){y[i]=sin(a);}else{y[i]=cos(a);}}
  else if(c<131u){y[i]=sc[r*3u+c-128u];}else{y[i]=breaks[r];}
 }`,
);

export const pairs = shader(
  [["x"], ["sc"], ["indices"], ["bins"], ["y", "read_write", "u32"]],
  `
 fn bucket(i:u32,j:u32,selfcond:bool)->u32{var sum=0.0;for(var k=0u;k<3u;k++){var d=x[i*3u+k]-x[j*3u+k];if(selfcond){d=sc[i*3u+k]-sc[j*3u+k];}sum+=d*d;}let dist=sqrt(sum);var count=63u;var offset=127u;if(selfcond){count=127u;offset=0u;}var b=0u;for(var k=0u;k<count;k++){if(dist>bins[offset+k]){b++;}}return b;}
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let r=(wid.x+wid.y*65535u)*256u+li;if(r>=p.n*p.n){return;}let i=r/p.n;let j=r%p.n;
  y[r*3u]=u32(clamp(ceil(indices[i]-indices[j]+62.5),0.0,126.0));y[r*3u+1u]=0xffffffffu;if(p.a==1u){y[r*3u+1u]=127u+bucket(i,j,true);}y[r*3u+2u]=255u+bucket(i,j,false);
 }`,
);

export function sparseLinear(dtype, useF16) {
  const half = dtype === "F16" && useF16;
  return (
    (half ? "enable f16;\n" : "") +
    shader(
      [
        ["indices", "read", "u32"],
        ["w", "read", half ? "f16" : "u32"],
        ["scale"],
        ["y", "read_write"],
      ],
      `
 ${weightScalarFn(dtype, half)}
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let i=(wid.x+wid.y*65535u)*256u+li;if(i>=p.n*p.c){return;}let r=i/p.c;let c=i%p.c;var s=0.0;for(var j=0u;j<3u;j++){let idx=indices[r*3u+j];if(idx!=0xffffffffu){s+=weight(c*p.k+idx,c);}}y[i]=s;}
 `,
    )
  );
}

// Register-tiled FlashAttention for pair-biased gated multi-head attention (12 heads x 64 dim).
export const gatedAttention = (() => {
  const perVec = (fn) => Array.from({ length: 16 }, (_, i) => fn(i)).join("\n  ");
  return shader(
    [
      ["q", "read", "vec4<f32>"],
      ["k", "read", "vec4<f32>"],
      ["v", "read", "vec4<f32>"],
      ["gate", "read", "vec4<f32>"],
      ["bias"],
      ["y", "read_write", "vec4<f32>"],
    ],
    `
 var<workgroup> sk:array<vec4<f32>,128>;
 var<workgroup> sv:array<vec4<f32>,128>;
 @compute @workgroup_size(64)
 fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let row=wid.x*64u+lane;
  let h=wid.y;
  let headBase=h*16u;
  let safeRow=min(row,p.n-1u);
  let qBase=safeRow*192u+headBase;
  let vStride=select(192u,p.a,p.a!=0u);
  let vOff=p.b;
  ${perVec((i) => `let q${i}=q[qBase+${i}u];var acc${i}=vec4<f32>(0.0);`)}
  var mx=-3.402823e38;
  var den=0.0;
  for(var j0=0u;j0<p.n;j0+=8u){
   for(var t=lane;t<128u;t+=64u){
    let col=j0+t/16u;
    let d=t%16u;
    if(col<p.n){
     sk[t]=k[col*192u+headBase+d];
     sv[t]=v[col*vStride+vOff+headBase+d];
    }
   }
   workgroupBarrier();
   let tileEnd=min(8u,p.n-j0);
   for(var jj=0u;jj<tileEnd;jj++){
    let col=j0+jj;
    let ko=jj*16u;
    var s=0.0;
    ${perVec((i) => `{let kv=sk[ko+${i}u];s+=q${i}.x*kv.x;s+=q${i}.y*kv.y;s+=q${i}.z*kv.z;s+=q${i}.w*kv.w;}`)}
    let score=s*0.125+bias[(safeRow*p.n+col)*12u+h];
    let nextMx=max(mx,score);
    let prev=exp(mx-nextMx);
    let wt=exp(score-nextMx);
    den=den*prev+wt;
    ${perVec((i) => `acc${i}=acc${i}*prev+wt*sv[ko+${i}u];`)}
    mx=nextMx;
   }
   workgroupBarrier();
  }
  if(row<p.n){
   let invDen=1.0/den;
   ${perVec((i) => `{let gv=gate[qBase+${i}u];y[qBase+${i}u]=(acc${i}*invDen)/(vec4<f32>(1.0)+exp(-gv));}`)}
  }
 }`,
  );
})();
