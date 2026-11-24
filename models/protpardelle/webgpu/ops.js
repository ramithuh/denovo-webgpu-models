// All activation arithmetic and learned operations execute as WGSL compute.
import { graphOps } from "./graph.js";
import { constants } from "../shared/constants.js";
const U = () => GPUBufferUsage;
const MAX_GRID = 65535;
const fold2 = (groups) => {
  const g = Math.max(1, groups);
  return g <= MAX_GRID ? [g, 1] : [MAX_GRID, Math.ceil(g / MAX_GRID)];
};
const scalar = (x) => `${Number(x).toExponential(12)}`;
const UNIFORM_SLAB_BYTES = 262144;
const UNIFORM_SLOT_BYTES = 256;
const SLOTS_PER_SLAB = UNIFORM_SLAB_BYTES / UNIFORM_SLOT_BYTES;

// Range-reduced polynomials avoid native trig error at large chain-gap angles.
const trig = `fn preciseSinCos(x:f32)->vec2<f32>{
let q=round(x*0.6366197723675814);var r=fma(-q,1.5703125,x);r=fma(-q,0.0004837512969970703125,r);r=fma(-q,0.00000007549789954891882,r);
let z=r*r;let sp=fma(z,fma(z,fma(z,fma(z,-0.00000002505210838544172,0.0000027557319223985893),-0.0001984126984126984),0.008333333333333333),-0.16666666666666666);let s=fma(r*z,sp,r);
let cp=fma(z,fma(z,fma(z,fma(z,-0.0000002755731922398589,0.0000248015873015873),-0.001388888888888889),0.041666666666666664),-0.5);let c=fma(z,cp,1.0);let quadrant=i32(q)&3;
if(quadrant==0){return vec2<f32>(s,c);}if(quadrant==1){return vec2<f32>(c,-s);}if(quadrant==2){return vec2<f32>(-s,-c);}return vec2<f32>(-c,s);}`;

const UNPACK_I8_FN = `fn unpack4i8(word:u32)->vec4<f32>{
let s=i32(word);
return vec4<f32>(f32((s<<24u)>>24u),f32((s<<16u)>>24u),f32((s<<8u)>>24u),f32(s>>24u));}`;

function linearTiledSource(dtype, large, hasF16) {
  const useF16 = Boolean(dtype === "F16" && hasF16);
  const EB = useF16 ? "f16" : "f32";
  const wType = dtype === "F32" ? "f32" : "u32";
  const head = `${useF16 ? "enable f16;\n" : ""}${dtype === "I8" ? UNPACK_I8_FN + "\n" : ""}
struct P { m:u32, n:u32, k:u32, hasBias:u32, tilesN:u32, totalTiles:u32, pad0:u32, pad1:u32 };
@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read> W:array<${wType}>;
@group(0) @binding(2) var<storage,read> B:array<f32>;
@group(0) @binding(3) var<storage,read> S:array<f32>;
@group(0) @binding(4) var<storage,read_write> Y:array<f32>;
@group(0) @binding(5) var<uniform> p:P;`;

  if (large) {
    // 64x128 tile (256 threads, 4x8 per thread, BK=16)
    const isI8 = dtype === "I8";
    const setupB =
      dtype === "F16"
        ? `let kkB=(li%8u)*2u;let r4B=li/8u;let rowB0=n0+r4B*4u;let sBnHalf=K/2u;let bBase0=(rowB0*K+kkB)/2u;let fullB=rowB0+3u<N;let tbWrite0=kkB*33u+r4B;let tbWrite1=(kkB+1u)*33u+r4B;`
        : isI8
          ? `let rowB=li/2u;let kHalf=(li%2u)*2u;let gRowB=n0+rowB;let validRowB=gRowB<N;let scB=select(0.0,S[min(gRowB,N-1u)],validRowB);let sBnQuad=K/4u;let bRowBase=gRowB*sBnQuad+kHalf;let tbRowBase=rowB*17u+kHalf*4u;`
          : "";
    const loopB =
      dtype === "F16"
        ? `{
      var lo=vec4<${EB}>(0.0);var hi=vec4<${EB}>(0.0);
      if(inTile&&k0+kkB<K){
        let bOff=bBase0+(k0/2u);
        if(fullB){
          let p0=unpack2x16float(W[bOff]);let p1=unpack2x16float(W[bOff+sBnHalf]);
          let p2=unpack2x16float(W[bOff+2u*sBnHalf]);let p3=unpack2x16float(W[bOff+3u*sBnHalf]);
          lo=vec4<${EB}>(${EB}(p0.x),${EB}(p1.x),${EB}(p2.x),${EB}(p3.x));
          hi=vec4<${EB}>(${EB}(p0.y),${EB}(p1.y),${EB}(p2.y),${EB}(p3.y));
        }else{
          if(rowB0<N){let p0=unpack2x16float(W[bOff]);lo.x=${EB}(p0.x);hi.x=${EB}(p0.y);}
          if(rowB0+1u<N){let p1=unpack2x16float(W[bOff+sBnHalf]);lo.y=${EB}(p1.x);hi.y=${EB}(p1.y);}
          if(rowB0+2u<N){let p2=unpack2x16float(W[bOff+2u*sBnHalf]);lo.z=${EB}(p2.x);hi.z=${EB}(p2.y);}
          if(rowB0+3u<N){let p3=unpack2x16float(W[bOff+3u*sBnHalf]);lo.w=${EB}(p3.x);hi.w=${EB}(p3.y);}
        }
      }
      tb[tbWrite0]=lo;tb[tbWrite1]=hi;
    }`
        : isI8
          ? `{
      var q0=vec4<f32>(0.0);var q1=vec4<f32>(0.0);
      if(inTile&&validRowB){
        let k0q=k0/4u;
        if(k0+kHalf*4u<K){q0=unpack4i8(W[bRowBase+k0q])*scB;}
        if(k0+(kHalf+1u)*4u<K){q1=unpack4i8(W[bRowBase+k0q+1u])*scB;}
      }
      tb[tbRowBase+0u]=q0.x;tb[tbRowBase+1u]=q0.y;tb[tbRowBase+2u]=q0.z;tb[tbRowBase+3u]=q0.w;
      tb[tbRowBase+4u]=q1.x;tb[tbRowBase+5u]=q1.y;tb[tbRowBase+6u]=q1.z;tb[tbRowBase+7u]=q1.w;
    }`
          : `for(var e=0u;e<2u;e++){
      let slot=li+e*256u;let kk=slot%16u;let r4=slot/16u;let k=k0+kk;
      var v=vec4<f32>(0.0);
      if(inTile&&k<K){
        let row0=n0+r4*4u;let bAddr0=row0*K+k;
        if(row0+3u<N){v=vec4<f32>(W[bAddr0],W[bAddr0+K],W[bAddr0+2u*K],W[bAddr0+3u*K]);}
        else{
          if(row0<N){v.x=W[bAddr0];}if(row0+1u<N){v.y=W[bAddr0+K];}
          if(row0+2u<N){v.z=W[bAddr0+2u*K];}if(row0+3u<N){v.w=W[bAddr0+3u*K];}
        }
      }
      tb[kk*33u+r4]=v;
    }`;
    const fmas = Array.from(
      { length: 16 },
      (_, kk) => `{
      let av=ta[${kk * 17}u+tm4];
      ${
        isI8
          ? `let bv0=vec4<f32>(tb[(tn0+0u)*17u+${kk}u],tb[(tn0+1u)*17u+${kk}u],tb[(tn0+2u)*17u+${kk}u],tb[(tn0+3u)*17u+${kk}u]);
      let bv1=vec4<f32>(tb[(tn1+0u)*17u+${kk}u],tb[(tn1+1u)*17u+${kk}u],tb[(tn1+2u)*17u+${kk}u],tb[(tn1+3u)*17u+${kk}u]);`
          : `let bv0=vec4<f32>(tb[${kk * 33}u+tx]);let bv1=vec4<f32>(tb[${kk * 33 + 16}u+tx]);`
      }
      acc_0_0=fma(vec4<f32>(av.x),bv0,acc_0_0);acc_0_1=fma(vec4<f32>(av.x),bv1,acc_0_1);
      acc_1_0=fma(vec4<f32>(av.y),bv0,acc_1_0);acc_1_1=fma(vec4<f32>(av.y),bv1,acc_1_1);
      acc_2_0=fma(vec4<f32>(av.z),bv0,acc_2_0);acc_2_1=fma(vec4<f32>(av.z),bv1,acc_2_1);
      acc_3_0=fma(vec4<f32>(av.w),bv0,acc_3_0);acc_3_1=fma(vec4<f32>(av.w),bv1,acc_3_1);
    }`,
    ).join("\n    ");
    const stores = [0, 1, 2, 3]
      .map(
        (i) => `{
    let m=m0+tm+${i}u;
    if(m<M){
      let r0=acc_${i}_0;let r1=acc_${i}_1;let rowOut=m*N;
      ${[0, 1, 2, 3, 4, 5, 6, 7]
        .map((j) => {
          const nExpr = j < 4 ? `n0+tn0+${j}u` : `n0+tn1+${j - 4}u`;
          const valExpr = j < 4 ? `r0.${"xyzw"[j]}` : `r1.${"xyzw"[j - 4]}`;
          return `{let n=${nExpr};if(n<N){Y[rowOut+n]=${valExpr};}}`;
        })
        .join("")}
    }
  }`,
      )
      .join("");
    return `${head}
var<workgroup> ta:array<vec4<f32>,272>;
var<workgroup> tb:array<${isI8 ? "f32,2176" : `vec4<${EB}>,528`}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) li:u32){
  _ = B[0]; _ = S[0];
  let M=p.m;let N=p.n;let K=p.k;
  let flat=wid.x+wid.y*65535u;let inTile=flat<p.totalTiles;let tile=min(flat,p.totalTiles-1u);
  let m0=(tile/p.tilesN)*64u;let n0=(tile%p.tilesN)*128u;
  let tm4=li/16u;let tx=li%16u;let tm=tm4*4u;let tn0=tx*4u;let tn1=tn0+64u;
  var bInit0=vec4<f32>(0.0);var bInit1=vec4<f32>(0.0);
  if(p.hasBias!=0u){
    let c0=n0+tn0;let c1=n0+tn1;
    if(c0<N){bInit0.x=B[c0];}if(c0+1u<N){bInit0.y=B[c0+1u];}if(c0+2u<N){bInit0.z=B[c0+2u];}if(c0+3u<N){bInit0.w=B[c0+3u];}
    if(c1<N){bInit1.x=B[c1];}if(c1+1u<N){bInit1.y=B[c1+1u];}if(c1+2u<N){bInit1.z=B[c1+2u];}if(c1+3u<N){bInit1.w=B[c1+3u];}
  }
  var acc_0_0=bInit0;var acc_0_1=bInit1;
  var acc_1_0=bInit0;var acc_1_1=bInit1;
  var acc_2_0=bInit0;var acc_2_1=bInit1;
  var acc_3_0=bInit0;var acc_3_1=bInit1;
  let kkA=li%16u;let r4A=li/16u;let rowA0=m0+r4A*4u;let aBase0=rowA0*K+kkA;let fullA=rowA0+3u<M;let taWrite=kkA*17u+r4A;
  ${setupB}
  for(var k0=0u;k0<K;k0+=16u){
    {
      var v=vec4<f32>(0.0);
      if(inTile&&k0+kkA<K){
        let aOff=aBase0+k0;
        if(fullA){v=vec4<f32>(X[aOff],X[aOff+K],X[aOff+2u*K],X[aOff+3u*K]);}
        else{
          if(rowA0<M){v.x=X[aOff];}if(rowA0+1u<M){v.y=X[aOff+K];}
          if(rowA0+2u<M){v.z=X[aOff+2u*K];}if(rowA0+3u<M){v.w=X[aOff+3u*K];}
        }
      }
      ta[taWrite]=v;
    }
    ${loopB}
    workgroupBarrier();
    ${fmas}
    workgroupBarrier();
  }
  if(!inTile){return;}
  ${stores}
}`;
  }

  // 32x64 tile (256 threads, 2x4 per thread, BK=16)
  const isI8 = dtype === "I8";
  const setupBSmall =
    dtype === "F16"
      ? `let kkB=(li%8u)*2u;let r4B=li/8u;let rowB0=n0+r4B*4u;let sBnHalf=K/2u;let bBase0=(rowB0*K+kkB)/2u;let fullB=rowB0+3u<N;let tbWrite0=kkB*17u+r4B;let tbWrite1=(kkB+1u)*17u+r4B;`
      : isI8
        ? `let rowB=li/4u;let kQuad=li%4u;let gRowB=n0+rowB;let validRowB=gRowB<N;let scB=select(0.0,S[min(gRowB,N-1u)],validRowB);let sBnQuad=K/4u;let bRowBase=gRowB*sBnQuad+kQuad;let tbRowBase=rowB*17u+kQuad*4u;`
        : `let kkB=li%16u;let r4B=li/16u;let rowB0=n0+r4B*4u;let bBase0=rowB0*K+kkB;let fullB=rowB0+3u<N;let tbWrite=kkB*17u+r4B;`;
  const loopBSmall =
    dtype === "F16"
      ? `if(li<128u){
      var lo=vec4<${EB}>(0.0);var hi=vec4<${EB}>(0.0);
      if(inTile&&k0+kkB<K){
        let bOff=bBase0+(k0/2u);
        if(fullB){
          let p0=unpack2x16float(W[bOff]);let p1=unpack2x16float(W[bOff+sBnHalf]);
          let p2=unpack2x16float(W[bOff+2u*sBnHalf]);let p3=unpack2x16float(W[bOff+3u*sBnHalf]);
          lo=vec4<${EB}>(${EB}(p0.x),${EB}(p1.x),${EB}(p2.x),${EB}(p3.x));
          hi=vec4<${EB}>(${EB}(p0.y),${EB}(p1.y),${EB}(p2.y),${EB}(p3.y));
        }else{
          if(rowB0<N){let p0=unpack2x16float(W[bOff]);lo.x=${EB}(p0.x);hi.x=${EB}(p0.y);}
          if(rowB0+1u<N){let p1=unpack2x16float(W[bOff+sBnHalf]);lo.y=${EB}(p1.x);hi.y=${EB}(p1.y);}
          if(rowB0+2u<N){let p2=unpack2x16float(W[bOff+2u*sBnHalf]);lo.z=${EB}(p2.x);hi.z=${EB}(p2.y);}
          if(rowB0+3u<N){let p3=unpack2x16float(W[bOff+3u*sBnHalf]);lo.w=${EB}(p3.x);hi.w=${EB}(p3.y);}
        }
      }
      tb[tbWrite0]=lo;tb[tbWrite1]=hi;
    }`
      : isI8
        ? `{
      var q=vec4<f32>(0.0);
      if(inTile&&validRowB&&k0+kQuad*4u<K){
        q=unpack4i8(W[bRowBase+k0/4u])*scB;
      }
      tb[tbRowBase+0u]=q.x;tb[tbRowBase+1u]=q.y;tb[tbRowBase+2u]=q.z;tb[tbRowBase+3u]=q.w;
    }`
        : `{
      var v=vec4<f32>(0.0);
      if(inTile&&k0+kkB<K){
        let bOff=bBase0+k0;
        if(fullB){v=vec4<f32>(W[bOff],W[bOff+K],W[bOff+2u*K],W[bOff+3u*K]);}
        else{
          if(rowB0<N){v.x=W[bOff];}if(rowB0+1u<N){v.y=W[bOff+K];}
          if(rowB0+2u<N){v.z=W[bOff+2u*K];}if(rowB0+3u<N){v.w=W[bOff+3u*K];}
        }
      }
      tb[tbWrite]=v;
    }`;
  const fmasSmall = Array.from(
    { length: 16 },
    (_, kk) => `{
      let av4=ta[${kk * 9}u+r4Read];let av=select(av4.xy,av4.zw,hiPair);
      ${
        isI8
          ? `let bv=vec4<f32>(tb[(tn0+0u)*17u+${kk}u],tb[(tn0+1u)*17u+${kk}u],tb[(tn0+2u)*17u+${kk}u],tb[(tn0+3u)*17u+${kk}u]);`
          : `let bv=vec4<f32>(tb[${kk * 17}u+tx]);`
      }
      acc0=fma(vec4<f32>(av.x),bv,acc0);acc1=fma(vec4<f32>(av.y),bv,acc1);
    }`,
  ).join("\n    ");
  const storesSmall = [0, 1]
    .map(
      (i) => `{
    let m=m0+tm+${i}u;
    if(m<M){
      let r=acc${i};let rowOut=m*N;
      ${[0, 1, 2, 3]
        .map((j) => `{let n=n0+tn0+${j}u;if(n<N){Y[rowOut+n]=r.${"xyzw"[j]};}}`)
        .join("")}
    }
  }`,
    )
    .join("");
  return `${head}
var<workgroup> ta:array<vec4<f32>,144>;
var<workgroup> tb:array<${isI8 ? "f32,1088" : `vec4<${EB}>,272`}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) li:u32){
  _ = B[0]; _ = S[0];
  let M=p.m;let N=p.n;let K=p.k;
  let flat=wid.x+wid.y*65535u;let inTile=flat<p.totalTiles;let tile=min(flat,p.totalTiles-1u);
  let m0=(tile/p.tilesN)*32u;let n0=(tile%p.tilesN)*64u;
  let tm4=li/16u;let tx=li%16u;let tm=tm4*2u;let tn0=tx*4u;let r4Read=tm4/2u;let hiPair=(tm4&1u)!=0u;
  var bInit=vec4<f32>(0.0);
  if(p.hasBias!=0u){
    let c0=n0+tn0;
    if(c0<N){bInit.x=B[c0];}if(c0+1u<N){bInit.y=B[c0+1u];}if(c0+2u<N){bInit.z=B[c0+2u];}if(c0+3u<N){bInit.w=B[c0+3u];}
  }
  var acc0=bInit;var acc1=bInit;
  let kkA=li%16u;let r4A=li/16u;let rowA0=m0+r4A*4u;let aBase0=rowA0*K+kkA;let fullA=rowA0+3u<M;let taWrite=kkA*9u+r4A;
  ${setupBSmall}
  for(var k0=0u;k0<K;k0+=16u){
    if(li<128u){
      var v=vec4<f32>(0.0);
      if(inTile&&k0+kkA<K){
        let aOff=aBase0+k0;
        if(fullA){v=vec4<f32>(X[aOff],X[aOff+K],X[aOff+2u*K],X[aOff+3u*K]);}
        else{
          if(rowA0<M){v.x=X[aOff];}if(rowA0+1u<M){v.y=X[aOff+K];}
          if(rowA0+2u<M){v.z=X[aOff+2u*K];}if(rowA0+3u<M){v.w=X[aOff+3u*K];}
        }
      }
      ta[taWrite]=v;
    }
    ${loopBSmall}
    workgroupBarrier();
    ${fmasSmall}
    workgroupBarrier();
  }
  if(!inTile){return;}
  ${storesSmall}
}`;
}

function linearOddKSource(dtype) {
  const wType = dtype === "F32" ? "f32" : "u32";
  const load =
    dtype === "F32"
      ? "return W[idx];"
      : dtype === "F16"
        ? "let pair=unpack2x16float(W[idx/2u]);return pair[idx%2u];"
        : "let bits=(W[idx/4u]>>((idx%4u)*8u))&255u;let value=i32(bits<<24u)>>24;return f32(value)*sc;";
  return `
struct P { m:u32, n:u32, k:u32, hasBias:u32, tilesN:u32, totalTiles:u32, pad0:u32, pad1:u32 };
@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read> W:array<${wType}>;
@group(0) @binding(2) var<storage,read> B:array<f32>;
@group(0) @binding(3) var<storage,read> S:array<f32>;
@group(0) @binding(4) var<storage,read_write> Y:array<f32>;
@group(0) @binding(5) var<uniform> p:P;
var<workgroup> tileA:array<f32,544>;
var<workgroup> tileB:array<f32,528>;
fn weight(idx:u32, sc:f32)->f32 { ${load} }
@compute @workgroup_size(16,16)
fn main(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wg:vec3<u32>){
  _ = B[0]; _ = S[0];
  let M=p.m;let N=p.n;let K=p.k;
  let flat=wg.x+wg.y*65535u;let inTile=flat<p.totalTiles;let tile=min(flat,p.totalTiles-1u);
  let m0=(tile/p.tilesN)*32u;let n0=(tile%p.tilesN)*32u;
  let lx=lid.x;let ly=lid.y;
  let c0=n0+ly;let c1=c0+16u;
  let sc0=select(1.0,S[min(c0,N-1u)],${dtype === "I8" ? "c0<N" : "false"});
  let sc1=select(1.0,S[min(c1,N-1u)],${dtype === "I8" ? "c1<N" : "false"});
  let colOut0=n0+lx*2u;
  let bi0=select(0.0,B[min(colOut0,N-1u)],p.hasBias!=0u&&colOut0<N);
  let bi1=select(0.0,B[min(colOut0+1u,N-1u)],p.hasBias!=0u&&colOut0+1u<N);
  var acc00=bi0;var acc01=bi1;var acc10=bi0;var acc11=bi1;
  for(var k0=0u;k0<K;k0+=16u){
    let r0=m0+ly;let r1=r0+16u;let kA=k0+lx;
    tileA[ly*17u+lx]=select(0.0,X[min(r0,M-1u)*K+min(kA,K-1u)],inTile&&r0<M&&kA<K);
    tileA[(ly+16u)*17u+lx]=select(0.0,X[min(r1,M-1u)*K+min(kA,K-1u)],inTile&&r1<M&&kA<K);
    let kB=k0+lx;
    var w0=0.0;var w1=0.0;
    if(inTile&&c0<N&&kB<K){w0=weight(c0*K+kB,sc0);}
    if(inTile&&c1<N&&kB<K){w1=weight(c1*K+kB,sc1);}
    tileB[lx*33u+ly]=w0;tileB[lx*33u+ly+16u]=w1;
    workgroupBarrier();
    for(var kk=0u;kk<16u;kk++){
      let a0=tileA[(ly*2u)*17u+kk];let a1=tileA[(ly*2u+1u)*17u+kk];
      let b0=tileB[kk*33u+(lx*2u)];let b1=tileB[kk*33u+(lx*2u+1u)];
      acc00=fma(a0,b0,acc00);acc01=fma(a0,b1,acc01);
      acc10=fma(a1,b0,acc10);acc11=fma(a1,b1,acc11);
    }
    workgroupBarrier();
  }
  if(!inTile){return;}
  let rOut0=m0+ly*2u;
  if(rOut0<M&&colOut0<N){Y[rOut0*N+colOut0]=acc00;}
  if(rOut0<M&&colOut0+1u<N){Y[rOut0*N+colOut0+1u]=acc01;}
  if(rOut0+1u<M&&colOut0<N){Y[(rOut0+1u)*N+colOut0]=acc10;}
  if(rOut0+1u<M&&colOut0+1u<N){Y[(rOut0+1u)*N+colOut0+1u]=acc11;}
}`;
}

const NORM_SOURCE = `
struct P { rows:u32, d:u32, pad0:u32, pad1:u32 };
@group(0) @binding(0) var<storage,read> b0:array<f32>;
@group(0) @binding(1) var<storage,read> b1:array<f32>;
@group(0) @binding(2) var<storage,read> b2:array<f32>;
@group(0) @binding(3) var<storage,read_write> b3:array<f32>;
@group(0) @binding(4) var<uniform> p:P;
var<workgroup> rowTile:array<f32,2048>;
var<workgroup> wm:array<vec4<f32>,64>;
var<workgroup> ws:array<vec4<f32>,64>;
var<workgroup> wc:array<u32,64>;
var<workgroup> stats:array<f32,2>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) li:u32){
  let r=wg.x+wg.y*65535u;let live=r<p.rows;let row=min(r,p.rows-1u);let d=p.d;let base=row*d;
  if(d<=2048u&&(d&3u)==0u){
    for(var j=li;j<d;j+=64u){rowTile[j]=b0[base+j];}
    workgroupBarrier();
    let n4=d/4u;let chunks=(n4+15u)/16u;
    var m=vec4<f32>(0.0);var s=vec4<f32>(0.0);var cnt=0u;
    if(li<chunks){
      cnt=min(16u,n4-li*16u);
      for(var j=0u;j<cnt;j++){
        let off=li*64u+j*4u;
        let value=vec4<f32>(rowTile[off],rowTile[off+1u],rowTile[off+2u],rowTile[off+3u]);
        let delta=value-m;
        m=fma(vec4<f32>(1.0/f32(j+1u)),delta,m);
        s=fma(delta,value-m,s);
      }
    }
    wm[li]=m;ws[li]=s;wc[li]=cnt;
    workgroupBarrier();
    if(li==0u){
      var mc:array<u32,7>;var mm:array<vec4<f32>,7>;var ms:array<vec4<f32>,7>;
      for(var lv=0u;lv<7u;lv++){mc[lv]=0u;mm[lv]=vec4<f32>(0.0);ms[lv]=vec4<f32>(0.0);}
      for(var ch=0u;ch<chunks;ch++){
        let cIn=wc[ch];let mIn=wm[ch];let sIn=ws[ch];
        {
          let old=mc[0];let total=old+cIn;let factor=f32(cIn)/f32(max(total,1u));
          let delta=mIn-mm[0];let cd=delta*factor;
          ms[0]=fma(delta*f32(old),cd,ms[0]+sIn);mm[0]+=cd;mc[0]=total;
        }
        var bits=ch+1u;
        for(var lv=1u;lv<7u&&(bits&1u)==0u;lv++){
          let cPrev=mc[lv-1u];let mPrev=mm[lv-1u];let sPrev=ms[lv-1u];
          let old=mc[lv];let total=old+cPrev;let factor=f32(cPrev)/f32(max(total,1u));
          let delta=mPrev-mm[lv];let cd=delta*factor;
          ms[lv]=fma(delta*f32(old),cd,ms[lv]+sPrev);mm[lv]+=cd;mc[lv]=total;
          mc[lv-1u]=0u;mm[lv-1u]=vec4<f32>(0.0);ms[lv-1u]=vec4<f32>(0.0);bits>>=1u;
        }
      }
      for(var lv=1u;lv<7u;lv++){
        let cLv=mc[lv];
        if(cLv>0u){
          let old=mc[0];let total=old+cLv;let factor=f32(cLv)/f32(max(total,1u));
          let delta=mm[lv]-mm[0];let cd=delta*factor;
          ms[0]=fma(delta*f32(old),cd,ms[0]+ms[lv]);mm[0]+=cd;mc[0]=total;
        }
      }
      var meanAcc=0.0;var sqAcc=0.0;var countAcc=0u;
      for(var lane=0u;lane<4u;lane++){
        let total=countAcc+n4;let factor=f32(n4)/f32(total);let delta=mm[0][lane]-meanAcc;
        meanAcc=fma(factor,delta,meanAcc);sqAcc+=ms[0][lane]+((delta*delta)*factor)*f32(countAcc);countAcc=total;
      }
      stats[0]=meanAcc;stats[1]=inverseSqrt(sqAcc/f32(d)+0.00001);
    }
    workgroupBarrier();
    if(!live){return;}
    let mean=stats[0];let inv=stats[1];
    for(var j=li;j<d;j+=64u){b3[base+j]=(rowTile[j]-mean)*inv*b1[j]+b2[j];}
    return;
  }
  var s=0.0;for(var j=li;j<d;j+=64u){s+=b0[base+j];}
  rowTile[li]=s;workgroupBarrier();
  if(li<32u){rowTile[li]+=rowTile[li+32u];}workgroupBarrier();
  if(li<16u){rowTile[li]+=rowTile[li+16u];}workgroupBarrier();
  if(li<8u){rowTile[li]+=rowTile[li+8u];}workgroupBarrier();
  if(li==0u){stats[0]=(((rowTile[0]+rowTile[4])+(rowTile[2]+rowTile[6]))+((rowTile[1]+rowTile[5])+(rowTile[3]+rowTile[7])))/f32(d);}
  workgroupBarrier();let mean=stats[0];workgroupBarrier();
  var q=0.0;for(var j=li;j<d;j+=64u){let v=b0[base+j]-mean;q+=v*v;}
  rowTile[li]=q;workgroupBarrier();
  if(li<32u){rowTile[li]+=rowTile[li+32u];}workgroupBarrier();
  if(li<16u){rowTile[li]+=rowTile[li+16u];}workgroupBarrier();
  if(li<8u){rowTile[li]+=rowTile[li+8u];}workgroupBarrier();
  if(li==0u){stats[1]=inverseSqrt((((rowTile[0]+rowTile[4])+(rowTile[2]+rowTile[6]))+((rowTile[1]+rowTile[5])+(rowTile[3]+rowTile[7])))/f32(d)+0.00001);}
  workgroupBarrier();if(!live){return;}
  let inv=stats[1];
  for(var j=li;j<d;j+=64u){b3[base+j]=(b0[base+j]-mean)*inv*b1[j]+b2[j];}
}`;

const SOFTMAX_SOURCE = `
struct P { rows:u32, n:u32, pad0:u32, pad1:u32 };
@group(0) @binding(0) var<storage,read> b0:array<f32>;
@group(0) @binding(1) var<storage,read_write> b1:array<f32>;
@group(0) @binding(2) var<uniform> p:P;
var<workgroup> red:array<f32,64>;
var<workgroup> smTile:array<f32,2048>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) li:u32){
  let r=wg.x+wg.y*65535u;let live=r<p.rows;let row=min(r,p.rows-1u);let n=p.n;let base=row*n;
  var mx=-3.402823e38;
  if(n<=2048u){
    for(var j=li;j<n;j+=64u){let v=b0[base+j];smTile[j]=v;mx=max(mx,v);}
  }else{
    for(var j=li;j<n;j+=64u){mx=max(mx,b0[base+j]);}
  }
  red[li]=mx;workgroupBarrier();
  if(li<32u){red[li]=max(red[li],red[li+32u]);}workgroupBarrier();
  if(li<16u){red[li]=max(red[li],red[li+16u]);}workgroupBarrier();
  if(li<8u){red[li]=max(red[li],red[li+8u]);}workgroupBarrier();
  if(li==0u){red[0]=max(max(max(red[0],red[4]),max(red[2],red[6])),max(max(red[1],red[5]),max(red[3],red[7])));}
  workgroupBarrier();let top=red[0];workgroupBarrier();
  if(n<=64u){
    if(li==0u){
      var total=0.0;
      for(var j=0u;j<n;j++){let e=exp(smTile[j]-top);smTile[j]=e;total+=e;}
      red[0]=total;
    }
    workgroupBarrier();
    let total=red[0];
    if(!live){return;}
    if(li<n){b1[base+li]=smTile[li]/total;}
    return;
  }
  var s=0.0;
  if(n<=2048u){
    for(var j=li;j<n;j+=64u){let e=exp(smTile[j]-top);smTile[j]=e;s+=e;}
  }else{
    for(var j=li;j<n;j+=64u){s+=exp(b0[base+j]-top);}
  }
  red[li]=s;workgroupBarrier();
  if(li<32u){red[li]+=red[li+32u];}workgroupBarrier();
  if(li<16u){red[li]+=red[li+16u];}workgroupBarrier();
  if(li<8u){red[li]+=red[li+8u];}workgroupBarrier();
  if(li==0u){red[0]=((red[0]+red[4])+(red[2]+red[6]))+((red[1]+red[5])+(red[3]+red[7]));}
  workgroupBarrier();let total=red[0];
  if(!live){return;}
  if(n<=2048u){
    for(var j=li;j<n;j+=64u){b1[base+j]=smTile[j]/total;}
  }else{
    for(var j=li;j<n;j+=64u){b1[base+j]=exp(b0[base+j]-top)/total;}
  }
}`;

export class GPUOps {
  constructor(device, weights, opts = {}) {
    this.device = device;
    this.hasF16 = Boolean(opts.hasF16 && device.features?.has?.("shader-f16"));
    this.weights = {};
    this.buffers = new Map();
    this.pipelines = new Map();
    this.pool = new Map();
    this.pinned = new Set();
    this.pass = null;
    this.stepCounter = 0;
    this.uniformSlabs = [];
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
    try {
      for (const [key, value] of Object.entries(weights)) {
        const buffer = this.allocate(value.data.byteLength, true);
        device.queue.writeBuffer(
          buffer,
          0,
          new Uint8Array(
            value.data.buffer,
            value.data.byteOffset,
            value.data.byteLength,
          ),
        );
        let scale;
        if (value.scale) {
          scale = this.allocate(value.scale.byteLength, true);
          device.queue.writeBuffer(scale, 0, value.scale);
        }
        this.weights[key] = { ...value, data: undefined, buffer, scale };
      }
      this.zero = this.allocate(4, true);
    } catch (error) {
      this.destroy();
      throw error;
    }
  }
  allocate(bytes, resident = false, forUpload = false) {
    const size = Math.max(4, Math.ceil(bytes / 4) * 4);
    if (
      size > this.device.limits.maxBufferSize ||
      size > this.device.limits.maxStorageBufferBindingSize
    )
      throw new Error(
        `WebGPU buffer admission failed: ${size} bytes exceeds device storage limit`,
      );
    if (!resident && this.pool.get(size)?.length) {
      const bucket = this.pool.get(size);
      const idx = forUpload
        ? bucket.findIndex((b) => !this.buffers.get(b).dirtyInEncoder)
        : bucket.length - 1;
      if (idx >= 0) {
        const [buffer] = bucket.splice(idx, 1);
        const info = this.buffers.get(buffer);
        info.pooled = false;
        info.uploaded = forUpload;
        info.dirtyInEncoder = true;
        return buffer;
      }
    }
    const buffer = this.device.createBuffer({
      size,
      usage: U().STORAGE | U().COPY_DST | U().COPY_SRC,
    });
    this.buffers.set(buffer, {
      size,
      resident,
      pooled: false,
      uploaded: forUpload,
      dirtyInEncoder: !resident,
    });
    return buffer;
  }
  allocUniform(words) {
    if (this.uniformSlotIdx >= SLOTS_PER_SLAB) {
      this.uniformSlabIdx++;
      this.uniformSlotIdx = 0;
    }
    while (this.uniformSlabs.length <= this.uniformSlabIdx) {
      const buffer = this.device.createBuffer({
        size: UNIFORM_SLAB_BYTES,
        usage: U().UNIFORM | U().COPY_DST,
      });
      this.buffers.set(buffer, {
        size: UNIFORM_SLAB_BYTES,
        resident: false,
        pooled: false,
        uniformSlab: true,
      });
      this.uniformSlabs.push({
        buffer,
        cpu: new Uint32Array(UNIFORM_SLAB_BYTES / 4),
        usedWords: 0,
      });
    }
    const slab = this.uniformSlabs[this.uniformSlabIdx];
    const byteOffset = this.uniformSlotIdx * UNIFORM_SLOT_BYTES;
    const wordOffset = byteOffset >>> 2;
    slab.cpu.set(words, wordOffset);
    slab.usedWords = wordOffset + words.length;
    this.uniformSlotIdx++;
    return { buffer: slab.buffer, offset: byteOffset, size: words.byteLength };
  }
  flushUniforms() {
    const limit = Math.min(this.uniformSlabIdx + 1, this.uniformSlabs.length);
    for (let i = 0; i < limit; i++) {
      const slab = this.uniformSlabs[i];
      if (slab.usedWords > 0) {
        this.device.queue.writeBuffer(
          slab.buffer,
          0,
          slab.cpu.buffer,
          0,
          slab.usedWords * 4,
        );
        slab.usedWords = 0;
      }
    }
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
  }
  endPass() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }
  tensor(rows, cols, data) {
    const buffer = this.allocate(rows * cols * 4, false, Boolean(data));
    if (data) {
      this.device.queue.writeBuffer(
        buffer,
        0,
        data instanceof Float32Array ? data : Float32Array.from(data),
      );
    }
    return { buffer, rows, cols };
  }
  start() {
    this.endPass();
    this.encoder = this.device.createCommandEncoder();
    this.inputs = new Map();
    this.stepCounter = 0;
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
  }
  attach(weights) {
    for (const [key, value] of Object.entries(weights)) {
      if (!key.startsWith("mpnn_model.") || this.weights[key]) continue;
      const buffer = this.allocate(value.data.byteLength, true);
      this.device.queue.writeBuffer(
        buffer,
        0,
        new Uint8Array(
          value.data.buffer,
          value.data.byteOffset,
          value.data.byteLength,
        ),
      );
      let scale;
      if (value.scale) {
        scale = this.allocate(value.scale.byteLength, true);
        this.device.queue.writeBuffer(scale, 0, value.scale);
      }
      this.weights[key] = { ...value, data: undefined, buffer, scale };
    }
  }
  async stepBoundary(keep) {
    this.endPass();
    this.flushUniforms();
    this.device.queue.submit([this.encoder.finish()]);
    this.stepCounter++;
    if (this.stepCounter % 16 === 0) {
      await this.device.queue.onSubmittedWorkDone();
    }
    const retained = new Set(keep.filter(Boolean).map((t) => t.buffer));
    for (const buffer of this.traceBuffers ?? []) retained.add(buffer);
    for (const [buffer, info] of this.buffers) {
      info.dirtyInEncoder = false;
      if (
        !info.resident &&
        !info.uniformSlab &&
        !info.pooled &&
        !retained.has(buffer)
      ) {
        info.pooled = true;
        info.uploaded = false;
        if (!this.pool.has(info.size)) this.pool.set(info.size, []);
        this.pool.get(info.size).push(buffer);
      }
    }
    this.inputs.clear();
    this.pinned = new Set(this.traceBuffers ?? []);
    this.encoder = this.device.createCommandEncoder();
  }
  input(data, length) {
    if (data?.buffer instanceof GPUBuffer) return data.buffer;
    const key = data ?? `zero-${length}`;
    if (!this.inputs.has(key)) {
      const value = this.tensor(1, length, data ?? new Float32Array(length));
      this.inputs.set(key, value);
      this.pinned.add(value.buffer);
    }
    return this.inputs.get(key).buffer;
  }
  getPipeline(code, bindingsCount, hasUniform = false) {
    let entry = this.pipelines.get(code);
    if (!entry) {
      if (hasUniform) {
        const pipeline = this.device.createComputePipeline({
          layout: "auto",
          compute: {
            module: this.device.createShaderModule({ code }),
            entryPoint: "main",
          },
        });
        entry = { pipeline, layout: pipeline.getBindGroupLayout(0) };
      } else {
        const bindLayout = this.device.createBindGroupLayout({
          entries: Array.from({ length: bindingsCount }, (_, binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: new RegExp(
                "@binding\\(\\s*" +
                  binding +
                  "\\s*\\)[^;]*var<\\s*storage\\s*,\\s*read_write\\s*>",
              ).test(code)
                ? "storage"
                : "read-only-storage",
            },
          })),
        });
        const pipeline = this.device.createComputePipeline({
          layout: this.device.createPipelineLayout({
            bindGroupLayouts: [bindLayout],
          }),
          compute: {
            module: this.device.createShaderModule({ code }),
            entryPoint: "main",
          },
        });
        entry = { pipeline, layout: bindLayout };
      }
      this.pipelines.set(code, entry);
    }
    return entry;
  }
  dispatchUniform(code, buffers, uniformWords, groups) {
    const entry = this.getPipeline(code, buffers.length, true);
    const uRes = this.allocUniform(uniformWords);
    const entries = buffers.map((buffer, binding) => ({
      binding,
      resource: { buffer },
    }));
    entries.push({ binding: buffers.length, resource: uRes });
    const group = this.device.createBindGroup({
      layout: entry.layout,
      entries,
    });
    this.pass ??= this.encoder.beginComputePass();
    this.pass.setPipeline(entry.pipeline);
    this.pass.setBindGroup(0, group);
    this.pass.dispatchWorkgroups(...groups);
  }
  dispatch(source, bindings, groups, workgroup = "64") {
    let finalGroups = groups;
    let code = source;
    if (!source.includes("@compute")) {
      const folded = groups.length === 1 ? fold2(groups[0]) : groups;
      finalGroups = folded;
      const idExpr =
        folded.length === 2 && folded[1] > 1
          ? "vec3<u32>(gid.x + wg.y * 4194240u, gid.y, gid.z)"
          : "gid";
      code = `${trig}\n${bindings.map((b, i) => `@group(0) @binding(${i}) var<storage, ${i === bindings.length - 1 ? "read_write" : "read"}> b${i}: array<f32>;`).join("\n")}\n@compute @workgroup_size(${workgroup}) fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) { let id = ${idExpr}; ${source} }`;
    } else if (groups.length === 1 && groups[0] > MAX_GRID) {
      finalGroups = fold2(groups[0]);
    }
    const entry = this.getPipeline(code, bindings.length, false);
    const group = this.device.createBindGroup({
      layout: entry.layout,
      entries: bindings.map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
    this.pass ??= this.encoder.beginComputePass();
    this.pass.setPipeline(entry.pipeline);
    this.pass.setBindGroup(0, group);
    this.pass.dispatchWorkgroups(...finalGroups);
  }
  element(rows, cols, inputs, expr) {
    const out = this.tensor(rows, cols),
      count = rows * cols;
    this.dispatch(
      `let i=id.x; if(i>=${count}u){return;} let r=i/${cols}u; let c=i%${cols}u; b${inputs.length}[i]=${expr};`,
      [...inputs.map((x) => x.buffer ?? x), out.buffer],
      [Math.ceil(count / 64)],
    );
    return out;
  }
  weight(name) {
    const w = this.weights[name];
    if (!w) throw new Error(`Missing weight: ${name}`);
    return w;
  }
  linear(x, prefix) {
    const w = this.weight(prefix + ".weight"),
      bias = this.weights[prefix + ".bias"],
      [n, k] = w.shape,
      m = x.rows;
    if (x.cols !== k) throw new Error(`Linear shape mismatch ${prefix}`);
    const out = this.tensor(m, n);
    const aligned =
      w.dtype === "F32" ||
      (w.dtype === "F16" && k % 2 === 0) ||
      (w.dtype === "I8" && k % 4 === 0);
    if (aligned) {
      const large = Math.ceil(m / 64) * Math.ceil(n / 128) >= 20;
      const BM = large ? 64 : 32;
      const BN = large ? 128 : 64;
      const tilesM = Math.ceil(m / BM);
      const tilesN = Math.ceil(n / BN);
      const totalTiles = tilesM * tilesN;
      const u = new Uint32Array([
        m,
        n,
        k,
        bias ? 1 : 0,
        tilesN,
        totalTiles,
        0,
        0,
      ]);
      this.dispatchUniform(
        linearTiledSource(w.dtype, large, this.hasF16),
        [
          x.buffer,
          w.buffer,
          bias?.buffer ?? this.zero,
          w.scale ?? this.zero,
          out.buffer,
        ],
        u,
        fold2(totalTiles),
      );
      return out;
    }
    const tilesM = Math.ceil(m / 32);
    const tilesN = Math.ceil(n / 32);
    const totalTiles = tilesM * tilesN;
    const u = new Uint32Array([
      m,
      n,
      k,
      bias ? 1 : 0,
      tilesN,
      totalTiles,
      0,
      0,
    ]);
    this.dispatchUniform(
      linearOddKSource(w.dtype),
      [
        x.buffer,
        w.buffer,
        bias?.buffer ?? this.zero,
        w.scale ?? this.zero,
        out.buffer,
      ],
      u,
      fold2(totalTiles),
    );
    return out;
  }
  norm(x, prefix) {
    const w =
        this.weights[prefix + ".gamma"] ?? this.weight(prefix + ".weight"),
      b = this.weights[prefix + ".beta"] ?? this.weight(prefix + ".bias");
    const out = this.tensor(x.rows, x.cols);
    const u = new Uint32Array([x.rows, x.cols, 0, 0]);
    this.dispatchUniform(
      NORM_SOURCE,
      [x.buffer, w.buffer, b.buffer, out.buffer],
      u,
      fold2(x.rows),
    );
    return out;
  }
  silu(x) {
    return this.element(x.rows, x.cols, [x], "b0[i]/(1.0+exp(-b0[i]))");
  }
  add(x, y) {
    return this.element(x.rows, x.cols, [x, y], "b0[i]+b1[i]");
  }
  mask(x, data) {
    return this.element(
      x.rows,
      x.cols,
      [x, this.input(data, x.rows)],
      "b0[i]*b1[r]",
    );
  }
  modulate(x, t, mode) {
    const offset = `(r%${t.rows}u)*${t.cols}u+c`;
    return this.element(
      x.rows,
      x.cols,
      [x, t],
      mode === "gate"
        ? `b0[i]*(b1[${offset}+${2 * x.cols}u]+1.0)`
        : `b0[i]*(b1[${offset}]+1.0)+b1[${offset}+${x.cols}u]`,
    );
  }
  noise(input, d) {
    const out = this.tensor(input.seqMask.length, d),
      freq = constants.noiseFrequencies[d].map(scalar).join(",");
    this.dispatch(
      `let i=id.x;if(i>=${input.seqMask.length * d}u){return;}let r=i/${d}u;let c=i%${d}u;var frequencies=array<f32,${d / 2}>(${freq});let cs=preciseSinCos(0.25*log(max(b0[r],0.000001))*frequencies[c%${d / 2}u]);b1[i]=select(cs.y,cs.x,c>=${d / 2}u);`,
      [this.input(input.noiseLevel, input.seqMask.length), out.buffer],
      [Math.ceil((input.seqMask.length * d) / 64)],
    );
    return out;
  }
  features(input, config) {
    const m = config.model,
      a = m.struct_model.n_atoms,
      n = input.seqMask.length,
      sd = config.data.sigma_data;
    const concat =
        m.crop_conditional && m.conditioning_style.includes("concat"),
      hot = m.crop_conditional && m.conditioning_style.includes("hotspot");
    const channels = 6 + (concat ? 3 : 0) + (hot ? 1 : 0),
      out = this.tensor(n, a * channels);
    this.dispatch(
      `let i=id.x; if(i>=${n * a * channels}u){return;} let r=i/${a * channels}u;let c=(i%${a * channels}u)/${a}u;let atom=i%${a}u;let j=r*${a * 3}u+atom*3u;var v=0.0;
if(c<3u){let s=max(0.000001,b1[r]);v=b0[j+c]/sqrt(s*s+${scalar(sd * sd)});}else if(c<6u){v=b2[j+c-3u];}else if(c<9u){v=b3[j+c-6u]/${scalar(sd)};}else if(atom==0u || atom==1u || atom==2u || atom==4u){v=b4[r];} b5[i]=v;`,
      [
        this.input(input.coords, n * a * 3),
        this.input(input.noiseLevel, n),
        this.input(input.selfCond, n * a * 3),
        this.input(input.cropCond, n * a * 3),
        this.input(input.hotspotMask, n),
        out.buffer,
      ],
      [Math.ceil((n * a * channels) / 64)],
    );
    return out;
  }
  attention(q, kv, input, config, prefix) {
    const n = q.rows,
      d = q.cols,
      {
        n_heads: heads,
        dim_head: dh,
        position_embedding_type: position,
        position_embedding_max: max,
      } = config.model.struct_model.uvit;
    const rotary = position.includes("rotary"),
      relpos = position.includes("relative"),
      chain = position.includes("relchain");
    const queries = this.tensor(n, d),
      keys = this.tensor(n, d);
    const freq = rotary ? this.weight(prefix + "rope.freqs").buffer : this.zero;
    for (const [isQuery, out] of [
      [true, queries],
      [false, keys],
    ]) {
      const at = isQuery ? "i" : `r*${d * 2}u+c`,
        other = isQuery ? "r*" + d + "u+(c^1u)" : `r*${d * 2}u+(c^1u)`;
      this.dispatch(
        `let i=id.x;if(i>=${n * d}u){return;}let r=i/${d}u;let c=i%${d}u;let scale=${isQuery ? scalar(dh ** -0.5) : "1.0"}; let x=b0[${at}]*scale;
${rotary ? `let angle=b1[r]*b3[(c%${dh}u)/2u]+${input.chainIndex ? "b2[r]*b3[0]" : "0.0*b2[r]"};let other=b0[${other}]*scale;let cs=preciseSinCos(angle); b4[i]=x*cs.y+select(-other,other,c%2u==1u)*cs.x;` : "b4[i]=x+0.0*(b1[r]+b2[r]+b3[0]);"}`,
        [
          isQuery ? q.buffer : kv.buffer,
          this.input(input.residueIndex, n),
          this.input(input.chainIndex, n),
          freq,
          out.buffer,
        ],
        [Math.ceil((n * d) / 64)],
      );
    }
    const scores = this.tensor(heads * n, n),
      probs = this.tensor(heads * n, n),
      result = this.tensor(n, d);
    // Four small positional arrays share a single packed F32 buffer to remain below binding limits.
    if (!this.positional) this.positional = new Map();
    let positional = this.positional.get(prefix);
    if (!positional) {
      const size = heads * (max * 2 + 2),
        packed = new Float32Array(size * 2);
      positional = this.allocate(packed.byteLength, true);
      this.device.queue.writeBuffer(positional, 0, packed);
      this.endPass();
      for (const [kind, offset] of [
        ["relpos", 0],
        ["relchain", size * 4],
      ]) {
        const w = this.weights[prefix + kind + ".0.linear.weight"],
          b = this.weights[prefix + kind + ".0.linear.bias"];
        if (w) {
          this.encoder.copyBufferToBuffer(
            w.buffer,
            0,
            positional,
            offset,
            heads * (max * 2 + 1) * 4,
          );
          this.encoder.copyBufferToBuffer(
            b.buffer,
            0,
            positional,
            offset + heads * (max * 2 + 1) * 4,
            heads * 4,
          );
        }
      }
      this.positional.set(prefix, positional);
    }
    const bins = max * 2 + 1,
      span = heads * (bins + 1);
    this.dispatch(
      `let idx=id.x;if(idx>=${heads * n * n}u){return;}let j=idx%${n}u;let i=(idx/${n}u)%${n}u;let h=idx/${n * n}u;var sum=0.0;for(var c=0u;c<${dh}u;c++){sum+=b0[i*${d}u+h*${dh}u+c]*b1[j*${d}u+h*${dh}u+c];}var bias=0.0;
${chain ? `let ci=select(0u,1u,b4[i]!=b4[j])+${max}u;bias=b5[${span}u+h*${bins}u+ci]+b5[${span + heads * bins}u+h];` : ""}
${relpos ? `let ri=u32(clamp(b3[i]-b3[j],-${max}.0,${max}.0)+${max}.0);bias+=b5[h*${bins}u+ri]+b5[${heads * bins}u+h];${chain ? "bias*=2.0;" : ""}` : ""}
b6[idx]=sum+bias-(1.0-b2[i]*b2[j])*1000000.0+0.0*(b3[0]+b4[0]+b5[0]);`,
      [
        queries.buffer,
        keys.buffer,
        this.input(input.seqMask, n),
        this.input(input.residueIndex, n),
        this.input(input.chainIndex, n),
        positional,
        scores.buffer,
      ],
      [Math.ceil((heads * n * n) / 64)],
    );
    this.dispatchUniform(
      SOFTMAX_SOURCE,
      [scores.buffer, probs.buffer],
      new Uint32Array([heads * n, n, 0, 0]),
      fold2(heads * n),
    );
    this.dispatch(
      `let idx=id.x;if(idx>=${n * d}u){return;}let i=idx/${d}u;let h=(idx%${d}u)/${dh}u;let c=idx%${dh}u;var value=0.0;for(var j=0u;j<${n}u;j++){value+=b0[(h*${n}u+i)*${n}u+j]*b1[j*${d * 2}u+${d}u+h*${dh}u+c];}b2[idx]=value;`,
      [probs.buffer, kv.buffer, result.buffer],
      [Math.ceil((n * d) / 64)],
    );
    return {
      ...result,
      attentionCache: { queries, keys, probs, kv, input, config, prefix },
    };
  }
  output(patch, input, config) {
    const a = config.model.struct_model.n_atoms,
      sd = config.data.sigma_data,
      n = patch.rows;
    return this.element(
      n,
      a * 3,
      [
        patch,
        this.input(input.coords, n * a * 3),
        this.input(input.noiseLevel, n),
        this.input(input.seqMask, n),
      ],
      `(b0[r*${a * 3}u+(c%3u)*${a}u+c/3u]*(max(b2[r],0.000001)*${scalar(sd)}/sqrt(max(b2[r],0.000001)*max(b2[r],0.000001)+${scalar(sd * sd)}))+b1[i]*(${scalar(sd * sd)}/(max(b2[r],0.000001)*max(b2[r],0.000001)+${scalar(sd * sd)})))*b3[r]`,
    );
  }
  beginBlock() {
    this.blockBuffers = new Set();
    for (const [buffer, info] of this.buffers) {
      if (!info.pooled) this.blockBuffers.add(buffer);
    }
  }
  endBlock(keep) {
    const retained = new Set(keep.map((t) => t.buffer));
    for (const [buffer, info] of this.buffers)
      if (
        !info.resident &&
        !info.uniformSlab &&
        !info.uploaded &&
        !info.pooled &&
        !this.blockBuffers.has(buffer) &&
        !retained.has(buffer) &&
        !this.pinned.has(buffer)
      ) {
        info.pooled = true;
        if (!this.pool.has(info.size)) this.pool.set(info.size, []);
        this.pool.get(info.size).push(buffer);
      }
  }
  async readback(tensors) {
    const entries = Object.entries(tensors);
    const slices = [];
    let totalBytes = 0;
    for (const [key, tensor] of entries) {
      const bytes = tensor.rows * tensor.cols * 4;
      slices.push({ key, tensor, off: totalBytes, bytes });
      totalBytes += Math.max(4, Math.ceil(bytes / 4) * 4);
    }
    totalBytes = Math.max(4, totalBytes);
    const staging = this.device.createBuffer({
      size: totalBytes,
      usage: U().MAP_READ | U().COPY_DST,
    });
    this.buffers.set(staging, { size: totalBytes, resident: false });
    try {
      this.endPass();
      this.flushUniforms();
      for (const s of slices) {
        if (s.bytes > 0) {
          this.encoder.copyBufferToBuffer(
            s.tensor.buffer,
            0,
            staging,
            s.off,
            s.bytes,
          );
        }
      }
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder = null;
      await staging.mapAsync(GPUMapMode.READ);
      const mapped = staging.getMappedRange();
      const result = {};
      for (const s of slices) {
        result[s.key] = new Float32Array(mapped.slice(s.off, s.off + s.bytes));
      }
      staging.unmap();
      return result;
    } finally {
      staging.destroy();
      this.buffers.delete(staging);
    }
  }
  cleanup() {
    this.endPass();
    if (this.encoder) {
      try {
        this.flushUniforms();
        this.device.queue.submit([this.encoder.finish()]);
      } catch {}
      this.encoder = null;
    }
    for (const [buffer, info] of this.buffers)
      if (!info.resident) {
        buffer.destroy();
        this.buffers.delete(buffer);
      }
    this.uniformSlabs.length = 0;
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
    this.pool.clear();
    this.pinned.clear();
    this.inputs?.clear();
  }
  memoryUsage() {
    let bytes = 0,
      buffers = 0,
      transientBuffers = 0,
      transientBytes = 0;
    for (const info of this.buffers.values()) {
      buffers++;
      bytes += info.size;
      if (!info.resident) {
        transientBuffers++;
        transientBytes += info.size;
      }
    }
    return { buffers, bytes, transientBuffers, transientBytes };
  }
  destroy() {
    this.endPass();
    this.encoder = null;
    for (const buffer of this.buffers.keys()) buffer.destroy();
    this.buffers.clear();
    this.uniformSlabs.length = 0;
    this.uniformSlabIdx = 0;
    this.uniformSlotIdx = 0;
    this.pipelines.clear();
    this.pool.clear();
    this.pinned.clear();
    this.positional = null;
  }
}
Object.assign(GPUOps.prototype, graphOps);
