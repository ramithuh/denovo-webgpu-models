import { fmaF32 } from "./fma.js";
// Float32 exponential polynomial used by SLEEF 3.6.1 (Boost Software License).
// Reduction and FMA ordering follow PyTorch's ARM64 vector exp implementation.
export function expF32(x) {
  const f = Math.fround;
  if (x < -104) return 0;
  if (x > 88.72283905206835) return Infinity;
  const v = f(x * f(1.4426950408889634)),
    q = Math.round(v),
    s0 = fmaF32(q, f(-0.693145751953125), x),
    s = fmaF32(q, f(-1.428606765330187e-6), s0);
  let u = f(0.00019852761761285365);
  for (const c of [
    0.0013930435525253415, 0.008333360776305199, 0.041666485369205475,
    0.1666666716337204, 0.5,
  ])
    u = fmaF32(u, s, f(c));
  u = f(1 + fmaF32(f(s * s), u, s));
  return f(u * 2 ** q);
}
// PyTorch's ARM NEON erf uses the Arm u20 exponential, not ordinary expf.
export function erfF32(a) {
  const f = Math.fround, x = Math.abs(a),
    t = f(1 / fmaF32(f(0.3275911), x, 1));
  let p = fmaF32(f(1.061405429), t, f(-1.453152027));
  for (const c of [1.421413741, -0.284496736, 0.254829592])
    p = fmaF32(p, t, f(c));
  const z = f(-f(a * a));
  let e;
  if (Math.abs(z) > 87.336) e = expF32(z);
  else {
    const v = f(z * f(1.4426950216293335)),
      n = Math.sign(v) * Math.floor(Math.abs(v) + 0.5),
      r0 = fmaF32(-n, f(0.693145751953125), z),
      r = fmaF32(-n, f(1.428606765330187e-6), r0),
      r2 = f(r * r), scale = f(2 ** n),
      c = fmaF32(r, f(0.008247390389442444), f(0.04189976677298546));
    let q = fmaF32(r, f(0.16668395698070526), f(0.4999912679195404));
    q = fmaF32(c, r2, q);
    const poly = fmaF32(q, r2, f(f(0.9999994039535522) * r));
    e = fmaF32(scale, poly, scale);
  }
  const value = fmaF32(f(t * -e), p, 1);
  return a < 0 || Object.is(a, -0) ? -value : value;
}
export const numericWGSL = `
fn rfExp(x:f32)->f32 {
 if(x < -104.0){return 0.0;}
 let q=round(x*1.4426950408889634);
 let s0=fma(q,-0.693145751953125,x);
 let s=fma(q,-1.428606765330187e-6,s0);
 var u=0.00019852761761285365;
 u=fma(u,s,0.0013930435525253415);
 u=fma(u,s,0.008333360776305199);
 u=fma(u,s,0.041666485369205475);
 u=fma(u,s,0.1666666716337204);
 u=fma(u,s,0.5);
 u=1.0+fma(s*s,u,s);
 return ldexp(u,i32(q));
}
`;
export const trigWGSL = `
// FMA range reduction before evaluating sine/cosine on [-pi/4, pi/4].
// Native GPU trigonometry loses accuracy at the larger ESM rotary angles.
fn rfSinCos(x:f32)->vec2<f32>{
 let q=round(x*0.6366197723675814);
 let r0=fma(-q,1.5707963705062866,x);
 let r1=fma(-q,-4.371138828673793e-8,r0);
 let r=fma(-q,-1.715124499442883e-15,r1);let z=r*r;
 var p=1.6059043836821613e-10;
 p=fma(p,z,-2.505210838544172e-8);p=fma(p,z,2.7557319223985893e-6);
 p=fma(p,z,-1.984126984126984e-4);p=fma(p,z,8.333333333333333e-3);p=fma(p,z,-1.6666666666666666e-1);
 let s=fma(r*z,p,r);
 p=-1.1470745597729725e-11;p=fma(p,z,2.08767569878681e-9);p=fma(p,z,-2.755731922398589e-7);
 p=fma(p,z,2.48015873015873e-5);p=fma(p,z,-1.388888888888889e-3);p=fma(p,z,4.1666666666666664e-2);p=fma(p,z,-0.5);
 let c=fma(z,p,1.0);let k=bitcast<u32>(i32(q))&3u;
 if(k==0u){return vec2<f32>(s,c);}if(k==1u){return vec2<f32>(c,-s);}
 if(k==2u){return vec2<f32>(-s,-c);}return vec2<f32>(-c,s);
}
fn rfSin(x:f32)->f32{return rfSinCos(x).x;}
fn rfCos(x:f32)->f32{return rfSinCos(x).y;}
`;
export const erfWGSL = `
// PyTorch ARM NEON vec128_float_neon.h: erf() and vexpq_f32_u20().
// The coefficient order and intermediate F32 rounding are intentional.
fn erfExp(x:f32)->f32 {
 if(abs(x)>87.336){return rfExp(x);}
 let z=x*1.4426950216293335;
 let n=sign(z)*floor(abs(z)+0.5);
 let r0=fma(-n,0.693145751953125,x);
 let r=fma(-n,1.428606765330187e-6,r0);
 let scale=bitcast<f32>(bitcast<u32>(i32(n)*8388608)+0x3f800000u);
 let r2=r*r;
 var p=fma(r,0.008247390389442444,0.04189976677298546);
 var q=fma(r,0.16668395698070526,0.4999912679195404);
 q=fma(p,r2,q);p=0.9999994039535522*r;
 let poly=fma(q,r2,p);
 return fma(scale,poly,scale);
}
fn rfErf(a:f32)->f32 {
 let x=abs(a);let t=rfdiv(1.0,fma(0.3275911,x,1.0));
 var p=fma(1.061405429,t,-1.453152027);
 p=fma(p,t,1.421413741);p=fma(p,t,-0.284496736);p=fma(p,t,0.254829592);
 let e=erfExp(-(a*a));let v=fma(t*(-e),p,1.0);
 return bitcast<f32>(bitcast<u32>(v)^(bitcast<u32>(a)&0x80000000u));
}
`;

// Scalar expf table polynomial used by DGL's native CPU edge softmax.
// Arm optimized-routines, MIT. The learned tensors remain float32.
export const EXP_TABLE = Array.from({ length: 32 }, (_, i) => 2 ** (i / 32));
export const EXP_COEFFICIENTS = [
  1.6938359250920212e-6, 0.00023459809789509004, 0.021660849396613134,
];
export function scalarExpF32(x) {
  if (x < -104) return 0;
  if (x > 88.72283905206835) return Infinity;
  const z = (32 / Math.LN2) * x,
    q = Math.round(z),
    r = z - q,
    s = EXP_TABLE[((q % 32) + 32) % 32] * 2 ** Math.floor(q / 32),
    c = EXP_COEFFICIENTS,
    p = c[0] * r + c[1],
    r2 = r * r,
    y = c[2] * r + 1;
  return Math.fround((p * r2 + y) * s);
}
