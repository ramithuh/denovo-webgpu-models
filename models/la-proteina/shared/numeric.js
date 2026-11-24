import { fmaF32 } from "./fma.js";
// Float32 exponential polynomial used by SLEEF 3.8.0 (Boost Software License).
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

function dfAdd2F32(a, b) {
  const f = Math.fround;
  const s = fmaF32(1, a, b);
  const v = fmaF32(-1, a, s);
  return [s, fmaF32(1, fmaF32(-1, fmaF32(-1, v, s), a), fmaF32(-1, v, b))];
}
function dfAddF32(a, b) {
  const s = fmaF32(1, a[0], b);
  return [s, fmaF32(1, fmaF32(1, fmaF32(-1, s, a[0]), b), a[1])];
}
function dfMulF32(a, b) {
  const p = fmaF32(a[0], b[0], 0);
  return [p, fmaF32(a[0], b[1], fmaF32(a[1], b[0], fmaF32(a[0], b[0], -p)))];
}
function reducedSinF32(t) {
  const f = Math.fround;
  const square = fmaF32(t[0], t[0], 0);
  const s = [square, fmaF32(f(t[0] + t[0]), t[1], fmaF32(t[0], t[0], -square))];
  let u = f(2.6083159809786593541503e-6);
  u = fmaF32(u, s[0], f(-0.0001981069071916863322258));
  u = fmaF32(u, s[0], f(0.00833307858556509017944336));
  const c = f(-0.166666597127914428710938);
  const p = fmaF32(u, s[0], 0);
  const cp = fmaF32(1, c, p);
  const a = [cp, fmaF32(1, fmaF32(-1, cp, c), p)];
  const y = dfMulF32(a, s);
  const yh = fmaF32(1, y[0], 1);
  const x = [yh, fmaF32(1, fmaF32(1, fmaF32(-1, yh, 1), y[0]), y[1])];
  return fmaF32(t[0], x[0], fmaF32(t[1], x[0], f(t[0] * x[1])));
}
function trigReduceF32(x, q, large) {
  const f = Math.fround;
  const a = fmaF32(q, f(-1.57073974609375), 0);
  let t = dfAdd2F32(x, a);
  t[1] = fmaF32(1, t[1], fmaF32(q, f(-1.57073974609375), -a));
  const y = f(q * f(-0.00005657970905303955));
  const next = dfAdd2F32(t[0], y);
  t = [next[0], fmaF32(1, next[1], t[1])];
  if (large) t[1] = f(t[1] + fmaF32(q, f(-0.00005657970905303955), -y));
  t = dfAddF32(t, f(q * f(-9.92093629470503e-10)));
  if (large) {
    const h = fmaF32(1, t[0], t[1]);
    return [h, fmaF32(1, fmaF32(-1, h, t[0]), t[1])];
  }
  return t;
}
export function sinCosF32(x, large = Math.abs(x) >= 125) {
  const f = Math.fround;
  const invPi = f(0.3183098861837907);
  const roundEven = (v) => {
    const r = Math.round(v);
    return Math.abs(v - r) === 0.5 ? 2 * Math.round(v * 0.5) : r;
  };
  const qs = f(roundEven(f(x * invPi)));
  const qc = f(2 * roundEven(fmaF32(x, invPi, -0.5)) + 1);
  const sn = reducedSinF32(trigReduceF32(x, f(2 * qs), large));
  const cs = reducedSinF32(trigReduceF32(x, qc, large));
  return [(qs | 0) & 1 ? -sn : sn, ((qc | 0) & 2) === 0 ? -cs : cs];
}
export const numericWGSL = `
// SLEEF u10 sine polynomial with double-float reduction on the time
// embedding domain |x| <= 2000. Float pairs preserve cancellation near pi.
fn rfDFAdd2(a:f32,b:f32)->vec2<f32>{
 let s=fma(1.0,a,b);let v=fma(-1.0,a,s);return vec2<f32>(s,fma(1.0,fma(-1.0,fma(-1.0,v,s),a),fma(-1.0,v,b)));
}
fn rfDFAdd(a:vec2<f32>,b:f32)->vec2<f32>{
 let s=fma(1.0,a.x,b);return vec2<f32>(s,fma(1.0,fma(1.0,fma(-1.0,s,a.x),b),a.y));
}
fn rfDFMul(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
 let p=fma(a.x,b.x,0.0);return vec2<f32>(p,fma(a.x,b.y,fma(a.y,b.x,fma(a.x,b.x,-p))));
}
fn rfReducedSin(t:vec2<f32>)->f32{
 let square=fma(t.x,t.x,0.0);
 let s=vec2<f32>(square,fma(t.x+t.x,t.y,fma(t.x,t.x,-square)));
 var u=2.6083159809786593541503e-6;
 u=fma(u,s.x,-0.0001981069071916863322258);
 u=fma(u,s.x,0.00833307858556509017944336);
 let c:f32=-0.166666597127914428710938;let p=fma(u,s.x,0.0);
 let cp=fma(1.0,c,p);let a=vec2<f32>(cp,fma(1.0,fma(-1.0,cp,c),p));
 let y=rfDFMul(a,s);
 let yh=fma(1.0,y.x,1.0);let x=vec2<f32>(yh,fma(1.0,fma(1.0,fma(-1.0,yh,1.0),y.x),y.y));
 return fma(t.x,x.x,fma(t.y,x.x,t.x*x.y));
}
fn rfTrigReduce(x:f32,q:f32,large:bool)->vec2<f32>{
 let a=fma(q,-1.57073974609375,0.0);
 var t=rfDFAdd2(x,a);
 t.y=fma(1.0,t.y,fma(q,-1.57073974609375,-a));
 let y=q*(-0.00005657970905303955);
 let next=rfDFAdd2(t.x,y);
 t=vec2<f32>(next.x,fma(1.0,next.y,t.y));
 if(large){t.y=t.y+fma(q,-0.00005657970905303955,-y);}
 t=rfDFAdd(t,q*(-9.92093629470503e-10));
 if(large){let h=fma(1.0,t.x,t.y);return vec2<f32>(h,fma(1.0,fma(-1.0,h,t.x),t.y));}
 return t;
}
fn rfSinCos(x:f32,large:bool)->vec2<f32>{
 let qs=round(x*0.3183098861837907);
 let qc=2.0*round(fma(x,0.3183098861837907,-0.5))+1.0;
 let sn=rfReducedSin(rfTrigReduce(x,2.0*qs,large));
 let cs=rfReducedSin(rfTrigReduce(x,qc,large));
 return vec2<f32>(select(sn,-sn,(i32(qs)&1)==1),select(cs,-cs,(i32(qc)&2)==0));
}
fn rfSin(x:f32)->f32{return rfSinCos(x,abs(x)>=125.0).x;}
fn rfCos(x:f32)->f32{return rfSinCos(x,abs(x)>=125.0).y;}
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
