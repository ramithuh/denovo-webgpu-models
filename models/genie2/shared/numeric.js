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
