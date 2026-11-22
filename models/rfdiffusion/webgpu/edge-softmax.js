// DGL's CPU edge softmax calls scalar expf, whose table polynomial is evaluated
// with greater precision than the learned tensor operations. Two F32 components
// preserve that evaluation in WGSL. Scratch writes prevent contraction from
// discarding the low component. Constants follow Arm optimized-routines expf.
const f = Math.fround;
import {
  EXP_TABLE as TABLE,
  EXP_COEFFICIENTS as COEFFICIENTS,
} from "../shared/numeric.js";
function pair(o, shape, value) {
  const hi = f(value);
  return { hi: o.constant(shape, hi), lo: o.constant(shape, f(value - hi)) };
}
function mul(o, a, b) {
  const hi = o.mul(a.hi, b.hi),
    lo = o.kernel(
      hi.shape,
      [a.hi, b.hi, hi, a.lo, b.lo],
      "",
      `let error=fma(a0[i],a1[i],-a2[i]);o[i]=((error+a0[i]*a4[i])+a3[i]*a1[i])+a3[i]*a4[i];`,
    );
  return { hi, lo };
}
function add(o, a, b) {
  const hi = o.add(a.hi, b.hi),
    bv = o.sub(hi, a.hi),
    lo = o.kernel(
      hi.shape,
      [a.hi, b.hi, hi, bv, a.lo, b.lo],
      "",
      "o[i]=((a0[i]-(a2[i]-a3[i]))+(a1[i]-a3[i]))+(a4[i]+a5[i]);",
    );
  return { hi, lo };
}
export function scalarExp(o, input) {
  const x = o.unary(input, "max($0,-104.0)"),
    shape = x.shape,
    z = mul(o, { hi: x, lo: o.constant(shape) }, pair(o, shape, 32 / Math.LN2));
  const q = o.kernel(
      shape,
      [z.hi, z.lo],
      "",
      "let h=a0[i];var q=floor(h+0.5);if(h+0.5==q && a1[i]<0.0){q-=1.0;}o[i]=q;",
    ),
    r = { hi: o.sub(z.hi, q), lo: z.lo };
  const tableHi = o.tensor([32], TABLE.map(f)),
    tableLo = o.tensor(
      [32],
      TABLE.map((v) => f(v - f(v))),
    );
  const scale = (table) =>
    o.kernel(
      shape,
      [q, table],
      "",
      "let q=i32(a0[i]);let j=u32((q%32+32)%32);let exponent=i32(floor(f32(q)/32.0));o[i]=ldexp(a1[j],exponent);",
    );
  const s = { hi: scale(tableHi), lo: scale(tableLo) },
    p = add(
      o,
      mul(o, pair(o, shape, COEFFICIENTS[0]), r),
      pair(o, shape, COEFFICIENTS[1]),
    ),
    r2 = mul(o, r, r),
    linear = add(
      o,
      mul(o, pair(o, shape, COEFFICIENTS[2]), r),
      pair(o, shape, 1),
    ),
    poly = add(o, mul(o, p, r2), linear),
    answer = mul(o, poly, s);
  return o.kernel(
    shape,
    [input, answer.hi, answer.lo],
    "",
    "o[i]=select(a1[i]+a2[i],0.0,a0[i]<=-104.0);",
  );
}
export function edgeSoftmax(o, x, axis) {
  if (axis !== 0) throw Error("Edge softmax expects incoming edges on axis 0");
  const n = x.shape[0],
    shape = x.shape.slice(1),
    inner = x.size / n,
    max = o.kernel(
      shape,
      [x],
      "",
      `var m=-3.402823e38;for(var j=0u;j<${n}u;j++){m=max(m,a0[j*${inner}u+i]);}o[i]=m;`,
    ),
    exp = scalarExp(o, o.sub(x, o.reshape(max, [1, ...shape]))),
    sum = o.kernel(
      shape,
      [exp],
      "",
      `var s=0.0;for(var j=0u;j<${n}u;j++){s+=a0[j*${inner}u+i];}o[i]=s;`,
    );
  return o.div(exp, o.reshape(sum, [1, ...shape]));
}
