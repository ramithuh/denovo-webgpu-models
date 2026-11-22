// Correctly rounded F32 fused multiply-add using exact F64 products and an
// error-free sum. Ordinary fround(a*b+c) can double-round at F32 midpoints.
const value = new Float32Array(1),
  bits = new Uint32Array(value.buffer);
export function fmaF32(a, b, c) {
  const product = a * b,
    sum = product + c,
    rounded = Math.fround(sum);
  const error =
    Math.abs(product) >= Math.abs(c) ? product - sum + c : c - sum + product;
  if (error * (sum - rounded) > 0) {
    value[0] = rounded;
    if (rounded === 0) bits[0] = error > 0 ? 1 : 0x80000001;
    else bits[0] += error > 0 === rounded >= 0 ? 1 : -1;
    const next = value[0];
    if (sum === (rounded + next) * 0.5) return next;
  }
  return rounded;
}
