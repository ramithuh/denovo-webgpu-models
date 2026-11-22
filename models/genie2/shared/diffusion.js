const f = Math.fround;
export function diffusionSchedule(steps = 1000) {
  if (!Number.isSafeInteger(steps) || steps < 1 || steps > 100000)
    throw new RangeError("steps must be an integer in [1, 100000]");
  const beta = new Float32Array(steps + 1),
    alpha = new Float32Array(steps + 1),
    cum = new Float32Array(steps + 1),
    cos2 = new Float32Array(steps + 1);
  alpha[0] = cum[0] = 1;
  for (let t = 0; t <= steps; t++) {
    const c = f(Math.cos(f(f(f(t / (steps + 1)) * f(Math.PI)) * 0.5)));
    cos2[t] = f(c * c);
  }
  let prod = 1;
  for (let t = 1; t <= steps; t++) {
    beta[t] = Math.min(0.999, Math.max(0, f(1 - f(cos2[t] / cos2[t - 1]))));
    alpha[t] = f(1 - beta[t]);
    prod *= alpha[t];
    cum[t] = prod;
  }
  return { beta, alpha, cum };
}
export function seededRandom(seed = 1) {
  let x = seed >>> 0;
  return () => {
    x = (x + 0x6d2b79f5) >>> 0;
    let z = x;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
}
export function gaussianNoise(size, rng = Math.random) {
  const a = new Float32Array(size);
  for (let i = 0; i < size; i += 2) {
    const r = Math.sqrt(-2 * Math.log(Math.max(Number.MIN_VALUE, rng()))),
      theta = 2 * Math.PI * rng();
    a[i] = r * Math.cos(theta);
    if (i + 1 < size) a[i + 1] = r * Math.sin(theta);
  }
  return a;
}
export function reverseStep(o, x, z, noise, mask, schedule, t, scale) {
  const { alpha, cum, beta } = schedule,
    L = x.shape[0];
  const wz = f(f(1 - alpha[t]) / f(Math.sqrt(f(1 - cum[t])))),
    recip = f(1 / f(Math.sqrt(alpha[t])));
  let mean = o.mul(
    o.mul(o.sub(x, o.mul(z, wz)), recip),
    o.reshape(mask, [L, 1]),
  );
  if (t > 1)
    mean = o.mul(
      o.add(mean, o.mul(noise, f(f(scale) * f(Math.sqrt(beta[t]))))),
      o.reshape(mask, [L, 1]),
    );
  return mean;
}
