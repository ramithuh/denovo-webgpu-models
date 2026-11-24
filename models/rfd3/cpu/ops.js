import { decodeWeight } from '../shared/weights.js';
// Explicit FP32 product and accumulation boundaries. This reference is intentionally
// scalar and prioritizes debuggability over speed; GPU has its own WGSL arithmetic.
export function linear(input, weight, bias) {
  const [n, k] = weight.shape;
  if (weight.shape.length !== 2 || input.length % k || bias && bias.length !== n)
    throw new Error('Invalid linear dimensions');
  const w = weight.data ?? decodeWeight(weight), m = input.length / k, output = new Float32Array(m * n);
  for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) {
    let sum = 0;
    for (let i = 0; i < k; i++) sum = Math.fround(sum + Math.fround(input[r * k + i] * w[c * k + i]));
    output[r * n + c] = sum + (bias?.[c] ?? 0);
  }
  return output;
}
export function rmsNorm(input, channels, gain, epsilon = 2 ** -23) {
  if (input.length % channels || gain && gain.length !== channels) throw new Error('Invalid RMSNorm dimensions');
  const output = new Float32Array(input.length);
  for (let base = 0; base < input.length; base += channels) {
    let square = 0;
    for (let c = 0; c < channels; c++) square = Math.fround(square + Math.fround(input[base + c] ** 2));
    const scale = Math.fround(1 / Math.sqrt(Math.fround(square / channels + epsilon)));
    for (let c = 0; c < channels; c++) output[base + c] = Math.fround(input[base + c] * scale) * (gain?.[c] ?? 1);
  }
  return output;
}
export function transition(input, { norm, w1, w2, w3 }) {
  const normalized = rmsNorm(input, norm.length, norm);
  const a = linear(normalized, w1), b = linear(normalized, w2);
  for (let i = 0; i < a.length; i++) a[i] = Math.fround(a[i] / (1 + Math.exp(-a[i]))) * b[i];
  return linear(a, w3);
}
