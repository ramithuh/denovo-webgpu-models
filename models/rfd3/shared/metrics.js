export function metrics(actual, expected) {
  if (actual.length !== expected.length || !actual.length) throw new Error('Metric shape mismatch');
  let maxAbsDiff = 0, square = 0, dot = 0, aa = 0, bb = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i], b = expected[i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) throw new Error('Nonfinite comparison');
    const diff = Math.abs(a - b);
    maxAbsDiff = Math.max(maxAbsDiff, diff); square += diff * diff;
    dot += a * b; aa += a * a; bb += b * b;
  }
  return { maxAbsDiff, rmse: Math.sqrt(square / actual.length), cosineSim: aa && bb ? dot / Math.sqrt(aa * bb) : aa === bb ? 1 : 0 };
}
