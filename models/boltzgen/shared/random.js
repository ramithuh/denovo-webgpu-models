export function randomSource(seed = 42) {
  if (!Number.isSafeInteger(seed)) throw Error("Seed must be an integer");
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () =>
    Math.fround(
      Math.sqrt(-2 * Math.log(Math.max(uniform(), 2 ** -32))) *
        Math.cos(2 * Math.PI * uniform()),
    );
  return { uniform, normal };
}
export function randomInputs(atoms, numSteps, seed = 42) {
  const rng = randomSource(seed),
    out = new Map();
  let id = 0;
  const add = (shape) => {
    const data = Float32Array.from(
      { length: shape.reduce((a, b) => a * b, 1) },
      rng.normal,
    );
    out.set(`rng:${id++}`, { shape, data });
  };
  add([1, atoms, 3]);
  for (let i = 0; i < numSteps; i++) {
    add([1, 4]);
    add([1, 1, 3]);
    add([1, atoms, 3]);
  }
  return out;
}
export function sigmaSchedule(numSteps) {
  if (!Number.isSafeInteger(numSteps) || numSteps < 1)
    throw Error("numSteps must be a positive integer");
  const f = Math.fround,
    hi = 160 ** (1 / 7),
    lo = 0.0004 ** (1 / 7);
  return [
    ...Array.from(
      { length: numSteps },
      (_, i) =>
        f(f(f(f(i / Math.max(1, numSteps - 1)) * f(lo - hi)) + f(hi)) ** 7) *
        16,
    ),
    0,
  ];
}
