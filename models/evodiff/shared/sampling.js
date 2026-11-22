import { normalizeTokens, decode, MASK } from "./tokenizer.js";

export function random(seed = 0) {
  if (!Number.isInteger(seed)) throw new TypeError("seed must be an integer");
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function prepareGeneration({
  length,
  fixedSequence,
  maskPositions,
  seed = 0,
  order,
  uniforms,
}) {
  let tokens =
    fixedSequence === undefined
      ? new Int32Array(validLength(length)).fill(MASK)
      : normalizeTokens(fixedSequence);
  validLength(tokens.length);
  if (length !== undefined && length !== tokens.length)
    throw new RangeError("Length mismatch");
  if (maskPositions !== undefined) {
    const set = new Set(maskPositions);
    if (
      set.size !== maskPositions.length ||
      [...set].some((i) => !Number.isInteger(i) || i < 0 || i >= tokens.length)
    )
      throw new RangeError("Invalid mask positions");
    for (const i of set) tokens[i] = MASK;
  }
  const positions = Array.from(tokens.keys()).filter((i) => tokens[i] === MASK),
    rng = random(seed);
  let permutation = order === undefined ? positions.slice() : Array.from(order);
  if (order === undefined)
    for (let i = permutation.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [permutation[i], permutation[j]] = [permutation[j], permutation[i]];
    }
  if (
    permutation.length !== positions.length ||
    new Set(permutation).size !== positions.length ||
    permutation.some((i) => !positions.includes(i))
  )
    throw new RangeError(
      "Order must permute every masked position exactly once",
    );
  const draws =
    uniforms === undefined
      ? permutation.map(() => rng())
      : Array.from(uniforms);
  if (
    draws.length !== permutation.length ||
    draws.some((u) => !Number.isFinite(u) || u < 0 || u >= 1)
  )
    throw new RangeError("Uniforms must be in [0,1), one per step");
  return { tokens, order: permutation, uniforms: draws };
}
export function validLength(n) {
  if (!Number.isSafeInteger(n) || n < 1)
    throw new RangeError("Length must be a positive integer");
  return n;
}
export function choose(logits, position, u) {
  let max = -Infinity,
    sum = 0;
  const p = new Float64Array(21),
    offset = position * 31;
  for (let i = 0; i < 21; i++) max = Math.max(max, logits[offset + i]);
  for (let i = 0; i < 21; i++) {
    p[i] = Math.exp(logits[offset + i] - max);
    sum += p[i];
  }
  let cumulative = 0;
  for (let i = 0; i < 21; i++) {
    cumulative += p[i] / sum;
    if (u < cumulative || i === 20) return i;
  }
}
export async function generate(forward, options) {
  const start = performance.now(),
    plan = prepareGeneration(options),
    steps = [];
  for (let i = 0; i < plan.order.length; i++) {
    options.signal?.throwIfAborted();
    const { logits } = await forward(plan.tokens, { signal: options.signal });
    const position = plan.order[i],
      uniform = plan.uniforms[i],
      token = choose(logits, position, uniform);
    plan.tokens[position] = token;
    steps.push({ position, uniform, token, logits });
    options.onProgress?.({
      phase: "sampling",
      completed: i + 1,
      total: plan.order.length,
      position,
      token,
    });
    // Allow cancellation from a DOM event or Worker message between steps.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  options.signal?.throwIfAborted();
  return {
    sequence: decode(plan.tokens),
    tokens: plan.tokens,
    steps,
    timingMs: performance.now() - start,
  };
}
